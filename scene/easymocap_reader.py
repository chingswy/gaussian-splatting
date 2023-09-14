import os
import numpy as np
import cv2
from tqdm import tqdm
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from .dataset_readers import CameraInfo, SceneInfo, getNerfppNorm, read_points3D_binary, storePly, fetchPly

def readEasyMocapCameras(cameras, images_folder, scale=0.5, frame_start=0):
    cam_infos = []
    for idx, (key, camera) in enumerate(cameras.items()):
        # the exact output you're looking for:
        print("Reading camera {}/{}: {}".format(idx+1, len(cameras), key))
        height, width = 1024, 1024

        uid = idx
         # 注意：这里转置了，他的代码用的就是转置的
        R = camera['R'].T
        T = camera['T'].reshape(3,)
        K = camera['K']

        image_path = os.path.join(images_folder, 'images', key, f'{frame_start:06d}.jpg')
        mask_path = os.path.join(images_folder, 'masks', key, f'{frame_start:06d}.png')
        if True:
            image_path_new = os.path.join(images_folder, 'images_scale{}'.format(scale), key, f'{frame_start:06d}.jpg')
            mask_path_new = os.path.join(images_folder, 'masks_scale{}'.format(scale), key, f'{frame_start:06d}.png')
            img = cv2.imread(image_path)
            #  If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. 
            # So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. This function also returns an image ROI which can be used to crop the result.
            newK, roi = cv2.getOptimalNewCameraMatrix(K, camera['dist'], 
                            (width, height), 0, (width,height), centerPrincipalPoint=True)
            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(K, camera['dist'], None, newK, (width, height), 5)
            for frame in tqdm(range(frame_start, frame_start + 1, 1)):
                image_path = os.path.join(images_folder, 'images', key, f'{frame:06d}.jpg')
                dump_path = os.path.join(images_folder, 'images_scale{}'.format(scale), key, f'{frame:06d}.jpg')
                if not os.path.exists(dump_path):
                    img = cv2.imread(image_path)
                    # dst = cv2.undistort(img, K, camera['dist'], None, newK)
                    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                    # crop, resize, dump
                    dst = cv2.resize(dst, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                    cv2.imwrite(dump_path, dst)
                mask_path = os.path.join(images_folder, 'masks', key, f'{frame:06d}.png')
                dump_mask_path = os.path.join(images_folder, 'masks_scale{}'.format(scale), key, f'{frame:06d}.png')
                if not os.path.exists(dump_mask_path):
                    mask = cv2.imread(mask_path)
                    dst = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)
                    # crop, resize, dump
                    dst = cv2.resize(dst, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    os.makedirs(os.path.dirname(dump_mask_path), exist_ok=True)
                    cv2.imwrite(dump_mask_path, dst)
            height = int(height * scale)
            width = int(width * scale)
            K[:2, :] = K[:2, :] * scale
            image_path = image_path_new
            mask_path = mask_path_new

        focal_length_x = newK[0, 0]
        focal_length_y = newK[1, 1]

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, mask_path=mask_path, width=width, height=height,
                              image=None, image_name=None)
        cam_infos.append(cam_info)
    return cam_infos

def create_center_radius(center, radius=5., up='y', ranges=[0, 360, 36], angle_x=0, **kwargs):
    center = np.array(center).reshape(1, 3)
    thetas = np.deg2rad(np.linspace(*ranges))
    st = np.sin(thetas)
    ct = np.cos(thetas)
    zero = np.zeros_like(st)
    Rotx = cv2.Rodrigues(np.deg2rad(angle_x) * np.array([1., 0., 0.]))[0]
    if up == 'z':
        center = np.stack([radius*ct, radius*st, zero], axis=1) + center
        R = np.stack([-st, ct, zero, zero, zero, zero-1, -ct, -st, zero], axis=-1)
    elif up == 'y':
        center = np.stack([radius*ct, zero, radius*st, ], axis=1) + center
        R = np.stack([
            +st,  zero,  -ct,
            zero, zero-1, zero, 
            -ct,  zero, -st], axis=-1)
    R = R.reshape(-1, 3, 3)
    R = np.einsum('ab,fbc->fac', Rotx, R)
    center = center.reshape(-1, 3, 1)
    T = - R @ center
    RT = np.dstack([R, T])
    return RT

def create_cameras_mean(cameras, allstep=36):
    Told = np.stack([d.T.reshape(3, 1) for d in cameras])
    Rold = np.stack([d.R.T for d in cameras])
    Cold = - np.einsum('bmn,bnp->bmp', Rold.transpose(0, 2, 1), Told)
    center = Cold.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(Cold - center, axis=1).mean() * 0.9
    zmean = Rold[:, 2, 2].mean()
    xynorm = np.sqrt(1. - zmean**2)
    thetas = np.linspace(0., 2*np.pi, allstep)
    # 计算第一个相机对应的theta
    dir0 = Cold[0] - center[0]
    dir0[2, 0] = 0.
    dir0 = dir0 / np.linalg.norm(dir0)
    theta0 = np.arctan2(dir0[1,0], dir0[0,0]) + np.pi/2
    thetas += theta0
    sint = np.sin(thetas)
    cost = np.cos(thetas)
    R1 = np.stack([cost, sint, np.zeros_like(sint)]).T
    R3 = xynorm * np.stack([-sint, cost, np.zeros_like(sint)]).T
    R3[:, 2] = zmean
    R2 = - np.cross(R1, R3)
    Rnew = np.stack([R1, R2, R3], axis=1)
    # set locations
    loc = np.stack([radius * sint, -radius * cost, np.zeros_like(sint)], axis=1)[..., None] + center
    print('[sample] camera centers: ', center[0].T[0])
    print('[sample] camera radius: ', radius)
    print('[sample] camera start theta: ', theta0)
    Tnew = -np.einsum('bmn,bnp->bmp', Rnew, loc)
    return Rnew, Tnew

def readEasyMocapInfo(path):
    from easymocap.mytools.camera_utils import read_cameras
    cameras = read_cameras(path)

    cam_infos = readEasyMocapCameras(cameras, path)

    train_cam_infos = cam_infos
    test_cam_infos = []
    # RT = create_center_radius([0, 0, 1.], radius=3., up='z', ranges=[0, 360, 36], angle_x=30)
    Rnew, Tnew = create_cameras_mean(train_cam_infos, allstep=180)
    for i in range(Rnew.shape[0]):
        cam = copy.deepcopy(train_cam_infos[0])
        # cam = cam._replace(R=RT[i, :3, :3].T)
        # cam = cam._replace(T=RT[i, :3, 3])
        cam = cam._replace(R=Rnew[i].T)
        cam = cam._replace(T=Tnew[i].reshape(3,))
        
        test_cam_infos.append(cam)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "colmap/sparse/model/points3D.ply")
    bin_path = os.path.join(path, "colmap/sparse/model/points3D.bin")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info