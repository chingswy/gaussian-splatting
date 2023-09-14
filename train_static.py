import os
import math
from os.path import join
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from easymocap.mytools.camera_utils import read_cameras
from scene import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim

def getNerfppNorm(cam_centers):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

class ColmapDataset:
    def read_pcd(self, filename, bbox3d=None):
        from plyfile import PlyData, PlyElement
        from utils.graphics_utils import BasicPointCloud
        plydata = PlyData.read(filename)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        normals = np.zeros_like(colors)
        print('> Max positions: {}, Min positions: {}'.format(positions.max(axis=0), positions.min(axis=0)))
        if bbox3d is not None:
            bbox3d = np.array(bbox3d)
            valid = (positions[:, 0] > bbox3d[0, 0]) & (positions[:, 1] > bbox3d[0, 1]) & (positions[:, 2] > bbox3d[0, 2]) &\
                    (positions[:, 0] < bbox3d[1, 0]) & (positions[:, 1] < bbox3d[1, 1]) & (positions[:, 2] < bbox3d[1, 2])
            print('> filter {}/{} points'.format(valid.sum(), valid.shape[0]))
            return BasicPointCloud(points=positions[valid], colors=colors[valid], normals=normals[valid])
        else:
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

    def __init__(self, path, cache_path, camera, image) -> None:
        self.cache_path = cache_path
        # ATTN: 注意这里
        cameras = read_cameras(join(path, camera))
        for key, cam in cameras.items():
            cam['center'] = - cam['R'].T @ cam['T']
            cam['H'] = 1024
            cam['W'] = 1024
        self.znear = 0.1
        self.zfar = 100.0
        if False:
            self.pcd = self.read_pcd(join(path, camera, 'sparse.ply'), bbox3d)
            center = np.array([100, -700, 250])
            dx, dy = 200, 200
            bbox3d = [[center[0]-dx, center[1]-dy, center[2]-100], [center[0]+dx, center[1]+dy, center[2]+100]]
            cameras_filter = {}
            for key, cam in cameras.items():
                cam_center = - cam['R'].T @ cam['T']
                cam['center'] = cam_center
                cam_center = cam_center[:, 0]
                lookat = cam['R'].T @ np.array([0, 0, 1]).reshape(3, 1)
                target = center - cam_center
                distance = np.linalg.norm(target)
                target = target / distance
                inner = (lookat.flatten() * target.flatten()).sum()
                if cam_center[0] < bbox3d[0][0] or cam_center[1] < bbox3d[0][1] or\
                cam_center[0] > bbox3d[1][0] or cam_center[1] > bbox3d[1][1]:
                    continue
                if inner < 0.6:
                    continue
                # print(f'{key}: {inner}, angle={np.rad2deg(np.arccos(inner)):.1f}, dist={distance:.1f}')
                cameras_filter[key] = cam
            cameras = cameras_filter
        self.cameras = cameras
        self.pcd = self.read_pcd(join(path, camera, 'sparse.ply'))
        cam_centers = [d['center'] for d in self.cameras.values()]
        self.nerf_normalization = getNerfppNorm(cam_centers)
        print('Totally {} cameras'.format(len(self.cameras.keys())))
        print('>> normalization: {}'.format(self.nerf_normalization))
        image_dir = join(path, image)
        subs = sorted(os.listdir(image_dir))
        self.use_camparam_per_image = False
        imgnames = []
        for sub in subs:
            imgs = sorted(os.listdir(join(image_dir, sub)))[:1]
            for imgname in imgs:
                if self.use_camparam_per_image:
                    keyname = f"{sub}/{imgname.replace('.JPG', '')}"
                else:
                    keyname = sub
                if keyname not in self.cameras:continue
                imgnames.append(join(image_dir, sub, imgname))
        print('Totally {} images'.format(len(imgnames)))
        self.imgnames = imgnames
        self.info = {}
        self.undistort(downscale=1)
    
    def convert_to_gaussian_camera2(batch, znear=0.01, zfar=100., batch_id: int = 0):
        output = dotdict()
        
        output.image_height = batch.H[batch_id]
        output.image_width = batch.W[batch_id]
        
        output.K = batch.K[batch_id]
        output.R = batch.R[batch_id]
        output.T = batch.T[batch_id]

        fl_x = batch.K[batch_id, 0, 0]
        fl_y = batch.K[batch_id, 1, 1]
        
        output.FoVx = focal2fov(fl_x, output.image_width)
        output.FoVy = focal2fov(fl_y, output.image_height)

        output.world_view_transform = getWorld2View(output.R, output.T).transpose(0, 1)
        output.projection_matrix = getProjectionMatrix2(output.K, output.image_height, output.image_width, znear, zfar).transpose(0, 1)
        output.full_proj_transform = torch.matmul(output.world_view_transform, 
                                                output.projection_matrix)
        output.camera_center = output.world_view_transform.inverse()[3, :3]
        
        return output

    def prepare_camera(self, camera, scale):
        ret = {}
        focal_length_x = camera['K'][0, 0] / scale
        focal_length_y = camera['K'][1, 1] / scale
        ret['FoVy'] = focal2fov(focal_length_y, camera['H'] / scale)
        ret['FoVx'] = focal2fov(focal_length_x, camera['W'] / scale)
        # w, h
        ret['image_width'] = camera['W'] / scale
        ret['image_height'] = camera['H'] / scale
        ret['projection_matrix'] = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=ret['FoVx'], fovY=ret['FoVy']).T
        world_view_transform = np.eye(4)
        world_view_transform[:3, :3] = camera['R']
        world_view_transform[:3, 3:] = camera['T']
        world_view_transform = world_view_transform.T
        # ret['camera_center'] = np.linalg.inv(world_view_transform)[3, :3]
        ret['camera_center'] = camera['center'].reshape(3,)
        ret['world_view_transform'] = world_view_transform
        ret['full_proj_transform'] = world_view_transform @ ret['projection_matrix']
        for key, val in ret.items():
            if isinstance(val, np.ndarray):
                ret[key] = val.astype(np.float32)
        return ret

    def undistort(self, downscale=8):
        cameras_cache = {}
        imgnames = []
        for imgname in tqdm(self.imgnames, desc='undistort'):
            sub = self.get_sub(imgname)
            basename = os.path.basename(imgname).split('.')[0]
            outname = join(self.cache_path, 'scale_{}'.format(downscale), f'{sub}/{basename}.jpg')
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            if self.use_camparam_per_image:
                camname = f'{sub}/{basename}'
            else:
                camname = sub
            if sub not in cameras_cache:
                camera = self.cameras[camname]
                width, height = camera['W'], camera['H']
                newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                            (width, height), 0, (width,height), centerPrincipalPoint=True)
                mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
                cameras_cache[sub] = (mapx, mapy, newK)
            mapx, mapy, newK = cameras_cache[sub]
            Knew = newK.copy()
            camera = {
                'K': newK,
                'R': self.cameras[camname]['R'],
                'T': self.cameras[camname]['T'],
                'W': self.cameras[camname]['W'],
                'H': self.cameras[camname]['H'],
                'center': self.cameras[camname]['center']
            }
            camera = self.prepare_camera(camera, scale=downscale)

            imgnames.append({
                'imgname': outname,
                'camera': camera
            })
            if os.path.exists(outname):
                continue
            img = cv2.imread(imgname)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            dst = cv2.resize(dst, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(outname, dst)
            
        self.info[downscale] = imgnames

    @staticmethod    
    def get_sub(imgname):
        sub = os.path.basename(os.path.dirname(imgname))
        return sub

    def __len__(self):
        return len(self.imgnames)
    
    @staticmethod
    def read_image(imgname):
        # assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        img = img.astype(np.float32)/255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        scale = 1
        data = self.info[scale][index]
        imgname = data['imgname']
        camera = data['camera']
        img = self.read_image(imgname)
        return {
            'image': img,
            'camera': camera
        }

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class TrainRender(nn.Module):
    def __init__(self):
        super().__init__()
        background = torch.tensor([0., 0., 0.], dtype=torch.float32)
        self.register_buffer('background', background)

    def forward(self, batch, model):
        camera = {}
        for key in ['camera_center', 'world_view_transform', 'full_proj_transform', 'image_width', 'image_height', 'FoVx', 'FoVy']:
            camera[key] = batch['camera'][key][0]
        render_pkg = render(camera, model, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        output = {
            'output': image,
            'viewspace_point_tensor': viewspace_point_tensor,
            'visibility_filter': visibility_filter,
            'radii': radii
        }
        print(image.min(), image.max())
        if True:
            gt_image = batch['image'][0].permute(2, 0, 1)
            Ll1 = l1_loss(image[:3], gt_image[:3])
            lambda_dssim = 0.2
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image[:3], gt_image[:3]))
            output['loss'] = loss
            output['gt'] = gt_image
        return output

if __name__ == '__main__':
    if False:
        path = '/home/xuzhen/demo/local/gaussian-splatting/data/crab'
        cache_path = '/mnt/data2/home/shuaiqing/gaussian-cache/zju'
        camera = 'distorted/sparse/0'
        image = 'input'
    elif True:
        path = '/nas/ZJUMoCap/Part3/20211204/527'
        cache_path = '/mnt/data2/home/shuaiqing/gaussian-cache/527'
        camera = ''
        image = 'images'
    else:
        path = '../gaussian-splatting-debug/data/fig_mvs'
        cache_path = '/mnt/data2/home/shuaiqing/gaussian-cache/fig_mvs'
        camera = 'sparse/0'
        image = 'images'
    dataset = ColmapDataset(path, cache_path, camera=camera, image=image)
    model = GaussianModel(sh_degree=3)
    model.create_from_pcd(dataset.pcd, dataset.nerf_normalization["radius"])
    renderer = TrainRender()
    import argparse
    from arguments import ModelParams, PipelineParams, OptimizationParams
    parser = argparse.ArgumentParser()
    opt = OptimizationParams(parser)

    model.training_setup(opt)

    device = torch.device('cuda:0')
    # model.to(device)
    renderer.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )
    iteration = 0
    for epoch in range(10000):
        for data in dataloader:
            iteration += 1
            model.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                model.oneupSHdegree()
                print('Increasing SH degree to {}'.format(model.active_sh_degree))
                print('Current points: {}'.format(model.get_xyz.shape[0]))

            batch = {}
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    batch[key] = torch.FloatTensor(val).to(device)
                elif key == 'camera':
                    for camk in val.keys():
                        if isinstance(val[camk], np.ndarray):
                            val[camk] = torch.FloatTensor(val[camk]).to(device)
                        elif torch.is_tensor(val[camk]):
                            val[camk] = val[camk].float().to(device)
                    batch[key] = val
                elif torch.is_tensor(val):
                    batch[key] = val.to(device)
                else:
                    batch[key] = val
            output = renderer(batch, model)
            loss = output['loss']
            loss.backward()
            if iteration % 100 == 0:
                print(f'{iteration}: {loss.item()}')
                pred = output['output'].detach()
                gt = output['gt']
                vis = torch.cat([pred, gt], dim=2).cpu().numpy().transpose(1, 2, 0)
                vis = (np.clip(vis[:,:,::-1], 0., 1.)*255).astype(np.uint8)
                cv2.imwrite('debug/iter_{:06d}.jpg'.format(iteration), vis)
            with torch.no_grad():
                # Densification
                viewspace_point_tensor = output['viewspace_point_tensor']
                visibility_filter = output['visibility_filter']
                radii = output['radii']
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    model.max_radii2D[visibility_filter] = torch.max(model.max_radii2D[visibility_filter], radii[visibility_filter])
                    model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        model.densify_and_prune(opt.densify_grad_threshold, 0.005, dataset.nerf_normalization["radius"], size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0: # or (dataset.white_background and iteration == opt.densify_from_iter):
                        model.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none = True)


    if False:
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
