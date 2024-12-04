import glob
from utils.render_utils import load_img
import os
import numpy as np
import open3d as o3d
import json
import torch
from scene.cameras import getProjectionMatrix, getWorld2View2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.gaussian_model import GaussianModel
from render import GaussianExtractor
from gaussian_renderer import render
import cv2
from tqdm import tqdm
import pyrender
import trimesh

class Pipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        self.depth_ratio = 0.0
        
class DummyCamera:
    def __init__(self,img_name, R, T, FoVx, FoVy, f_x, f_y, w, h, znear=0.01, zfar=100.0):
        """
        R: cam 2 world
        T: world 2 cam
        """
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.t = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = w
        self.image_height = h
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.f_x = f_x
        self.f_y = f_y
        self.k = np.array([[f_x, 0, w/2], [0, f_y, h/2], [w/2, h/2, 1]])
        self.cam_name = img_name
    
    def translate(self, t):
        """
        t: translation vector (in camera coordinate)
        """
        if isinstance(t, list):
            t = np.array(t)
        cam_pos = -self.R @ self.t + t
        t_w2c = -self.R.T @ cam_pos
        self.t = t_w2c
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def rotate(self, angles_list):
        """
        angles_list: [x, y, z] (in camera coordinate)
        """
        def get_rotation_matrix(angles):
            """
            Convert angles in degrees to rotation matrix
            angles: [x, y, z] angles in degrees
            pi: math.pi
            """
            # Convert degrees to radians
            rx, ry, rz = [angle * np.pi / 180.0 for angle in angles]
            
            # Rotation matrices around x, y, z axes
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]])
            
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0], 
                        [-np.sin(ry), 0, np.cos(ry)]])
            
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz), np.cos(rz), 0],
                        [0, 0, 1]])
            
            # Combined rotation matrix
            R = Rx @ Ry @ Rz
            return R
        assert len(angles_list) == 3
        self.R = self.R @ get_rotation_matrix(angles_list)
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def __str__(self):
        return f"DummyCamera(R={self.R}, T={self.t}, FoVx={self.FoVx}, FoVy={self.FoVy}, \
            image_width={self.image_width}, image_height={self.image_height}, world_view_transform={self.world_view_transform}, \
                projection_matrix={self.projection_matrix}, full_proj_transform={self.full_proj_transform}, camera_center={self.camera_center})"
class SceneReconstruction:
    def __init__(self, result_root, intrinsics, model_path):
        self.result_root = result_root
        self.intrinsics = intrinsics

        gaussians = GaussianModel(3)
        gaussians.load_ply(model_path)
        pipe = Pipeline()
        self.gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=[0, 0, 0])    

        os.makedirs(result_root, exist_ok=True)
        

    def generate_pcd(self, img, depth):
        # Get image dimensions
        height, width = depth.shape[:2]

        # Create pixel coordinate grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Convert to homogeneous coordinates
        pixels = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=0)

        # Get camera intrinsics
        fx = self.intrinsics[0,0]
        fy = self.intrinsics[1,1]
        cx = self.intrinsics[0,2] 
        cy = self.intrinsics[1,2]

        # Create inverse intrinsics matrix
        K_inv = np.array([
            [1/fx, 0, -cx/fx],
            [0, 1/fy, -cy/fy],
            [0, 0, 1]
        ])

        # Back-project pixels to 3D points
        points = depth.flatten() * (K_inv @ pixels)
        
        # Get colors for each point
        colors = img.reshape(-1, 3)

        # Remove points with invalid depth
        valid_mask = depth.flatten() > 0
        points = points[:, valid_mask]
        colors = colors[valid_mask]

        # Return points and colors as Nx3 arrays
        return [points.T, colors]

    @torch.no_grad()
    def reconstruction(self, cam_list: list[DummyCamera]):
        depths = []
        rgbs = []
        for cam in cam_list:
            render_pkg = self.gaussExtractor.render(cam, self.gaussExtractor.gaussians)
            depth = render_pkg['surf_depth'].cpu().numpy()[0]
            rgb = render_pkg['render'].cpu().numpy().transpose(1,2,0)
            rgb = (np.clip(np.nan_to_num(rgb), 0., 1.) * 255.).astype(np.uint8)
            depths.append(depth)
            rgbs.append(rgb)
        return rgbs, depths

    def get_relative_pose(self, src_cam: DummyCamera, target_cam: DummyCamera):
        """
        return relative rotation and translation
        """
        relative_R = target_cam.R.T @ src_cam.R
        relative_T = target_cam.t - relative_R @ src_cam.t
        return relative_R, relative_T

    def generate_stable_pcd(self, campair_list: list[list[DummyCamera, DummyCamera, float]], world_cam: DummyCamera):
        for cam_pair in campair_list:
            rgbs, depths = self.reconstruction(cam_pair[:2])
            cv2.imwrite(os.path.join(self.result_root, 'rgb', f"{cam_pair[0].cam_name}_left.png"), rgbs[0][...,::-1])
            cv2.imwrite(os.path.join(self.result_root, 'rgb', f"{cam_pair[1].cam_name}_right.png"), rgbs[1][...,::-1])
            vis_rgbs = np.concatenate(rgbs, axis=0)
            os.makedirs(os.path.join(self.result_root, 'rgb'), exist_ok=True)
            cv2.imwrite(os.path.join(self.result_root, 'rgb', f"{cam_pair[0].cam_name}_vis.png"), vis_rgbs[...,::-1])


            pcd_left = self.generate_pcd(rgbs[0], depths[0])
            pcd_right = self.generate_pcd(rgbs[1], depths[1])
            pcd_relative_R, pcd_relative_T = self.get_relative_pose(cam_pair[1], cam_pair[0])
            save_right = os.path.join(self.result_root, 'pcd', f"{cam_pair[0].cam_name}_right_ori.ply")
            save_ply(pcd_right[0], pcd_right[1], save_right)
            pcd_right[0] = pcd_right[0] @ pcd_relative_R.T + pcd_relative_T
            os.makedirs(os.path.join(self.result_root, 'pcd'), exist_ok=True)
            save_left = os.path.join(self.result_root, 'pcd', f"{cam_pair[0].cam_name}_left.ply")
            save_right = os.path.join(self.result_root, 'pcd', f"{cam_pair[0].cam_name}_right.ply")
            save_ply(pcd_left[0], pcd_left[1], save_left)
            save_ply(pcd_right[0], pcd_right[1], save_right)


            consistency_mask = verify_depth_consistency(depths[0], depths[1], cam_pair[2], self.intrinsics[0,0], 0.05)
            left_depth = depths[0]
            left_depth[~consistency_mask] = 0
            left_pcd = self.generate_pcd(rgbs[0], left_depth)
            # transform to world coordinate
            # left_pcd[0] = left_pcd[0] @ cam_pair[0].R.T - cam_pair[0].R @ cam_pair[0].t
            R, t = self.get_relative_pose(cam_pair[0], world_cam)
            left_pcd[0] = left_pcd[0] @ R.T + t
            img_name = int(cam_pair[0].cam_name)
            img_name = f"{img_name:05d}"
            save_path = os.path.join(self.result_root, 'pcd', f"{img_name}.ply")
            os.makedirs(os.path.join(self.result_root, 'pcd'), exist_ok=True)
            save_ply(left_pcd[0], left_pcd[1], save_path)

def calculate_fov(output_width, output_height, f_x, f_y):
    fovx = focal2fov(f_x, output_width)
    fovy = focal2fov(f_y, output_height)
    return fovx, fovy


def remap_left_view(left_img, right_img, left_depth, baseline, focal_length):
    height, width = left_img.shape[:2]

    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    disparity = baseline * focal_length / left_depth
    x_right = xx - disparity
    
    # Fill in the mapping coordinates
    map_x = (xx+disparity).astype(np.float32)
    map_y = yy.astype(np.float32)
    
        
    # Remap left image to right view perspective
    # dst(x,y) =  src(map_x(x,y),map_y(x,y))

    left_img_warped = cv2.remap(left_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Save warped image for visualization
    vis_warped = np.concatenate([right_img, left_img_warped], axis=0)
    return vis_warped

def verify_depth_consistency(left_depth, right_depth, baseline, focal_length, threshold = 0.1):
    height, width = left_depth.shape[:2]

    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    disparity = baseline * focal_length / left_depth
    x_right = xx - disparity

    valid_mask = (x_right >= 0) & (x_right < width)
    x_right_int = np.round(x_right).astype(int)

    right_depth_warped = np.zeros_like(left_depth)
    right_depth_warped[valid_mask] = right_depth[yy[valid_mask], x_right_int[valid_mask]]
    
    depth_diff = np.abs(left_depth - right_depth_warped)
    
    relative_diff = depth_diff / left_depth 
    relative_diff[~valid_mask] = 0
    
    consistency_mask = (relative_diff < threshold) & valid_mask
    return consistency_mask


def construct_camera_list(cam_info, baseline = 0.5, cam_idx=0, scale=1):
    stereo_pair_list = []
    R1 = np.array(cam_info[cam_idx]["rotation"]) # cam2world
    t1 = np.array(cam_info[cam_idx]["position"]) # cam2world
    t1 = -R1.T @ t1 # world2cam
    
    w_ori, h_ori = cam_info[cam_idx]["width"], cam_info[cam_idx]["height"]
    w, h = w_ori//scale, h_ori//scale
    # scale = w_ori / w
    f_x = cam_info[cam_idx]["fx"] / scale
    f_y = cam_info[cam_idx]["fy"] / scale
    fovx, fovy = calculate_fov(w, h, f_x, f_y)
    cam_left = DummyCamera(cam_info[cam_idx]["img_name"], R1, t1, fovx, fovy, f_x, f_y, w, h)
    # cam_left.translate([0,0,5])
    # cam_left.rotate([-15, 15, 0])
    # translate to right view
    t2 = cam_left.t + np.array([-baseline, 0, 0])
    cam_right = DummyCamera(cam_left.cam_name+'_translated', cam_left.R, t2, cam_left.FoVx, cam_left.FoVy,\
                             cam_left.f_x, cam_left.f_y, cam_left.image_width, cam_left.image_height)
    stereo_pair_list.append([cam_left, cam_right, baseline])
    return stereo_pair_list

def save_ply(points, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(save_path, pcd)

def load_intrinsics(data, scale):
    intrinsics = np.array([[data['fx'], 0, data['width']/2], [0, data['fy'], data['height']/2], [0, 0, 1]])
    return intrinsics/scale

def get_depth_from_mesh_pyrender(mesh_file, camera: DummyCamera, width, height):
    # Use OSMesa backend instead of GLFW
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    
    # Rest of the function remains the same
    trimesh_mesh = trimesh.load(mesh_file)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    
    scene = pyrender.Scene()
    scene.add(mesh)
    
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = camera.R.T
    cam_pose[:3, 3] = -camera.R @ camera.t
    
    pyrender_camera = pyrender.IntrinsicsCamera(
        fx=camera.f_x,
        fy=camera.f_y,
        cx=width/2,
        cy=height/2,
        znear=0.1,
        zfar=100.0
    )
    
    scene.add(pyrender_camera, pose=cam_pose)
    
    flags = pyrender.RenderFlags.DEPTH_ONLY
    r = pyrender.OffscreenRenderer(width, height)
    depth = r.render(scene, flags=flags)[0]
    
    r.delete()
    
    return depth

def get_depth_from_mesh_o3d(mesh_file, camera: DummyCamera, width, height):
    #os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # 计算相机位置和朝向
    cam_pos = -camera.R @ camera.t  # 相机在世界坐标系中的位置
    cam_forward = camera.R @ np.array([0, 0, 1])  # 相机朝向
    cam_up = camera.R @ np.array([0, -1, 0])  # 相机上方向
    look_at_point = cam_pos + cam_forward  # 看向的点
    
    # 设置相机参数
    render.scene.camera.look_at(
        look_at_point,  # 看向的点
        cam_pos,        # 相机位置
        cam_up          # 上方向
    )
    
    # 设置相机内参
    render.scene.camera.set_projection(
        camera.FoVy * 180 / np.pi,  # FOV in degrees
        float(width) / height,       # aspect ratio
        camera.znear,                         # near plane
        camera.zfar,                        # far plane
        o3d.visualization.rendering.Camera.FovType.Vertical
    )
    
    # 添加mesh到场景
    render.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.Material())
    
    # 渲染深度图
    depth = render.render_to_depth_image()
    depth_np = 1 - np.asarray(depth)
    invalid_mask = depth_np==0
    depth_np = camera.znear / depth_np
    depth_np[invalid_mask] = 0
    return depth_np

def save_depth_vis(depth, save_path):
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Convert to uint8 for cv2 visualization
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # Save visualization
    cv2.imwrite(save_path, depth_color)

if __name__ == "__main__":
    result_root = 'output/DJI_0019/pseudo_s2'
    camera_json_file = 'output/DJI_0019/cameras.json'
    model_path = 'output/DJI_0019/point_cloud/iteration_30000/point_cloud.ply'
    
    SCALE = 2
    BASELINE = 0.7
    with open(camera_json_file, "r") as f:
        cam_info = json.load(f)
    cam_info.sort(key=lambda x: int(x["img_name"]))
    intrinsics = load_intrinsics(cam_info[0], SCALE)
    
    sr = SceneReconstruction(result_root, intrinsics, model_path)
    custom_cam_list = construct_camera_list(cam_info, baseline=BASELINE, cam_idx=0, scale=SCALE)
    world_cam = custom_cam_list[0][0]
    # indices = list(range(0, len(cam_info)//3, 20))
    indices = [0, 110, 220, 330]
    for cam_idx in tqdm(indices):
        custom_cam_list = construct_camera_list(cam_info, baseline=BASELINE, cam_idx=cam_idx, scale=SCALE)
        # print(custom_cam_list[0][0])
        sr.generate_stable_pcd(custom_cam_list, world_cam)
    
    # cam = construct_camera_list(cam_info, baseline=BASELINE, cam_idx=0)[0][0]
    # mesh_file = 'output/DJI_0019/train/ours_30000/fuse_post.ply'
    # depth = get_depth_from_mesh_o3d(mesh_file, cam, 1920, 1080)
    # print(depth.shape)
    
    
