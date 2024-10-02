import torch
from isaacgym import gymapi
from isaacgym import gymtorch


class PointCloudGenerator:
    def __init__(self, proj_matrix, view_matrix, origin, camera_props=None, depth_max=1.0, depth_min=0.1, device='cpu'):
        self.cam_width = camera_props["width"]
        self.cam_height = camera_props["height"]
        self.device = device

        fu = 2 / proj_matrix[0, 0]
        fv = 2 / proj_matrix[1, 1]
        self.fu = self.cam_width / fu
        self.fv = self.cam_height / fv
        self.cu = self.cam_width / 2.
        self.cv = self.cam_height / 2.

        self.int_mat = torch.Tensor([[-self.fu, 0, self.cu], [0, self.fv, self.cv], [0, 0, 1]])
        self.ext_mat = torch.inverse(torch.Tensor(view_matrix).to(device))
        self.int_mat_T_inv = torch.inverse(self.int_mat.T).to(device)
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.origin = torch.Tensor(origin).to(device)

        x, y = torch.meshgrid(torch.arange(self.cam_height), torch.arange(self.cam_width))
        self._uv_one = torch.stack((y, x, torch.ones_like(x)), dim=-1).float().to(device)
        self._uv_one_in_cam = self._uv_one @ self.int_mat_T_inv
        self._uv_one_in_cam = self._uv_one_in_cam.repeat(1, 1, 1)

    def get_cam_int_mat(self):
        int_mat = self.int_mat.clone()
        int_mat[0][0] = - int_mat[0][0]
        return int_mat

    @torch.no_grad()
    def convert(self, depth_buffer, seg_buffer, downscale=1):
        depth_buffer = depth_buffer
        valid_ids = (depth_buffer > -self.depth_max) & (depth_buffer < -self.depth_min)

        depth_buffer = depth_buffer[::downscale, ::downscale]
        uv_one_in_cam = self._uv_one_in_cam[::downscale, ::downscale]
        valid_ids = valid_ids[::downscale, ::downscale]
        seg_buffer = seg_buffer[::downscale, ::downscale]

        valid_depth = depth_buffer[valid_ids]
        seg_mask = seg_buffer[valid_ids]
        uv_one_in_cam = uv_one_in_cam[valid_ids]

        pts_in_cam = torch.mul(uv_one_in_cam, valid_depth.unsqueeze(-1))
        pts_in_cam = torch.cat((pts_in_cam, torch.ones(*pts_in_cam.shape[:-1], 1, device=pts_in_cam.device)), dim=-1)

        pts_in_world = pts_in_cam @ self.ext_mat
        pcd_pts = pts_in_world[..., :3]
        pcd_pts -= self.origin

        cam_pose = self.ext_mat.T.clone()
        cam_pose[:3, 3] -= self.origin

        return pcd_pts, pts_in_cam, seg_mask, cam_pose, depth_buffer, seg_buffer

    def update_camera_pose(self, mtx):
        self.ext_mat = torch.inverse(torch.Tensor(mtx).to(self.device))


class CameraPointCloud:
    def __init__(self, sim, gym, envs, camera_handles, camera_params, depth_max=10.0, depth_min=0.1,
                 graphics_device='cpu', compute_device='cpu'):

        self.sim = sim
        self.gym = gym
        self.envs = envs
        self.camera_handles = camera_handles

        print(f'Depth max:{depth_max}')
        print(f'Depth min: {depth_min}')

        self.graphics_device = graphics_device
        self.compute_device = compute_device

        self.num_envs = len(self.envs)
        self.num_cams = len(camera_handles[0])
        print(f'Number of envs in camera:{self.num_envs}')
        print(f'Number of cameras:{self.num_cams}')

        self.depth_tensors = []
        self.seg_tensors = []
        self.pt_generators = []

        for idx in range(len(envs)):
            depth_buffers, seg_buffers, pt_generators = [], [], []
            for c in range(len(camera_handles[idx])):
                c_handle = camera_handles[idx][c]
                env = envs[idx]

                # create depth_tensor and seg_tensor
                depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, c_handle, gymapi.IMAGE_DEPTH)
                torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
                depth_buffers.append(torch_depth_tensor)

                seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, c_handle, gymapi.IMAGE_SEGMENTATION)
                torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
                seg_buffers.append(torch_seg_tensor)

                proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env, c_handle)
                view_matrix = self.gym.get_camera_view_matrix(self.sim, env, c_handle)
                origin = self.gym.get_env_origin(env)

                pt_generators.append(
                    PointCloudGenerator(
                        camera_props=camera_params,
                        proj_matrix=proj_matrix,
                        view_matrix=view_matrix,
                        origin=[origin.x, origin.y, origin.z],
                        depth_max=depth_max,
                        depth_min=depth_min,
                        device=self.graphics_device
                    )
                )
            self.depth_tensors.append(depth_buffers)
            self.seg_tensors.append(seg_buffers)
            self.pt_generators.append(pt_generators)

    def update_camera_pose(self):
        for idx in range(len(self.envs)):
            for c in range(len(self.camera_handles[idx])):
                c_handle = self.camera_handles[idx][c]
                env = self.envs[idx]
                view_mtx = self.gym.get_camera_view_matrix(self.sim, env, c_handle)
                self.pt_generators[idx][c].update_camera_pose(view_mtx)

    @torch.no_grad()
    def get_point_cloud(self, env_ids=None, downscale=1):
        if env_ids is None:
            env_ids = range(self.num_envs)

        out = []
        for i, idx in enumerate(env_ids):
            env_pts, env_segs_pts, env_cam_pts, env_cam_poses, env_depths, env_segs = \
                self.get_ptd_cuda(env_idx=idx, downscale=downscale)

            out.append({'pts': env_pts,
                        'seg': env_segs_pts,
                        'cam_pts': env_cam_pts,
                        'cam_poses': env_cam_poses,
                        'raw_depths': env_depths,
                        'raw_segs': env_segs,
            })
        return out

    @torch.no_grad()
    def get_ptd_cuda(self, env_idx, downscale=1):
        env_world_pts, env_cam_pts, env_seg_pts, env_cam_poses, env_depths, env_segs = [], [], [], [], [], []
        for i, (depth, seg) in enumerate(zip(self.depth_tensors[env_idx], self.seg_tensors[env_idx])):
            pts_world, pts_cam, seg_pts, cam_pose, n_depth, n_seg = self.pt_generators[env_idx][i].convert(depth, seg, downscale=downscale)
            env_world_pts.append(pts_world)
            env_seg_pts.append(seg_pts)
            env_cam_pts.append(pts_cam)
            env_cam_poses.append(cam_pose)
            env_depths.append(n_depth.clone())
            env_segs.append(n_seg.clone())

        return env_world_pts, env_seg_pts, env_cam_pts, env_cam_poses, env_depths, env_segs

    def get_cam_params(self, env_idx):
        # assume cam intrinsics are the same
        int_mat = self.pt_generators[env_idx][0].get_cam_int_mat().cpu().numpy()

        cam_params = {
            'height': self.pt_generators[env_idx][0].cam_height,
            'width': self.pt_generators[env_idx][0].cam_width,
            'z_near': self.pt_generators[env_idx][0].depth_min,
            'z_far': self.pt_generators[env_idx][0].depth_max,
            'fx': int_mat[0][0],
            'fy': int_mat[1][1],
            'cx': int_mat[0][2],
            'cy': int_mat[1][2]
        }

        return cam_params