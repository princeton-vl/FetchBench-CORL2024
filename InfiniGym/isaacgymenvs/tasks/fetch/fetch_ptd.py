
import numpy as np
import os
import torch
import trimesh
import trimesh.transformations as tr

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_base import FetchBase, image_to_video
from isaacgymenvs.tasks.fetch.utils.point_cloud_utils import CameraPointCloud


class FetchPointCloudBase(FetchBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        ptd_cam_handles = []
        for idx in range(len(self.envs)):
            ptd_cam_handles.append(self.cameras[idx][:-1])

        graphics_device = self.graphics_device_id if self.graphics_device_id >= 0 else 'cpu'
        self.cam_point_clouds = CameraPointCloud(self.sim, self.gym, self.envs, ptd_cam_handles,
                                                 camera_params=self.cfg["env"]["cam"],
                                                 depth_max=self.cfg["env"]["cam"]["depth_max"],
                                                 depth_min=self.cfg["env"]["cam"]["depth_min"],
                                                 graphics_device=graphics_device,
                                                 compute_device=self.device)

    def solve(self):
        # set goal obj color
        self.set_target_color()

        for _ in range(10):
            self.env_physics_step()
            self.post_phy_step()

        cam_data = self.get_camera_data(numpy_rgb=True, tensor_ptd=True, segmented_ptd=True, ptd_downscale=1)
        #cam_data = self.get_camera_data(numpy_rgb=True, tensor_ptd=True, segmented_ptd=True, ptd_downscale=2)
        numpy_render = cam_data['camera_render_raw']

        # set to default color
        self.set_default_color()

        return image_to_video(numpy_render["rgb"]), None

    """
    Reset cam
    """

    def reset_cam(self):
        super().reset_cam()
        if hasattr(self, 'cam_point_clouds'):
            self.cam_point_clouds.update_camera_pose()

    """
    Camera
    """

    @torch.no_grad()
    def ptd_to_robot(self, pts, env_idx):
        rq, rt = tf_inverse(self._robot_base_state[env_idx, 3:7].clone(), self._robot_base_state[env_idx, :3].clone())
        robot_base_pts = rt + quat_apply(rq, pts)

        return robot_base_pts

    @torch.no_grad()
    def filter_ptd_segmentation(self, pts, seg, env_idx):
        # filter robot, goal, scene
        robot_pixels = seg == 1
        goal_pixels = seg == (self.task_obj_index[env_idx][self.get_task_idx()] + 4)

        seg_pts = {
            'robot': pts[robot_pixels].detach(),
            'goal': pts[goal_pixels].detach(),
            'scene': pts[~(robot_pixels | goal_pixels)].detach()
        }

        return seg_pts

    def filter_pointcloud(self, tensor_ptd, convert_robot_base=True):
        seg_pts = []
        for i in range(self.num_envs):
            pts, seg = tensor_ptd[i]['pts'], tensor_ptd[i]['seg']
            pts = torch.concat(pts, dim=0)
            seg = torch.concat(seg, dim=0)
            if convert_robot_base:
                pts = self.ptd_to_robot(pts, i)
            seg_pts.append(self.filter_ptd_segmentation(pts, seg, i))

        return seg_pts

    def get_camera_data(self, numpy_rgb=False, numpy_seg=False, numpy_depth=False,
                        tensor_ptd=False, ptd_in_robot_base=False, segmented_ptd=True,
                        ptd_downscale=1):

        cam_data = {}
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        numpy_render, tensor_pointcloud = None, None

        if numpy_rgb or numpy_seg or numpy_depth:
            numpy_render = self.get_numpy_data(self.cameras, rgb=numpy_rgb, seg=numpy_seg, depth=numpy_depth)
            cam_data['camera_render_raw'] = numpy_render
        if tensor_ptd:
            tensor_pointcloud = self.cam_point_clouds.get_point_cloud(downscale=ptd_downscale)
            cam_data['camera_pointcloud_raw'] = tensor_pointcloud

        self.gym.end_access_image_tensors(self.sim)

        if tensor_ptd and segmented_ptd:
            pointcloud = self.filter_pointcloud(tensor_pointcloud, convert_robot_base=ptd_in_robot_base)
            cam_data['camera_pointcloud_seg'] = pointcloud

            if self.debug_viz and self.viewer is not None:
                for i in range(self.num_envs):
                    self.ptd_vis_debug(pointcloud[i], env_idx=i)
                    self.ptd_vis_debug2(tensor_pointcloud[i], env_idx=i)

        return cam_data

    def get_numpy_data(self, camera_handles, rgb=False, seg=False, depth=False):
        rgb_obs_buf, rgb_seg_buf, rgb_dep_buf = [], [], []

        for cam_handles, env in zip(camera_handles, self.envs):
            cam_ob, cam_seg, cam_dep = [], [], []
            for cam_handle in cam_handles:
                if rgb:
                    color_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                    cam_ob.append(color_image)
                if seg:
                    seg_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    seg_image = seg_image.reshape(seg_image.shape[0], -1, 1)
                    cam_seg.append(seg_image)
                if depth:
                    depth_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_DDEPTH)
                    depth_image = depth_image.reshape(depth_image.shape[0], -1, 1)
                    cam_dep.append(depth_image)

            if rgb:
                rgb_obs_buf.append(cam_ob)
            if seg:
                rgb_seg_buf.append(cam_seg)
            if depth:
                rgb_dep_buf.append(cam_dep)

        buf = {
            'rgb': rgb_obs_buf,
            'seg': rgb_seg_buf,
            'depth': rgb_dep_buf
        }

        return buf

    """
    Debug
    """

    def ptd_vis_debug(self, points, env_idx=0):
        scene = trimesh.Scene()

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        scene_pos, scene_quat = (self._scene_base_state[env_idx, :3].cpu().numpy(),
                                 self._scene_base_state[env_idx, 3:7].cpu().numpy())
        scene_quat = np.concatenate([scene_quat[-1:], scene_quat[:-1]], axis=-1)
        scene_translation = tr.translation_matrix(scene_pos)
        scene_rotation = tr.quaternion_matrix(scene_quat)

        for f in self.scene_asset[env_idx]['files']:
            mesh = trimesh.load(f)
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        object_state = self._root_state[env_idx, 3:]
        oq = torch.concat([object_state[..., 6:7], object_state[..., 3:6]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_state[..., :3].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
            trans = tr.translation_matrix(ot[i])
            rot = tr.quaternion_matrix(oq[i])
            mesh = o['mesh'].copy().apply_transform(trans @ rot)
            scene.add_geometry(mesh)

        robot_color = np.array([[200, 0, 0, 100]], dtype=np.uint8)
        goal_color = np.array([[0, 50, 200, 100]], dtype=np.uint8)
        scene_color = np.array([[0, 200, 0, 100]], dtype=np.uint8)

        robot_cloud = trimesh.points.PointCloud(points['robot'].cpu().numpy(), colors=robot_color.repeat(len(points['robot']), axis=0))
        scene.add_geometry(robot_cloud)

        goal_cloud = trimesh.points.PointCloud(points['goal'].cpu().numpy(), colors=goal_color.repeat(len(points['goal']), axis=0))
        scene.add_geometry(goal_cloud)

        scene_cloud = trimesh.points.PointCloud(points['scene'].cpu().numpy(), colors=scene_color.repeat(len(points['scene']), axis=0))
        scene.add_geometry(scene_cloud)

        scene.show()

    def ptd_vis_debug2(self, ptd_raw, env_idx=0):
        scene = trimesh.Scene()

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        scene_pos, scene_quat = (self._scene_base_state[env_idx, :3].cpu().numpy(),
                                 self._scene_base_state[env_idx, 3:7].cpu().numpy())
        scene_quat = np.concatenate([scene_quat[-1:], scene_quat[:-1]], axis=-1)
        scene_translation = tr.translation_matrix(scene_pos)
        scene_rotation = tr.quaternion_matrix(scene_quat)

        for f in self.scene_asset[env_idx]['files']:
            mesh = trimesh.load(f)
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        object_state = self._root_state[env_idx, 3:]
        oq = torch.concat([object_state[..., 6:7], object_state[..., 3:6]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_state[..., :3].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
            trans = tr.translation_matrix(ot[i])
            rot = tr.quaternion_matrix(oq[i])
            mesh = o['mesh'].copy().apply_transform(trans @ rot)
            scene.add_geometry(mesh)

        cam_ptds = ptd_raw['cam_pts']
        cam_poses = ptd_raw['cam_poses']

        for i, (pts, pose) in enumerate(zip(cam_ptds, cam_poses)):
            pts = pts.cpu().numpy()
            pose = pose.cpu().numpy()

            homo_world_pts = pose @ pts.T

            homo_color = np.array([[0, 0, 0, 100]], dtype=np.uint8)
            homo_color[0][i] = 200

            cloud = trimesh.points.PointCloud(homo_world_pts.T[:, :-1], colors=homo_color.repeat(pts.shape[0], 0))
            scene.add_geometry(cloud)

        scene.show()







