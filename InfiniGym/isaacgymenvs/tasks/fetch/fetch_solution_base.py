
import numpy as np
import os
import torch
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_base import FetchBase

from curobo.types.math import Pose


class FetchSolutionBase(FetchBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self._init_steps = self.cfg["solution"]["init_steps"]
        self._eval_steps = self.cfg["solution"]["eval_steps"]

        self._solution_video = []
        self._video_freq = self.cfg["solution"]["video_freq"]
        self._video_frame = 0

        assert self.arm_control_type == 'joint'

    """
    Solution Utils
    """

    def repeat(self):
        # curr state
        curr_states = self.states["q"].clone()
        grasp_command = {
            "joint_state": curr_states[:, :-2],
            "gripper_state": - torch.ones((self.num_envs,), device=self.device, dtype=torch.float)
        }

        # close the gripper
        for _ in range(self.cfg["solution"]["eval_steps"]):
            self.pre_phy_step(grasp_command)
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    """
    Grsp Pred Utils
    """

    def get_grasp_masks(self, success):
        indexed_lst = list(enumerate(success))
        k = min(len(indexed_lst), self.cfg["solution"]["cgn"]["top_k"])
        top_k = sorted(indexed_lst, key=lambda x: x[1], reverse=True)[:k]

        top_lst = []
        for i in top_k:
            if i[1] >= self.cfg["solution"]["cgn"]["confidence_th"]:
                top_lst.append(i[0])

        return np.array(top_lst)

    def get_cgn_input(self, cam_data):
        cgn_coord_convert = to_torch([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float,
                                     device=self.device)

        cfg_input = []
        for env_idx in range(self.num_envs):
            robot_seg_id = 1
            goal_seg_id = self.task_obj_index[env_idx][self.get_task_idx()] + 4

            segs = cam_data["camera_pointcloud_raw"][env_idx]["seg"]

            num_goal_pixels = []
            for seg in segs:
                num_goal_pixels.append((seg == goal_seg_id).float().sum().cpu().numpy())

            cam_idx = np.argmax(np.array(num_goal_pixels))
            cam_pose = cam_data["camera_pointcloud_raw"][env_idx]["cam_poses"][cam_idx]

            scene_pts, goal_pts = None, None
            if self.cfg["solution"]["cgn"]["full_ptd"]:
                world_pts = cam_data["camera_pointcloud_raw"][env_idx]["pts"]

                scene_pts, goal_pts = [], []
                for i in range(len(segs)):
                    s_pt = world_pts[i][~(segs[i] == robot_seg_id)]
                    g_pt = world_pts[i][segs[i] == goal_seg_id]
                    scene_pts.append(s_pt)
                    goal_pts.append(g_pt)

                scene_pts = torch.concat(scene_pts, dim=0)
                goal_pts = torch.concat(goal_pts, dim=0)

                scene_pts = torch.concat([scene_pts, torch.ones_like(scene_pts[:, :1])], dim=-1)
                goal_pts = torch.concat([goal_pts, torch.ones_like(goal_pts[:, :1])], dim=-1)

                scene_pts = (cgn_coord_convert @ torch.linalg.inv(cam_pose) @ scene_pts.T).T[:, :3]
                goal_pts = (cgn_coord_convert @ torch.linalg.inv(cam_pose) @ goal_pts.T).T[:, :3]

            else:
                cam_pts = cam_data["camera_pointcloud_raw"][env_idx]["cam_pts"][cam_idx]

                scene_pts = cam_pts[~(segs[cam_idx] == robot_seg_id)]
                goal_pts = cam_pts[segs[cam_idx] == goal_seg_id]

                scene_pts = (cgn_coord_convert @ scene_pts.T).T[:, :3]
                goal_pts = (cgn_coord_convert @ goal_pts.T).T[:, :3]

            cfg_input.append({
                'pc_full': scene_pts.cpu().numpy(),
                'pc_obj': goal_pts.cpu().numpy(),
                'cam_idx': cam_idx,
                'cam_pose': cam_pose.cpu().numpy()
            })

        return cfg_input

    """
    Gripper Control
    """

    def open_gripper(self, num_steps=None):
        self._refresh()

        curr_states = self.states["q"].clone()
        grasp_command = {
            "joint_state": curr_states[:, :-2],
            "gripper_state": torch.ones((self.num_envs,), device=self.device, dtype=torch.float)
        }

        if num_steps is None:
            num_steps = self.cfg["solution"]["gripper_steps"]
        # close the gripper
        for _ in range(num_steps):
            self.pre_phy_step(grasp_command)
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    def close_gripper(self, num_steps=None):
        self._refresh()

        curr_states = self.states["q"].clone()
        grasp_command = {
            "joint_state": curr_states[:, :-2],
            "gripper_state": - torch.ones((self.num_envs,), device=self.device, dtype=torch.float)
        }

        if num_steps is None:
            num_steps = self.cfg["solution"]["gripper_steps"]
        # close the gripper
        for _ in range(num_steps):
            self.pre_phy_step(grasp_command)
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    """
    Contact Info
    """

    def finger_goal_obj_contact(self):

        if self.device == 'cpu':
            contact_dict = self.get_robot_contacts()
            leftfinger_contact, rightfinger_contact = [], []
            for i, contact_list in enumerate(contact_dict):
                l, r = False, False

                for rel in contact_list:
                    goal_obj = f'obj_{self.task_obj_index[i][self.get_task_idx()]}'
                    if ('leftfinger' in rel[0] and goal_obj == rel[1]) or ('leftfinger' in rel[1] and goal_obj == rel[0]):
                        l = True
                    if ('rightfinger' in rel[0] and goal_obj == rel[1]) or ('rightfinger' in rel[1] and goal_obj == rel[0]):
                        r = True
                leftfinger_contact.append(l)
                rightfinger_contact.append(r)
        else:
            leftfinger_force = torch.norm(self._left_finger_force, dim=-1)
            rightfinger_force = torch.norm(self._right_finger_force, dim=-1)
            leftfinger_contact = leftfinger_force.cpu().numpy() > 1.
            rightfinger_contact = rightfinger_force.cpu().numpy() > 1.

        return {
            'left': leftfinger_contact,
            'right': rightfinger_contact
        }

    """
    Camera
    """

    def log_video(self, rgb):
        if self._video_frame % self._video_freq == 0:
            self._solution_video.append(rgb)
        self._video_frame += 1


    """
    Common control
    """

    def follow_cartesian_linear_motion(self, offset, gripper_state, eef_frame=True):
        old_arm_control_type = self.arm_control_type
        self.switch_arm_control_type('osc')

        state = self._get_pose_in_robot_frame()
        eef_pos = state['eef']['pos'].to(self.device)
        eef_quat = state['eef']['quat'].to(self.device)

        current_pose = Pose(
            position=eef_pos, quaternion=torch.concat([eef_quat[:, -1:], eef_quat[:, :-1]], dim=-1)
        )

        step_size = offset / self.cfg["solution"]["num_cartesian_steps"]
        target_poses = []
        for i in range(1, self.cfg["solution"]["num_cartesian_steps"]):
            offset_pose = Pose(
                position=to_torch([step_size * i for _ in range(self.num_envs)], device=self.device, dtype=torch.float),
                quaternion=to_torch([[1, 0, 0, 0] for _ in range(self.num_envs)], device=self.device, dtype=torch.float)
            )
            if eef_frame:
                target_pose = current_pose.multiply(offset_pose)
            else:
                target_pose = offset_pose.multiply(current_pose)

            target_pos, target_quat = target_pose.position.to(self.device), target_pose.quaternion.to(self.device)
            target_quat = torch.cat([target_quat[..., 1:], target_quat[..., :1]], dim=-1)
            target_poses.append([target_pos, target_quat])

        final_offset_pose = Pose(
            position=to_torch([offset for _ in range(self.num_envs)], device=self.device, dtype=torch.float),
            quaternion=to_torch([[1, 0, 0, 0] for _ in range(self.num_envs)],
                                device=self.device,
                                dtype=torch.float)
        )
        if eef_frame:
            final_target_pose = current_pose.multiply(final_offset_pose)
        else:
            final_target_pose = final_offset_pose.multiply(current_pose)
        final_target_pos, final_target_quat = (final_target_pose.position.to(self.device),
                                               final_target_pose.quaternion.to(self.device))
        final_target_quat = torch.cat([final_target_quat[..., 1:], final_target_quat[..., :1]], dim=-1)
        target_poses.append([final_target_pos, final_target_quat])

        if gripper_state == 0:
            gripper_command = None
        else:
            gripper_command = gripper_state * torch.ones((self.num_envs,), device=self.device)

        for step in range(len(target_poses)):
            for i in range(self.cfg["solution"]["num_osc_repeat"]):
                command = {
                    'eef_pos': target_poses[step][0],
                    'eef_quat': target_poses[step][1],
                    'gripper_state': gripper_command
                }

                self.pre_phy_step(command, robot_base=True)
                self.env_physics_step()
                self.post_phy_step()
                rgb, seg = self.get_camera_image(rgb=True, seg=False)
                self.log_video(rgb)

            if self.debug_viz and self.viewer is not None:
                curr_pose = self._get_pose_in_robot_frame()['eef']
                cmd_pos = command['eef_pos']
                cmd_quat = command['eef_quat']
                curr_pos = curr_pose['pos']
                curr_quat = curr_pose['quat']

                err_pos = torch.norm(cmd_pos - curr_pos, dim=-1)
                err_quat = quat_mul(cmd_quat, quat_conjugate(curr_quat))
                err_rot = torch.norm(err_quat[..., :3], dim=-1)
                print(f'Step {step}:', err_pos, err_rot)

        # switch back to default control
        self.arm_control_type = old_arm_control_type
        self.switch_arm_control_type('joint')

