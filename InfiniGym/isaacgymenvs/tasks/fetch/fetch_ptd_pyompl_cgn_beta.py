
import numpy as np
import os
import torch
import trimesh
import time

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd_pyompl import FetchPtdPyompl
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video, create_gripper_marker, plot_trajs

import sys

from isaacgymenvs.tasks.fetch.utils.contact_graspnet_utils import ContactGraspNet, CGN_PATH


class FetchPtdPyomplCGNBeta(FetchPtdPyompl):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.grasp_net = ContactGraspNet(
            root_dir=CGN_PATH,
            ckpt_dir=f'{CGN_PATH}/checkpoints/'
                     f'{self.cfg["solution"]["cgn"].get("ckpt_name", "contact_graspnet")}',
            forward_passes=self.cfg["solution"]["cgn"]["num_forward_passes"]
        )
        self.cgn_log_dir = f'./logs/cgn_log/{self.cfg["experiment_name"]}'
        if not os.path.exists(self.cgn_log_dir):
            os.makedirs(self.cgn_log_dir)

    """
    Sample Utils
    """

    def get_robot_world_matrix(self, env_idx):
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        rq = torch.concat([rq[:, -1:], rq[:, :-1]], dim=-1).cpu().numpy()
        rt = rt.cpu().numpy()

        trans = trimesh.transformations.translation_matrix(rt[env_idx])
        rot = trimesh.transformations.quaternion_matrix(rq[env_idx])
        pose = trans @ rot

        return pose

    def sample_goal_obj_collision_free_grasp_pose(self):
        cam_data = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=False, segmented_ptd=False)
        cgn_coord_convert = to_torch([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float,
                                     device=self.device)

        cgn_input = self.get_cgn_input(cam_data)
        cgn_pred = self.grasp_net.single_ptd_inference(cgn_input, local_regions=True, filter_grasps=True,
                                                       forward_passes=self.cfg["solution"]["cgn"]["num_forward_passes"])
        grasp_poses, pre_grasp_poses, grasp_successes, grasp_lsts = [], [], [], []
        grasp_command_offset = np.array([[[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])

        for env_idx in range(self.num_envs):
            if len(cgn_pred[env_idx]['grasp_poses'][1]) == 0:
                grasp_poses.append([])
                pre_grasp_poses.append([])
                grasp_successes.append([])
                grasp_lsts.append([])
                continue

            grasp_in_cam = cgn_pred[env_idx]['grasp_poses'][1]
            cam_pose = cgn_pred[env_idx]['cam_pose']
            cgn_coord = cgn_coord_convert.cpu().numpy()

            grasp_in_world = (cam_pose @ cgn_coord).reshape(1, 4, 4) @ grasp_in_cam
            robot_world_pose = self.get_robot_world_matrix(env_idx).reshape(1, 4, 4)
            grasp_in_robot = robot_world_pose @ grasp_in_world @ grasp_command_offset

            pre_grasp_offset_pos = trimesh.transformations.translation_matrix([0, 0, -self.cfg["solution"]["pre_grasp_offset"]])
            pre_grasp_pose = grasp_in_robot @ pre_grasp_offset_pos.reshape(1, 4, 4)

            grasp_lst = self.get_grasp_masks(cgn_pred[env_idx]['scores'][1])
            # create grasp mask
            grasp_success = np.zeros((grasp_in_cam.shape[0], ), dtype=np.int)
            grasp_success[grasp_lst] += 1

            grasp_poses.append(grasp_in_robot)
            pre_grasp_poses.append(pre_grasp_pose)
            grasp_successes.append(grasp_success)
            grasp_lsts.append(grasp_lst)

            if self.debug_viz and self.viewer is not None:
                self.grasp_pose_vis_debug(self.motion_generator[env_idx], grasp_in_robot,
                                          pre_grasp_pose, cgn_pred[env_idx]['pc_input'][1],
                                          robot_world_pose @ cam_pose @ cgn_coord,
                                          grasp_success)

        res = {
            'grasp_poses': grasp_poses,
            'pre_grasp_poses': pre_grasp_poses,
            'grasp_success': grasp_successes,
            'grasp_ordered_lst': grasp_lsts
        }

        return res, cgn_pred

    """
    Motion Plan Utils
    """

    def motion_gen_to_grasp_pose_ordered(self, command_poses, lsts):
        assert self.num_envs == 1
        for i in range(self.num_envs):
            lst = lsts[i]
            for l in lst:
                target_poses = command_poses[i][l].reshape(1, 1, 4, 4)
                traj, suc, pose = self.motion_gen_to_pose_goalset(target_poses)

                if suc[i]:
                    return traj, suc, pose

            return None, [False], [None]

    """
    Solve
    """

    def solve(self):
        # set goal obj color
        log = {}

        self.set_target_color()
        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()

        # Sample Good Grasp Pose
        self.update_ptd_motion_gen_world(attach_goal_obj=False)

        start_time = time.time()
        cgn_result, cgn_logs = self.sample_goal_obj_collision_free_grasp_pose()
        if self.cgn_log_dir is not None:
            np.save(f'{self.cgn_log_dir}/log_{self.get_task_idx()}.npy', cgn_logs)

        traj, success, poses = self.motion_gen_to_grasp_pose_ordered(cgn_result['pre_grasp_poses'],
                                                                     lsts=cgn_result['grasp_ordered_lst'])
        print("Pre Grasp Plan", success)
        log['pre_grasp_plan_success'] = success
        computing_time += time.time() - start_time

        if traj is not None:
            self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
        print("Pre Grasp Phase End")
        log['pre_grasp_execute_error'] = self.get_end_effect_error(poses)

        offset = np.array([0, 0, self.cfg["solution"]["pre_grasp_offset"] *
                           self.cfg["solution"]["grasp_overshoot_ratio"]])
        self.follow_cartesian_linear_motion(offset, gripper_state=0)

        print("Grasp Phase End")

        self.close_gripper()
        log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Gripper Close End")

        if self.cfg["solution"]["retract_offset"] > 0:
            offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
            log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

        # Fetch Phase
        self.update_ptd_motion_gen_world(attach_goal_obj=True)

        start_time = time.time()
        traj, success, poses = self.motion_gen_to_free_space(mask=success)
        print("Fetch Plan", success)
        log['fetch_plan_success'] = success
        computing_time += time.time() - start_time

        self.follow_motion_trajs(traj, gripper_state=-1)  # 0 means no movement
        print("Fetch Phase End")
        log['fetch_execute_error'] = self.get_end_effect_error(poses)

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log

    """
    Debug Visualization
    """

    def grasp_pose_vis_debug(self, motion_gen, grasp_poses, pre_grasp_poses, pt, cam_pose, grasp_success):
        scene = trimesh.Scene()

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        for i, o in enumerate(motion_gen.pb_ompl_interface.tm_obstacles):
            scene.add_geometry(o)

        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(grasp_poses.shape[0]):
            if not grasp_success[i]:
                continue

            grasp = grasp_poses[i] @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        for i in range(pre_grasp_poses.shape[0]):
            if not grasp_success[i]:
                continue

            grasp = pre_grasp_poses[i] @ vis_rot
            command_marker = create_gripper_marker([0, 255, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        pt = np.concatenate([pt, np.ones_like(pt[:, :1])], axis=-1)
        pt = (cam_pose.reshape(4, 4) @ pt.T).T[:, :3]

        cloud = trimesh.points.PointCloud(pt, colors=np.array([[0, 0, 200, 100]]).repeat(pt.shape[0], 0))
        scene.add_geometry(cloud)

        scene.show()