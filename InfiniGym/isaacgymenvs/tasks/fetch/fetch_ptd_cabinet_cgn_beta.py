
import numpy as np
import torch
import time
import os
import sys
import trimesh

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd_cabinet import FetchPtdCabinet
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video, create_gripper_marker, plot_trajs

from isaacgymenvs.tasks.fetch.utils.contact_graspnet_utils import ContactGraspNet, CGN_PATH


class FetchPtdCabinetCGNBeta(FetchPtdCabinet):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.grasp_net = ContactGraspNet(
            root_dir=CGN_PATH,
            ckpt_dir=f'{CGN_PATH}/checkpoints/contact_graspnet',
            forward_passes=self.cfg["solution"]["cgn"]["num_forward_passes"]
        )
        self.cgn_log_dir = f'./logs/cgn_log/{self.cfg["experiment_name"]}'
        if not os.path.exists(self.cgn_log_dir):
            os.makedirs(self.cgn_log_dir)

        assert self.arm_control_type == 'joint'
        assert self.num_envs == 1

    """
    Sample Utils
    """

    def get_robot_world_matrix(self, env_idx, device):
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        rq = torch.concat([rq[:, -1:], rq[:, :-1]], dim=-1).cpu().numpy()
        rt = rt.cpu().numpy()

        trans = trimesh.transformations.translation_matrix(rt[env_idx])
        rot = trimesh.transformations.quaternion_matrix(rq[env_idx])
        pose = trans @ rot

        pose = to_torch([pose], device=device)
        return pose

    """
    Motion Gen
    """

    def motion_gen_to_pose_goalset_single_env(self, target_poses, env_idx, offset=False):
        # motion generation to grasp the object
        self._refresh()

        q_start = JointState.from_position(
            self.states["q"][env_idx:env_idx+1, :-2].clone().to(self.tensor_args.device),
            joint_names=self.robot_joint_names
        )

        # get success mask
        if target_poses is None:
            return None, False, None, None

        assert len(target_poses.shape) == 3
        config = self.motion_plan_config_line.clone() if offset else self.motion_plan_config_graph.clone()
        result = self.motion_generators[env_idx].plan_goalset(q_start, target_poses, config)
        traj = result.get_interpolated_plan()

        if self.debug_viz and self.viewer is not None:

            traj_state = JointState.from_position(
                traj.position,
                joint_names=self.robot_joint_names
            )
            traj_pose = self.ik_solver.fk(traj_state.position).ee_pose
            target_pose = target_poses[0]

            self.motion_vis_debug(self.motion_generators[env_idx], target_pose, traj_pose)

        return traj, result.success.cpu().numpy()[0], target_poses, result

    def motion_gen_to_grasp_pose_ordered(self, command_poses, lsts):

        trajs, poses, success, results = [], [], [], []
        for i in range(self.num_envs):
            lst = lsts[i]
            for l in lst:
                target_poses = command_poses[i][l].unsqueeze(dim=0).clone()
                traj, suc, pose, result = self.motion_gen_to_pose_goalset_single_env(target_poses, env_idx=i)

                if suc:
                    trajs.append(traj)
                    poses.append(pose)
                    results.append(result)
                    success.append(suc)
                    break

            if len(trajs) < i + 1:
                trajs.append(None)
                poses.append(None)
                results.append(None)
                success.append(False)

        return self._get_batch_joint_trajs(trajs), success, poses, results

    def sample_goal_obj_collision_free_grasp_pose(self):
        cam_data = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=False, segmented_ptd=False)
        cgn_input = self.get_cgn_input(cam_data)

        cgn_coord_convert = to_torch([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float,
                                     device=self.device)

        cgn_pred = self.grasp_net.single_ptd_inference(cgn_input, local_regions=True, filter_grasps=True,
                                                       forward_passes=self.cfg["solution"]["cgn"]["num_forward_passes"])
        grasp_poses, pre_grasp_poses, grasp_successes, grasp_lsts = [], [], [], []
        grasp_command_offset = to_torch([[[0, -1, 0, 0], [1, 0, 0, 0],
                                         [0, 0, 1, 0], [0, 0, 0, 1]]], device=self.tensor_args.device)

        for env_idx in range(self.num_envs):
            if len(cgn_pred[env_idx]['grasp_poses'][1]) == 0:
                grasp_poses.append([])
                pre_grasp_poses.append([])
                grasp_successes.append([])
                grasp_lsts.append([])
                continue

            grasp_in_cam = to_torch(cgn_pred[env_idx]['grasp_poses'][1], device=self.tensor_args.device)
            cam_pose = to_torch(cgn_pred[env_idx]['cam_pose'], device=self.tensor_args.device)

            grasp_in_world = (cam_pose @ cgn_coord_convert).reshape(1, 4, 4) @ grasp_in_cam
            robot_world_pose = self.get_robot_world_matrix(env_idx, self.tensor_args.device).reshape(1, 4, 4)
            grasp_in_robot = robot_world_pose @ grasp_in_world @ grasp_command_offset

            grasp_pose = Pose.from_matrix(grasp_in_robot)
            pre_grasp_offset_pos = to_torch([0, 0, -self.cfg["solution"]["pre_grasp_offset"]],
                                            device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_pos = pre_grasp_offset_pos.unsqueeze(dim=0)
            pre_grasp_offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_quat = pre_grasp_offset_quat.unsqueeze(dim=0)
            pre_grasp_offset = Pose(pre_grasp_offset_pos, pre_grasp_offset_quat)
            pre_grasp_pose = grasp_pose.multiply(pre_grasp_offset)

            grasp_lst = self.get_grasp_masks(cgn_pred[env_idx]['scores'][1])
            # create grasp mask
            grasp_success = torch.zeros((grasp_in_cam.shape[0], ), dtype=torch.long, device=self.device)
            grasp_success[grasp_lst] += 1

            grasp_poses.append(grasp_pose)
            pre_grasp_poses.append(pre_grasp_pose)
            grasp_successes.append(grasp_success)
            grasp_lsts.append(grasp_lst)

            if self.debug_viz and self.viewer is not None:
                self.grasp_pose_vis_debug(self.motion_generators[env_idx], grasp_pose,
                                          pre_grasp_pose, cgn_pred[env_idx]['pc_input'][1],
                                          (robot_world_pose @ cam_pose @ cgn_coord_convert).cpu().numpy(),
                                          grasp_success)

        res = {
            'grasp_poses': grasp_poses,
            'pre_grasp_poses': pre_grasp_poses,
            'grasp_success': grasp_successes,
            'grasp_ordered_lst': grasp_lsts
        }

        return res, cgn_pred
    """
    Solve
    """

    def solve(self):
        log = {}

        # set goal obj color
        self.set_target_color()
        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()

        start_time = time.time()
        cgn_result, cgn_logs = self.sample_goal_obj_collision_free_grasp_pose()
        if self.cgn_log_dir is not None:
            np.save(f'{self.cgn_log_dir}/log_{self.get_task_idx()}.npy', cgn_logs)
        computing_time += time.time() - start_time

        grasp_lst = cgn_result['grasp_ordered_lst'][0]
        if len(grasp_lst) == 0:
            pass
        else:
            pre_grasp_poses = cgn_result['pre_grasp_poses'][0][grasp_lst].get_matrix()
            grasp_poses = cgn_result['grasp_poses'][0][grasp_lst].get_matrix()

            if self.cfg["solution"]["mppi"]["update_model_center"]:
                self.update_mppi_model_center()

            self.mppi_policy.reset(state=0)
            self.mppi_policy.update_goal(pre_grasp_poses.cpu().numpy(), grasp_poses.cpu().numpy())

            self.current_plan = None
            for s in range(self.cfg["solution"]["num_grasp_steps"]):
                st = time.time()
                self.update_mppi_state(add_goal_obj=False)
                plan = self.mppi_policy.get_plan()
                self._set_plan(plan)
                computing_time += time.time() - st

                if self.current_plan is None:
                    break

                r_steps = int(self.cfg["solution"]["control_freq"] /
                              (self.cfg["sim"]["dt"] * self.cfg["solution"]["num_step_repeat_per_plan_dt"]))
                for i in range(r_steps):
                    q = self._get_next_q_in_plan(self.states["q"][0].cpu().numpy())
                    self.step_q(q, gripper_state=0)

        print("Grasp Phase End")
        log['grasp_execute_error'] = self.get_end_effect_error([self.mppi_policy.cur_grasps_final])

        self.close_gripper()
        log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Gripper Close End")

        # move retract offset
        if self.cfg["solution"]["retract_offset"] > 0:
            offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
            log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

        self.mppi_policy.reset(state=2)
        self.current_plan = None

        free_space_pose = np.array([[[0.,   -1.,    0.,   -0.2], [-1.,    0.,   0.,   -0.25],
                                     [0.,    0.,   -1.,    0.66], [0.,    0.,    0.,    1.]],
                                    [[0.,    1.,    0.,   -0.2], [1.,    0.,    0.,    0.25],
                                     [0.,    0.,   -1.,    0.66], [0.,    0.,    0.,    1.]]])

        self.mppi_policy.update_goal(free_space_pose, free_space_pose)

        for s in range(self.cfg["solution"]["num_fetch_steps"]):
            st = time.time()
            self.update_mppi_state(add_goal_obj=True)

            plan = self.mppi_policy.get_plan()
            self._set_plan(plan)
            computing_time += time.time() - st

            if self.current_plan is None:
                break

            r_steps = int(self.cfg["solution"]["control_freq"] /
                          (self.cfg["sim"]["dt"] * self.cfg["solution"]["num_step_repeat_per_plan_dt"]))
            for i in range(r_steps):
                q = self._get_next_q_in_plan(self.states["q"][0].cpu().numpy())
                self.step_q(q, gripper_state=-1)

        log['fetch_execute_error'] = self.get_end_effect_error([self.mppi_policy.cur_grasps_final])

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log
