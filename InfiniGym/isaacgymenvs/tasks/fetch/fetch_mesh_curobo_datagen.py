
import numpy as np
import os
import torch
import trimesh.transformations as tr
import trimesh
import time

# cuRobo
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sphere_fit import SphereFitType


from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import (to_torch, get_axis_params, tensor_clamp,
                                                tf_vector, tf_combine, quat_mul, quat_conjugate,
                                                quat_to_angle_axis, tf_inverse, quat_apply,
                                                matrix_to_quaternion)

from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import create_gripper_marker, plot_trajs, FetchMeshCurobo

import sys
sys.path.append('../third_party/Optimus')
import optimus.modules.functional as F
import trimesh.transformations as tra

SPHERE_TYPE = {
    0: SphereFitType.SAMPLE_SURFACE,
    1: SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
}


class FetchCuroboDataGen(FetchMeshCurobo, FetchPointCloudBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        assert self.num_envs == 1, "Data Gen only support one object."

        self._trajs = []

        self._log_mode = self.cfg["solution"]["log_mode"]
        self._log_scene_ptd = self.cfg["solution"]["log_scene_ptd"]
        self._log_cam_ptd = self.cfg["solution"]["log_cam_ptd"]
        self._log_cam_render = self.cfg["solution"]["log_cam_render"]
        self._log_state = self.cfg["solution"]["log_state"]

    """
    Logger Utils
    """
    def _process_cam_render(self, depths, segs, cam_poses, env_idx=0):
        robot_seg_id = 1
        goal_seg_id = self.task_obj_index[env_idx][self.get_task_idx()] + 4

        base_pos = self._robot_base_state.cpu().numpy()[env_idx][:3]
        base_rot = self._robot_base_state.cpu().numpy()[env_idx][3:7]
        base_rot = np.concatenate([base_rot[-1:], base_rot[:-1]], axis=0)
        base_matrix = (tra.translation_matrix(base_pos) @ tra.quaternion_matrix(base_rot)).astype(np.float32)
        coord_convert = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

        p_depths, p_segs, p_poses = [], [], []
        for c in range(len(depths)):
            depth, seg = depths[c], segs[c]
            new_seg = torch.zeros_like(seg)
            new_seg[seg == robot_seg_id] = 1
            new_seg[seg == goal_seg_id] = 2
            depth *= -1.    # convert to standard depth_map

            valid_depth = ((depth >= self.cfg["env"]["cam"]["depth_min"]) &
                           (depth <= self.cfg["env"]["cam"]["depth_max"]))
            depth[~valid_depth] = 0

            depth = depth.cpu().numpy().astype(np.float16)
            new_seg = new_seg.cpu().numpy().astype(np.uint8)

            pc_trans = tra.inverse_matrix(base_matrix) @ cam_poses[c].cpu().numpy() @ coord_convert

            p_depths.append(depth)
            p_segs.append(new_seg)
            p_poses.append(pc_trans)

        return p_depths, p_segs, p_poses

    def get_curr_state(self, downscale=None):
        state = {}

        if downscale is None:
            downscale = self.cfg["solution"]["log_ptd_downscale"]

        tensor_ptd = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=True, segmented_ptd=True,
                                          ptd_downscale=downscale)
        if self._log_state:
            state = {
                'dof_state': self._dof_state.cpu().numpy()[0],
                'root_state': self._root_state.cpu().numpy()[0],
                'eef_state': self._eef_state.cpu().numpy()[0],
                'rigid_shape_state': self._rigid_body_state.cpu().numpy()[0],
                'camera_pose': self.task_camera_init_state[0][self._task_idx],
                'cam_poses': [t.cpu().numpy() for t in tensor_ptd['camera_pointcloud_raw'][0]['cam_poses']]
            }
        if self._log_scene_ptd:
            state['scene_point_cloud'] = dict()
            for k in ['scene', 'goal', 'robot']:
                ptc = tensor_ptd['camera_pointcloud_seg'][0][k]
                ptc = ptc[ptc.norm(dim=-1) <= self.cfg["solution"]["log_pts_range"]]
                state['scene_point_cloud'][k] = ptc.cpu().numpy()
        if self._log_cam_ptd:
            state['raw_point_cloud'] = {
                'segs': [t.cpu().numpy().astype(np.uint8) for t in tensor_ptd['camera_pointcloud_raw'][0]['seg']],
                'cam_pts': [t.cpu().numpy().astype(np.float16) for t in tensor_ptd['camera_pointcloud_raw'][0]['cam_pts']]
            }
        if self._log_cam_render:
            # process cam_renders
            depths, segs = tensor_ptd['camera_pointcloud_raw'][0]['raw_depths'], \
                tensor_ptd['camera_pointcloud_raw'][0]['raw_segs']
            depths, segs, pc_trans = self._process_cam_render(depths, segs,
                                                             tensor_ptd['camera_pointcloud_raw'][0]['cam_poses'])

            state['raw_cam_render'] = {
                'depths': depths,
                'segs': segs,
                'cam_poses': pc_trans
            }

        return state

    """
    Sample Grasp Pose
    """

    def _sample_goal_obj_annotated_grasp_pose(self):
        pose = self._get_pose_in_robot_frame()
        oq, ot = pose['object']['quat'], pose['object']['pos']

        sample_grasps = []
        for i in range(self.num_envs):
            goal_idx = self.task_obj_index[i][self.get_task_idx()]
            grasp_pose = self.obj_grasp_poses[i][goal_idx].to(self.tensor_args.device)

            sample_quat, sample_pos = grasp_pose[..., 3:7], grasp_pose[..., :3]
            oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(sample_quat.shape[0], 1),
                          ot[i:i+1, goal_idx].repeat(sample_pos.shape[0], 1))
            gq, gt = tf_combine(oq_i, ot_i, sample_quat, sample_pos)
            gq = torch.concat([gq[..., -1:], gq[..., :-1]], dim=-1)

            sample_grasps.append(torch.concat([gt, gq], dim=-1))

        sample_grasps = torch.stack(sample_grasps, dim=0)
        return sample_grasps

    def motion_gen_to_grasp_pose(self, poses, mask):
        target_poses = []
        for i, m in enumerate(mask):
            if m:
                target_poses.append(poses[i:i+1].unsqueeze(dim=0))
            else:
                target_poses.append(None)

        return self.motion_gen_to_pose_goalset(target_poses)

    """
    Arm Control
    """

    def follow_motion_trajs(self, trajs, gripper_state):
        # follow the traj
        log = []
        executed_pos, executed_vel = [], []
        for step in range(trajs.shape[1]):
            traj_command = {"joint_state": trajs.position[:, step].clone().to(self.device)}
            if gripper_state == 0:
                traj_command['gripper_state'] = None
            else:
                traj_command['gripper_state'] = gripper_state * torch.ones((self.num_envs,), device=self.device)

            for i in range(self.cfg["solution"]["num_step_repeat_per_plan_dt"]):
                self.pre_phy_step(traj_command)
                self.env_physics_step()
                self.post_phy_step()
                state = self.get_curr_state()
                if len(state) > 0 and self._log_mode == 'trajectory':
                    log.append(state)

            executed_pos.append(self.states["q"].clone())
            executed_vel.append(self.states["qd"].clone())

        if len(executed_pos) == 0:
            return

        if self.debug_viz and self.viewer is not None:

            executed_pos = torch.stack(executed_pos, dim=1).to(self.tensor_args.device)
            executed_vel = torch.stack(executed_vel, dim=1).to(self.tensor_args.device)
            for i in range(self.num_envs):
                traj = trajs[i]

                target_state = JointState.from_position(
                    traj.position,
                    joint_names=self.robot_joint_names
                )
                target_pose = self.ik_solver.fk(target_state.position).ee_pose

                traj_state = JointState.from_position(
                    executed_pos[i][..., :-2],
                    joint_names=self.robot_joint_names
                )
                traj_pose = self.ik_solver.fk(traj_state.position).ee_pose

                # plot and save the joint error
                plot_trajs([
                    [traj.position.cpu().numpy(), traj.velocity.cpu().numpy()],
                    [executed_pos[i].cpu().numpy(), executed_vel[i].cpu().numpy()]
                ], self.cfg["solution"]["cuRobo"]["motion_interpolation_dt"])

                self.motion_vis_debug(self.motion_generators[i], target_pose, traj_pose)

        return log

    def follow_cartesian_linear_motion(self, offset, gripper_state, eef_frame=True):
        log = []

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
                state = self.get_curr_state()
                if len(state) > 0 and self._log_mode == 'trajectory':
                    log.append(state)

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

        # switch back to default control
        self.arm_control_type = old_arm_control_type
        self.switch_arm_control_type('joint')

        return log

    """
    Gripper Control
    """

    def close_gripper(self):
        log = []
        self._refresh()

        curr_states = self.states["q"].clone()
        grasp_command = {
            "joint_state": curr_states[:, :-2],
            "gripper_state": - torch.ones((self.num_envs,), device=self.device, dtype=torch.float)
        }

        # close the gripper
        for _ in range(self.cfg["solution"]["gripper_steps"]):
            self.pre_phy_step(grasp_command)
            self.env_physics_step()
            self.post_phy_step()
            state = self.get_curr_state()
            if len(state) > 0 and self._log_mode == 'trajectory':
                log.append(state)

        return log

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
    Reset 
    """

    def reset_idx(self, env_ids):
        pos = self.robot_default_dof_pos.unsqueeze(0)
        pos = pos.repeat(len(env_ids), 1)

        if self.cfg["solution"]["randomize_init_state"]:
            rand_n = torch.randn(*pos.shape, device=pos.device) * 0.01
            pos = tensor_clamp(pos + rand_n, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

            pos[:, -2:] = self.robot_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        self._vel_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._vel_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        multi_env_ids_actors_int32 = self._global_indices.flatten()
        self._root_state[env_ids, :] = self.task_actor_init_state[env_ids, self._task_idx, :]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_actors_int32), len(multi_env_ids_actors_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0

        self.env_physics_step()
        self._refresh()


    """
    Data Generation
    """

    def solve(self):
        # set goal obj color
        self._trajs = []
        self.set_target_color()

        # Sample Good Grasp Pose
        self.update_cuRobo_world_collider_pose()
        ik_result = self.sample_goal_obj_collision_free_grasp_pose()

        if self.cfg["solution"]["direct_grasp"]:
            grasp_ik_index = ik_result['grasp_success'].nonzero(as_tuple=True)[1]
        else:
            grasp_ik_index = (ik_result['grasp_success'] & ik_result['pre_grasp_success']).nonzero(as_tuple=True)[1]

        planning_content = self.get_curr_state(downscale=1)
        for i in range(grasp_ik_index.shape[0]):
            idx = grasp_ik_index[i]

            traj_log = {}
            traj_log['task_obj_idx'] = self.task_obj_index[0][self.get_task_idx()].cpu().numpy()

            if self.cfg["solution"]["direct_grasp"]:
                # Move to Grasp Pose, disable goal_obj collision checking
                traj, success, poses, results = self.motion_gen_to_grasp_pose(ik_result['grasp_poses'][:, idx], mask=[True])
                print("Grasp Plan", success)
                traj_log['solution_config'] = 'direct_grasp'
                traj_log['grasp_plan_result'] = success
                traj_log['grasp_pose'] = ik_result['grasp_poses'][:, idx].get_numpy_matrix()[0]
                traj_log['grasp_traj'] = self.follow_motion_trajs(traj, gripper_state=0)

            else:
                traj, success, poses, results = self.motion_gen_to_grasp_pose(ik_result["pre_grasp_poses"][:, idx], mask=[True])
                print("Pre Grasp Plan", success)

                traj_log['solution_config'] = f'pre_grasp_{self.cfg["solution"]["move_offset_method"]}'
                traj_log['pre_grasp_plan_result'] = success
                traj_log['pre_grasp_pose'] = ik_result['pre_grasp_poses'][:, idx].get_numpy_matrix()[0]
                traj_log['pre_grasp_traj'] = self.follow_motion_trajs(traj, gripper_state=0)
                print("Pre Grasp Phase End")

                # Move to Grasp Pose, disable goal_obj collision checking
                if self.cfg["solution"]["move_offset_method"] == 'motion_planning':
                    self.update_cuRobo_world_collider_pose()
                    if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                        self._enable_goal_obj_collision_checking(False)
                    traj, success, poses, results = self.motion_gen_by_z_offset(z=self.cfg["solution"]["pre_grasp_offset"],
                                                                                mask=success)
                    print("Grasp Plan", success)
                    if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                        self._enable_goal_obj_collision_checking(True)

                    traj_log['grasp_pose'] = ik_result['grasp_poses'][:, idx].get_numpy_matrix()[0]
                    traj_log['grasp_traj'] = self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement

                elif self.cfg["solution"]["move_offset_method"] == 'cartesian_linear':
                    offset = np.array([0, 0, self.cfg["solution"]["pre_grasp_offset"] *
                                       self.cfg["solution"]["grasp_overshoot_ratio"]])

                    traj_log['grasp_pose'] = ik_result['grasp_poses'][:, idx].get_numpy_matrix()[0]
                    traj_log['grasp_traj'] = self.follow_cartesian_linear_motion(offset, gripper_state=0)

            print("Grasp Phase End")

            traj_log['gripper_close_traj'] = self.close_gripper()
            traj_log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
            print("Gripper Close End")

            if self.cfg["solution"]["retract_offset"] > 0:
                offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
                traj_log['retract_traj'] = self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
                traj_log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

            self.update_cuRobo_motion_gen_config(attach_goal_obj=True)
            traj, success, poses, results = self.motion_gen_to_free_space(mask=success)
            print("Fetch Plan", success)
            self.update_cuRobo_motion_gen_config(attach_goal_obj=False)

            traj_log['fetch_plan_result'] = success
            traj_log['fetch_traj'] = self.follow_motion_trajs(traj, gripper_state=-1)
            print("Fetch Phase End")

            traj_log['final_finger_obj_contact'] = self.finger_goal_obj_contact()
            print("Eval Phase End")
            eval_result = self.eval()
            traj_log['evaluation_result'] = eval_result

            if self._log_mode == 'grasp_pose' or (self._log_mode == 'trajectory' and eval_result['success'][0]):
                self._trajs.append(traj_log)

            # reset env
            self.reset_idx(torch.arange(self.num_envs, device=self.device))

            if self._log_mode == 'trajectory' and len(self._trajs) == self.cfg["solution"]["num_max_trajs"]:
                break

        self.set_default_color()

        return self._trajs, {'scene_config': self.scene_config_path[0],
                             'scene_state': planning_content,
                             'cam_params': self.cam_point_clouds.get_cam_params(0)}

    def process_traj(self, traj, type):

        n_traj = []
        if type in ['retract_traj', 'gripper_close_traj']:
            for traj_step in traj:
                n_step = {}
                for n, v in traj_step.items():
                    if n.endswith('point_cloud') or n.endswith('render'):
                        continue
                    n_step[n] = v
                n_traj.append(n_step)

            n_traj = n_traj[::self.cfg["solution"]["log_traj_frame_skip"]]
        elif type in ['grasp_traj', 'fetch_traj']:
            max_traj_steps = len(traj) - 1

            n_traj.append(traj[0])

            s = 0
            while s < max_traj_steps:
                ns = min(s + self.cfg["solution"]["log_traj_frame_skip"], max_traj_steps)
                for _ in range(60):
                    delta_q = np.linalg.norm(traj[ns]['dof_state'][:-2, 0] - traj[s]['dof_state'][:-2, 0], axis=-1)
                    if delta_q < self.cfg["solution"]["log_traj_min_delta_skip"]:
                        ns = min(ns + 1, max_traj_steps)

                n_traj.append(traj[ns])
                s = ns

            for n_step in n_traj:
                for n in ['scene', 'goal', 'robot']:
                    ptc = torch.from_numpy(n_step[f'scene_point_cloud'][n]).to(self.tensor_args.device)
                    if ptc.shape[0] <= self.cfg["solution"][f'log_{n}_pts']:
                        v = ptc
                    else:
                        v = ptc.reshape(-1, *ptc.shape[-2:]).transpose(1, 2)
                        v = F.furthest_point_sample(v, self.cfg["solution"][f'log_{n}_pts']).transpose(1, 2)[0]
                    n_step['scene_point_cloud'][n] = v.cpu().numpy().astype(np.float16)

        return n_traj

    def save_trajs(self, trajs, scene):
        # Custom func to save traj.
        for traj in trajs:
            traj['metadata'] = {
                'cam_params': scene['cam_params'],
                'scene_config': scene['scene_config'],
                'task_idx': self.get_task_idx(),
                'downscale': self.cfg["solution"]["log_ptd_downscale"]
            }

            for k in traj.keys():
                # update traj to save
                if k.endswith('traj'):
                    traj[k] = self.process_traj(traj[k], k)

        return trajs