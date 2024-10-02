
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

from isaacgymenvs.tasks.fetch.fetch_solution_base import FetchSolutionBase


SPHERE_TYPE = {
    0: SphereFitType.SAMPLE_SURFACE,
    1: SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
}


def image_to_video(obs_buf):
    video = []
    for s, images in enumerate(obs_buf):
        steps = []
        for e, imgs in enumerate(images):
            steps.append(np.concatenate(imgs, axis=0))
        video.append(np.concatenate(steps, axis=1))
    return video


class FetchMeshCurobo(FetchSolutionBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.tensor_args = TensorDeviceType()

        ik_config = IKSolverConfig.load_from_robot_config(
            self._get_cuRobo_robot_config(),
            self._get_cuRobo_world_config(),
            rotation_threshold=self.cfg["solution"]["cuRobo"]["ik_rot_th"],
            position_threshold=self.cfg["solution"]["cuRobo"]["ik_pos_th"],
            num_seeds=self.cfg["solution"]["cuRobo"]["ik_num_seed"],
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=self.cfg["solution"]["cuRobo"]["collision_activation_dist"]
        )
        self.ik_solver = IKSolver(ik_config)
        self.ik_collision = self.ik_solver.world_coll_checker

        # setup cuRobo batch motion gen (for each env)
        self.motion_generators, self.motion_generator_colliders = [], []
        world_cuRobo_cfg_list = self._get_cuRobo_world_config()
        for i in range(self.num_envs):
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self._get_cuRobo_robot_config(),
                world_cuRobo_cfg_list[i],
                tensor_args=self.tensor_args,
                trajopt_tsteps=self.cfg["solution"]["cuRobo"]["motion_trajopt_steps"],
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=False,
                num_trajopt_seeds=self.cfg["solution"]["cuRobo"]["trjopt_num_seed"],
                num_graph_seeds=self.cfg["solution"]["cuRobo"]["graph_num_seed"],
                num_ik_seeds=self.cfg["solution"]["cuRobo"]["ik_num_seed"],
                interpolation_dt=self.cfg["solution"]["cuRobo"]["motion_interpolation_dt"],
                collision_activation_distance=self.cfg["solution"]["cuRobo"]["collision_activation_dist"],
                interpolation_steps=self.cfg["solution"]["cuRobo"]["motion_interpolation_steps"],
                self_collision_check=True,
                self_collision_opt=True,
                maximum_trajectory_dt=0.1,
                rotation_threshold=self.cfg["solution"]["cuRobo"]["ik_rot_th"],
                position_threshold=self.cfg["solution"]["cuRobo"]["ik_pos_th"],
            )
            motion_generator = MotionGen(motion_gen_config)
            motion_generator.reset()

            self.motion_generators.append(motion_generator)
            self.motion_generator_colliders.append(motion_generator.world_coll_checker)

        self.motion_plan_config_graph = MotionGenPlanConfig(
            enable_graph=self.cfg["solution"]["cuRobo"]["enable_graph"],
            enable_opt=self.cfg["solution"]["cuRobo"]["enable_opt"],
            max_attempts=self.cfg["solution"]["cuRobo"]["motion_gen_max_attempts"],
            timeout=self.cfg["solution"]["cuRobo"]["motion_gen_timeout"],
            enable_finetune_trajopt=False
        )

        self.motion_plan_config_line = MotionGenPlanConfig(
            enable_graph=False,
            enable_opt=self.cfg["solution"]["cuRobo"]["enable_opt"],
            max_attempts=self.cfg["solution"]["cuRobo"]["motion_gen_max_attempts"],
            timeout=self.cfg["solution"]["cuRobo"]["motion_gen_timeout"],
            enable_finetune_trajopt=False
        )

        assert self.arm_control_type == 'joint'

    """
    Solver Utils
    """

    def _get_pose_in_robot_frame(self):
        self._refresh()
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        sq, st = tf_combine(rq, rt, self._scene_base_state[..., 3:7].clone(), self._scene_base_state[..., :3].clone())
        dq, dt = tf_combine(rq, rt, self._table_base_state[..., 3:7].clone(), self._table_base_state[..., :3].clone())
        oq, ot = tf_combine(rq.unsqueeze(1).repeat(1, self.num_objs, 1),
                            rt.unsqueeze(1).repeat(1, self.num_objs, 1),
                            self.states["obj_quat"].clone(),
                            self.states["obj_pos"].clone())
        eq, et = tf_combine(rq, rt,  self.states["eef_quat"].clone(), self.states["eef_pos"].clone())

        pose = {
            'scene': {'quat': sq.to(self.tensor_args.device), 'pos': st.to(self.tensor_args.device)},
            'table': {'quat': dq.to(self.tensor_args.device), 'pos': dt.to(self.tensor_args.device)},
            'object': {'quat': oq.to(self.tensor_args.device), 'pos': ot.to(self.tensor_args.device)},
            'eef': {'quat': eq.to(self.tensor_args.device), 'pos': et.to(self.tensor_args.device)}
        }

        return pose

    def _get_batch_joint_trajs(self, trajs):
        max_len = 0

        for t in trajs:
            if t is None or len(t) == 0:
                continue
            max_len = max(max_len, t.shape[0])

        max_len = max_len + 10  # To stabilize the grasp
        padded_trajs = dict(
            position=[],
            velocity=[],
            acceleration=[],
            jerk=[],
            joint_names=self.robot_joint_names
        )

        for i, tr in enumerate(trajs):
            if tr is None:
                halt_state = self.states["q"][i:i+1][..., :-2].clone().to(self.tensor_args.device)

                new_pos = torch.ones((max_len, *halt_state.shape[1:]), device=halt_state.device, dtype=halt_state.dtype) * halt_state
                new_vel = torch.zeros((max_len, *halt_state.shape[1:]), device=halt_state.device, dtype=halt_state.dtype)
                new_acc = torch.zeros((max_len, *halt_state.shape[1:]), device=halt_state.device, dtype=halt_state.dtype)
                new_jerk = torch.zeros((max_len, *halt_state.shape[1:]), device=halt_state.device, dtype=halt_state.dtype)

                padded_trajs['position'].append(new_pos)
                padded_trajs['velocity'].append(new_vel)
                padded_trajs['acceleration'].append(new_acc)
                padded_trajs['jerk'].append(new_jerk)

            else:

                # cut-off zeros
                zero_pos = torch.sum(torch.abs(tr.position), dim=-1) < 1e-3
                zero_pos = torch.argmax(zero_pos.float(), dim=-1)

                if zero_pos.cpu().numpy() == 0:
                    last_step = tr.position.shape[0] - 1
                else:
                    last_step = zero_pos - 1

                # position
                new_pos = torch.ones((max_len, *tr.position.shape[1:]),
                                     device=tr.position.device,
                                     dtype=tr.position.dtype) * tr.position[last_step]
                new_pos[:last_step] = tr.position[:last_step].clone()

                padded_trajs['position'].append(new_pos)

                # velocity
                new_vel = torch.zeros((max_len, *tr.velocity.shape[1:]),
                                      device=tr.velocity.device,
                                      dtype=tr.velocity.dtype)
                new_vel[:last_step] = tr.velocity[:last_step].clone()

                padded_trajs['velocity'].append(new_vel)

                # acceleration
                new_acc = torch.zeros((max_len, *tr.acceleration.shape[1:]),
                                      device=tr.acceleration.device,
                                      dtype=tr.acceleration.dtype)
                new_acc[:last_step] = tr.acceleration[:last_step].clone()

                padded_trajs['acceleration'].append(new_acc)

                # jerk
                new_jerk = torch.zeros((max_len, *tr.jerk.shape[1:]),
                                       device=tr.jerk.device,
                                       dtype=tr.jerk.dtype)
                new_jerk[:last_step] = tr.jerk[:last_step].clone()

                padded_trajs['jerk'].append(new_jerk)

        padded_trajs['position'] = torch.stack(padded_trajs['position'], dim=0)
        padded_trajs['velocity'] = torch.stack(padded_trajs['velocity'], dim=0)
        padded_trajs['acceleration'] = torch.stack(padded_trajs['acceleration'], dim=0)
        padded_trajs['jerk'] = torch.stack(padded_trajs['jerk'], dim=0)

        padded_trajs = JointState(**padded_trajs)

        return padded_trajs

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)
        robot_cuRobo_cfg.cspace.velocity_scale *= self.cfg['solution']['cuRobo']['velocity_scale']
        robot_cuRobo_cfg.cspace.acceleration_scale *= self.cfg['solution']['cuRobo']['acceleration_scale']

        return robot_cuRobo_cfg

    def _get_cuRobo_world_config(self):
        pose = self._get_pose_in_robot_frame()

        oq, sq, dq = pose['object']['quat'], pose['scene']['quat'], pose['table']['quat']
        oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
        sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
        dq = torch.concat([dq[..., -1:], dq[..., :-1]], dim=-1)

        sq, st = sq.cpu().numpy(), pose['scene']['pos'].cpu().numpy()
        oq, ot = oq.cpu().numpy(), pose['object']['pos'].cpu().numpy()
        dq, dt = dq.cpu().numpy(), pose['table']['pos'].cpu().numpy()

        world_config_list = []
        for i in range(self.num_envs):

            # add scene asset
            scene_meshes = []
            for j, f in enumerate(self.scene_asset[i]["files"]):
                c_mesh = Mesh(
                        name=f"env_{i}_mesh_{j}",
                        pose=[*st[i], *sq[i]],
                        file_path=f,
                        scale=[1.0, 1.0, 1.0],
                )
                scene_meshes.append(c_mesh)

            # add table asset
            t_cube = Cuboid(
                name=f"env_{i}_table",
                pose=[*dt[i], *dq[i]],
                dims=self.table_asset[i]['dim']
            )

            # Todo: Add Combo Asset

            # add object asset
            object_meshes = []
            oq_i, ot_i = oq[i], ot[i]

            for j, obj in enumerate(self.object_asset[i]):
                q, t = oq_i[j], ot_i[j]
                o_mesh = Mesh(
                    name=f"env_{i}_obj_{j}",
                    pose=[*t, *q],
                    file_path=obj['file'],
                    scale=[1.0, 1.0, 1.0]
                )
                object_meshes.append(o_mesh)

            world_config = WorldConfig(
                mesh=scene_meshes + object_meshes,
                cuboid=[t_cube]
            )
            world_config_list.append(world_config)

        return world_config_list

    """
    CuRobo World State
    """

    def update_cuRobo_world_collider_pose(self):
        pose = self._get_pose_in_robot_frame()
        scene_pose, table_pose, object_pose = pose['scene'], pose['table'], pose['object']

        for i in range(self.num_envs):
            dq, dt = table_pose['quat'], table_pose['pos']
            dq = torch.concat([dq[..., -1:], dq[..., :-1]], dim=-1)
            pose = Pose(dt[i:i+1], dq[i:i+1])
            self.ik_collision.update_obb_pose(w_obj_pose=pose, name=f'env_{i}_table', env_idx=i)
            self.motion_generator_colliders[i].update_obb_pose(w_obj_pose=pose, name=f'env_{i}_table')

            sq, st = scene_pose['quat'], scene_pose['pos']
            sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
            pose = Pose(st[i:i+1], sq[i:i+1])

            for j, f in enumerate(self.scene_asset[i]['files']):
                self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_mesh_{j}', env_idx=i)
                self.motion_generator_colliders[i].update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_mesh_{j}')

            oq, ot = object_pose['quat'], object_pose['pos']
            oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
            for j in range(self.num_objs):
                pose = Pose(ot[i:i+1, j], oq[i:i+1, j])
                self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_obj_{j}', env_idx=i)
                self.motion_generator_colliders[i].update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_obj_{j}')

        for m in self.motion_generators:
            m.reset()

    def update_cuRobo_motion_gen_config(self, attach_goal_obj=True):
        self.update_cuRobo_world_collider_pose()

        q, qd = (self.states["q"].clone().to(self.tensor_args.device),
                 self.states["qd"].clone().to(self.tensor_args.device))

        if attach_goal_obj:
            # as robot_cfg is unique, batch_env operation is not supported for attached object

            for i in range(self.num_envs):
                cu_js = JointState(
                    position=q[i, :-2],
                    velocity=qd[i, :-2],
                    acceleration=q[i, :-2] * 0.0,
                    jerk=q[i, :-2] * 0.0,
                    joint_names=self.robot_joint_names
                )
                goal_obj_idx = self.task_obj_index[i][self.get_task_idx()].cpu().numpy()

                # setup attached_object pre_transform pose
                ee_pose_inv = self.motion_generators[i].compute_kinematics(cu_js).ee_pose.inverse()

                mesh_idx = self.motion_generator_colliders[i].get_mesh_idx(f'env_{i}_obj_{goal_obj_idx}', 0)
                curr_mesh_pose_tensor = self.motion_generator_colliders[i].get_mesh_pose_tensor()[:, mesh_idx]
                curr_mesh_pose = Pose(curr_mesh_pose_tensor[:, :3], curr_mesh_pose_tensor[:, 3:7]).inverse()
                obj_init_pose = self.motion_generator_colliders[i].world_model.get_obstacle(f'env_{i}_obj_{goal_obj_idx}').pose
                obj_init_pose = Pose.from_list(obj_init_pose, self.tensor_args)

                offset_pos = to_torch([[0.0, 0.0, self.cfg["solution"]["cuRobo"]["attach_object_z_offset"]]], device=self.tensor_args.device, dtype=torch.float)
                offset_quat = to_torch([[1.0, 0.0, 0.0, 0.0]], device=self.tensor_args.device, dtype=torch.float)
                offset_pose = Pose(offset_pos.repeat(self.num_envs, 1), offset_quat.repeat(self.num_envs, 1))

                attached_obj_pre_transform = (
                    ee_pose_inv.multiply(offset_pose).multiply(curr_mesh_pose).multiply(obj_init_pose.inverse()))

                self.motion_generators[i].attach_objects_to_robot_custom(attached_obj_pre_transform,
                                                                         [f'env_{i}_obj_{goal_obj_idx}'],
                                                                         surface_sphere_radius=self.cfg["solution"]["cuRobo"]["surface_sphere_radius"],
                                                                         sphere_fit_type=SPHERE_TYPE[self.cfg["solution"]["cuRobo"]["sphere_approx_method"]])
                self.motion_generators[i].reset()

        else:
            self._enable_goal_obj_collision_checking(True)

            for m in self.motion_generators:
                m.detach_object_from_robot()
                m.reset()

        if self.debug_viz and self.viewer:
            for i in range(self.num_envs):
                cu_js = JointState(
                    position=q[i, :-2],
                    velocity=qd[i, :-2],
                    acceleration=q[i, :-2] * 0.0,
                    jerk=q[i, :-2] * 0.0,
                    joint_names=self.robot_joint_names
                )

                eef_pose = self.motion_generators[i].compute_kinematics(cu_js).ee_pose.get_numpy_matrix()[0]
                self.cuRobo_vis_debug(self.motion_generators[i], eef_pose=eef_pose)

    """
    Sample Grasp Pose
    """

    def _sample_goal_obj_annotated_grasp_pose(self):
        pose = self._get_pose_in_robot_frame()
        oq, ot = pose['object']['quat'], pose['object']['pos']

        max_pose_seed = self.cfg["solution"]["max_grasp_pose"]

        sample_grasps = []
        for i in range(self.num_envs):
            goal_idx = self.task_obj_index[i][self.get_task_idx()]
            grasp_pose = self.obj_grasp_poses[i][goal_idx].to(self.tensor_args.device)
            random_batch = torch.randint(grasp_pose.shape[0], size=(max_pose_seed,))

            sample_quat, sample_pos = grasp_pose[random_batch][..., 3:7], grasp_pose[random_batch][..., :3]
            oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(max_pose_seed, 1),
                          ot[i:i+1, goal_idx].repeat(max_pose_seed, 1))
            gq, gt = tf_combine(oq_i, ot_i, sample_quat, sample_pos)
            gq = torch.concat([gq[..., -1:], gq[..., :-1]], dim=-1)

            sample_grasps.append(torch.concat([gt, gq], dim=-1))

        sample_grasps = torch.stack(sample_grasps, dim=0)
        return sample_grasps

    def _enable_goal_obj_collision_checking(self, enable=True):
        for i in range(self.num_envs):
            goal_idx = self.task_obj_index[i][self.get_task_idx()].cpu().numpy()
            self.ik_collision.enable_obstacle(f'env_{i}_obj_{goal_idx}', enable=enable, env_idx=i)
            self.motion_generator_colliders[i].enable_obstacle(f'env_{i}_obj_{goal_idx}', enable=enable)
            self.motion_generators[i].reset()

    def sample_goal_obj_collision_free_grasp_pose(self):
        # Use IK solver to solve for candidate grasp pose
        annotated_grasp_pose = self._sample_goal_obj_annotated_grasp_pose()

        result_holder = torch.ones((self.num_envs, 1), device=self.tensor_args.device, dtype=torch.bool)
        ik_holder = (self.robot_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).to(self.tensor_args.device)).unsqueeze(1)

        grasp_poses, pre_grasp_poses = [], []
        grasp_success, pre_grasp_success = [], []
        grasp_pose_ik = []

        # Check collision-free IK at pre-grasp pose
        for i in range(annotated_grasp_pose.shape[1]):
            grasp_candidate = annotated_grasp_pose[:, i]
            grasp_pose = Pose(grasp_candidate[..., :3], grasp_candidate[..., 3:7])
            pre_grasp_offset_pos = to_torch([0, 0, -self.cfg["solution"]["pre_grasp_offset"]],
                                            device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_pos = pre_grasp_offset_pos.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_quat = pre_grasp_offset_quat.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset = Pose(pre_grasp_offset_pos, pre_grasp_offset_quat)
            pre_grasp_pose = grasp_pose.multiply(pre_grasp_offset)

            grasp_poses.append(grasp_pose)
            pre_grasp_poses.append(pre_grasp_pose)

            ik_result = self.ik_solver.solve_batch_env(pre_grasp_pose)
            torch.cuda.synchronize()

            pre_grasp_success.append(result_holder & ik_result.success)

        # Check collision-free IK at grasp pose (disable goal obj)
        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(False)

        for i in range(annotated_grasp_pose.shape[1]):
            grasp_pose = grasp_poses[i]
            ik_result = self.ik_solver.solve_batch_env(grasp_pose)
            torch.cuda.synchronize()
            grasp_success.append(result_holder & ik_result.success)

            ik = (ik_result.solution * ik_result.success.float().unsqueeze(-1) +
                  (1. - ik_result.success.float().unsqueeze(-1)) * ik_holder[..., :-2])
            grasp_pose_ik.append(torch.concat([ik, ik_holder[..., -2:]], dim=-1))

        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(True)

        grasp_poses, pre_grasp_poses = Pose.vstack(grasp_poses, dim=1), Pose.vstack(pre_grasp_poses, dim=1)
        grasp_success, pre_grasp_success = torch.cat(grasp_success, dim=1), torch.cat(pre_grasp_success, dim=1)
        grasp_pose_ik = torch.cat(grasp_pose_ik, dim=1)

        res = {
            'grasp_poses': grasp_poses,
            'pre_grasp_poses': pre_grasp_poses,
            'grasp_success': grasp_success,
            'pre_grasp_success': pre_grasp_success,
            'grasp_ik': grasp_pose_ik
        }

        if self.debug_viz and self.viewer:
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                success_ik = torch.masked_select(grasp_pose_ik[i], mask=grasp_success[i].unsqueeze(-1))
                success_ik = success_ik.reshape(-1, grasp_pose_ik.shape[-1])

                if len(success_ik) == 0:
                    ik_poses = None
                else:
                    ik_poses = self.ik_solver.fk(success_ik).ee_pose

                self.grasp_vis_debug(pose, grasp_poses[i], pre_grasp_poses[i], ik_poses, env_idx=i)

        return res

    """
    Motion Generation
    """

    def motion_gen_to_pose_goalset(self, target_poses, offset=False):
        # motion generation to grasp the object
        self._refresh()

        trajs, poses, success, results = [], [], [], []
        for i in range(self.num_envs):
            q_start = JointState.from_position(
                self.states["q"][i:i+1, :-2].clone().to(self.tensor_args.device),
                joint_names=self.robot_joint_names
            )

            # get success mask
            if target_poses[i] is None:
                trajs.append(None)
                success.append(False)
                poses.append(None)
                results.append(None)
                continue

            assert len(target_poses[i].shape) == 3
            config = self.motion_plan_config_line.clone() if offset else self.motion_plan_config_graph.clone()
            result = self.motion_generators[i].plan_goalset(q_start, target_poses[i], config)
            traj = result.get_interpolated_plan()

            trajs.append(traj)
            poses.append(target_poses[i].clone())
            success.append(result.success.cpu().numpy()[0])
            results.append(result)

        if self.debug_viz and self.viewer:
            for i in range(self.num_envs):
                traj = trajs[i]

                if traj is None:
                    continue

                traj_state = JointState.from_position(
                    traj.position,
                    joint_names=self.robot_joint_names
                )
                traj_pose = self.ik_solver.fk(traj_state.position).ee_pose
                target_pose = poses[i][0]

                self.motion_vis_debug(self.motion_generators[i], target_pose, traj_pose)

        return self._get_batch_joint_trajs(trajs), success, poses, results

    def motion_gen_to_grasp_pose(self, poses, mask):
        target_poses = []
        for i in range(self.num_envs):
            m = mask[i].nonzero()
            if len(m) == 0:
                target_poses.append(None)
            else:
                m = m.reshape(-1)
                target_poses.append(poses[i][m].unsqueeze(dim=0).clone())

        return self.motion_gen_to_pose_goalset(target_poses)

    def motion_gen_by_z_offset(self, z, mask):
        eef = self._get_pose_in_robot_frame()['eef']
        eef_pose = Pose(eef['pos'], torch.concatenate([eef['quat'][..., -1:],  eef['quat'][..., :-1]], dim=-1))

        offset_pos = to_torch([0, 0, z], device=self.tensor_args.device, dtype=torch.float)
        offset_pos = offset_pos.unsqueeze(dim=0).repeat(self.num_envs, 1)
        offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
        offset_quat = offset_quat.unsqueeze(dim=0).repeat(self.num_envs, 1)
        grasp_offset = Pose(offset_pos, offset_quat)
        grasp_pose = eef_pose.multiply(grasp_offset)

        target_poses = []
        for i, m in enumerate(mask):
            if m:
                target_poses.append(grasp_pose[i:i+1].unsqueeze(dim=0))
            else:
                target_poses.append(None)

        return self.motion_gen_to_pose_goalset(target_poses, offset=True)

    def motion_gen_to_free_space(self, mask):

        target_pos = to_torch([[[-0.2, -0.25, 0.66], [-0.2, 0.25, 0.66]]], device=self.tensor_args.device, dtype=torch.float)
        target_quat = to_torch([[[0, 0.707, -0.707, 0], [0, 0.707, 0.707, 0]]], device=self.tensor_args.device, dtype=torch.float)
        end_pose = Pose(target_pos, target_quat)

        target_poses = []
        for i, m in enumerate(mask):
            if m:
                target_poses.append(end_pose.clone())
            else:
                target_poses.append(None)

        return self.motion_gen_to_pose_goalset(target_poses)

    """
    Arm Control
    """

    def follow_motion_trajs(self, trajs, gripper_state):
        # follow the traj
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
                rgb, seg = self.get_camera_image(rgb=True, seg=False)
                self.log_video(rgb)

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

    """
    Get Control Error
    """

    def get_end_effect_error(self, target_poses):
        # assume the target poses are in robot frame
        scene_info = self._get_pose_in_robot_frame()['eef']
        err_pose = []

        for i in range(self.num_envs):
            eq, et = scene_info['quat'][i].cpu().numpy(), scene_info['pos'][i].cpu().numpy()
            translation_matrix = tr.translation_matrix(et)
            eq = np.concatenate([eq[-1:], eq[:-1]], axis=-1)
            rotation_matrix = tr.quaternion_matrix(eq)
            eef_pose = translation_matrix @ rotation_matrix

            t_pose = target_poses[i]

            if t_pose is None:
                err_pose.append({'pos_err': 10.0, 'rot_err': 2 * np.pi})
                continue

            t_pose = t_pose.clone()[0].get_numpy_matrix()

            delta_pose = eef_pose @ tr.inverse_matrix(t_pose)
            err_pos = np.linalg.norm(delta_pose[:, :3, 3], axis=-1)
            err_rot = np.arccos((np.trace(delta_pose[:, :3, :3], axis1=1, axis2=2) - 1) / 2.)
            err_idx = np.argmin(err_pos)

            err_pose.append({'pos_err': err_pos[err_idx], 'rot_err': err_rot[err_idx]})

        return err_pose

    """
    Your Solution
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
        self.update_cuRobo_world_collider_pose()
        ik_result = self.sample_goal_obj_collision_free_grasp_pose()

        if self.cfg["solution"]["direct_grasp"]:
            ik_success = ik_result['grasp_success']
            # Move to Grasp Pose, disable goal_obj collision checking

            self.update_cuRobo_world_collider_pose()
            start_time = time.time()
            traj, success, poses, results = self.motion_gen_to_grasp_pose(ik_result['grasp_poses'], mask=ik_success)
            print("Grasp Plan", success)
            log['grasp_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
            print("Grasp Phase End")
            log['grasp_execute_error'] = self.get_end_effect_error(poses)

        else:

            ik_success = ik_result['grasp_success'] & ik_result['pre_grasp_success']
            log['ik_plan_success'] = ik_success.any(dim=-1).cpu().numpy()
            start_time = time.time()
            traj, success, poses, results = self.motion_gen_to_grasp_pose(ik_result["pre_grasp_poses"], mask=ik_success)
            print("Pre Grasp Plan", success)
            log['pre_grasp_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
            print("Pre Grasp Phase End")
            log['pre_grasp_execute_error'] = self.get_end_effect_error(poses)

            # Move to Grasp Pose, disable goal_obj collision checking
            if self.cfg["solution"]["move_offset_method"] == 'motion_planning':
                self.update_cuRobo_world_collider_pose()
                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(False)

                start_time = time.time()
                traj, success, poses, results = self.motion_gen_by_z_offset(z=self.cfg["solution"]["pre_grasp_offset"],
                                                                            mask=success)
                computing_time += time.time() - start_time
                print("Grasp Plan", success)

                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(True)
                log['grasp_plan_success'] = success
                self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement

                log['grasp_execute_error'] = self.get_end_effect_error(poses)
                print("Grasp Phase End")

            elif self.cfg["solution"]["move_offset_method"] == 'cartesian_linear':
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

        # Fetch Object out
        self.update_cuRobo_motion_gen_config(attach_goal_obj=self.cfg["solution"]["attach_goal_obj"])
        if self.cfg["solution"]["disable_grasp_obj_motion_gen"] and (not self.cfg["solution"]["attach_goal_obj"]):
            self._enable_goal_obj_collision_checking(False)

        start_time = time.time()
        traj, success, poses, results = self.motion_gen_to_free_space(mask=success)
        print("Fetch Plan", success)
        computing_time += time.time() - start_time

        self.update_cuRobo_motion_gen_config(attach_goal_obj=False)
        log['fetch_plan_success'] = success
        fetch_failure = []
        for r in results:
            if r is None:
                fetch_failure.append(r)
            else:
                fetch_failure.append(r.status)
        log['fetch_plan_failure'] = fetch_failure

        self.follow_motion_trajs(traj, gripper_state=-1)
        log['fetch_execute_error'] = self.get_end_effect_error(poses)
        print("Fetch Phase End")

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

    def grasp_vis_debug(self, poses, grasp_pose, pre_grasp_pose, ik_pose, env_idx=0):
        scene = trimesh.Scene()
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        table_pose = poses['table']
        dq = torch.concat([table_pose['quat'][..., -1:], table_pose['quat'][..., :-1]], dim=-1)
        dq, dt = dq.cpu().numpy(), table_pose['pos'].cpu().numpy()

        table_translation = tr.translation_matrix(dt[env_idx])
        table_rotation = tr.quaternion_matrix(dq[env_idx])

        table = trimesh.creation.box(extents=self.table_asset[env_idx]['dim'], transform=table_translation @ table_rotation)
        scene.add_geometry(table)

        scene_pose = poses['scene']

        sq = torch.concat([scene_pose['quat'][..., -1:], scene_pose['quat'][..., :-1]], dim=-1)
        sq, st = sq.cpu().numpy(), scene_pose['pos'].cpu().numpy()

        # vis environment 0
        scene_translation = tr.translation_matrix(st[env_idx])
        scene_rotation = tr.quaternion_matrix(sq[env_idx])

        # vis scene
        for f in self.scene_asset[env_idx]['files']:
            mesh = trimesh.load(f)
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        object_poses = poses['object']
        oq = torch.concat([object_poses['quat'][..., -1:], object_poses['quat'][..., :-1]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_poses['pos'].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
            trans = tr.translation_matrix(ot[env_idx][i])
            rot = tr.quaternion_matrix(oq[env_idx][i])
            mesh = o['mesh'].copy().apply_transform(trans @ rot)
            scene.add_geometry(mesh)

        # grasp pose
        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(grasp_pose.position.shape[0]):
            trans = tr.translation_matrix(grasp_pose.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(grasp_pose.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        # pre grasp pose
        for i in range(pre_grasp_pose.position.shape[0]):
            trans = tr.translation_matrix(pre_grasp_pose.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(pre_grasp_pose.quaternion[i].cpu().numpy())
            pre_grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([0, 0, 255]).apply_transform(pre_grasp)
            scene.add_geometry(command_marker)

        if ik_pose is not None:
            for i in range(ik_pose.position.shape[0]):
                trans = tr.translation_matrix(ik_pose.position[i].cpu().numpy())
                rot = tr.quaternion_matrix(ik_pose.quaternion[i].cpu().numpy())
                ik_grasp = trans @ rot @ vis_rot
                command_marker = create_gripper_marker([0, 255, 0]).apply_transform(ik_grasp)
                scene.add_geometry(command_marker)

        scene.show()

    def cuRobo_vis_debug(self, motion_gen, eef_pose=None):
        scene = trimesh.Scene()

        model = motion_gen.world_model
        collider = motion_gen.world_coll_checker
        robot = motion_gen.kinematics.kinematics_config

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        for i, m in enumerate(model.mesh):
            mesh = trimesh.load(m.file_path)
            pose_tensor = collider.get_mesh_pose_tensor()
            pose_pos, pose_quat = pose_tensor[:, i, :3], pose_tensor[:, i, 3:7]

            pose = Pose(pose_pos.clone(), pose_quat.clone()).inverse().get_numpy_matrix()[0]
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)

        if eef_pose is not None:
            sphere_links = robot.link_spheres.cpu().numpy()
            attach_idx = robot.link_name_to_idx_map['attached_object']
            idx_map = robot.link_sphere_idx_map.cpu().numpy()
            spheres = sphere_links[np.where(idx_map == attach_idx)]

            for s in spheres:
                r = s[-1] # radius
                if r > 0:
                    sphere = trimesh.creation.icosphere(radius=r)
                    translation = tr.translation_matrix(s[:3])
                    sphere.apply_transform(translation)
                    sphere.apply_transform(eef_pose)
                    sphere.visual.face_colors = [0, 250, 0, 200]
                    scene.add_geometry(sphere)

        scene.show()

    def motion_vis_debug(self, motion_gen, target_poses, traj_poses):
        scene = trimesh.Scene()

        model = motion_gen.world_model
        collider = motion_gen.world_coll_checker

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        for i, m in enumerate(model.mesh):
            mesh = trimesh.load(m.file_path)
            pose_tensor = collider.get_mesh_pose_tensor()
            pose_pos, pose_quat = pose_tensor[:, i, :3], pose_tensor[:, i, 3:7]

            pose = Pose(pose_pos.clone(), pose_quat.clone()).inverse().get_numpy_matrix()[0]
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)

        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(target_poses.position.shape[0]):
            trans = tr.translation_matrix(target_poses.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(target_poses.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        for i in range(traj_poses.position.shape[0]):
            trans = tr.translation_matrix(traj_poses.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(traj_poses.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([0, 255, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        scene.show()


"""
Util Functions
"""


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def plot_trajs(trajs, dt):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(2, 1)
    for k, traj in enumerate(trajs):
        q = traj[0]
        qd = traj[1]

        if k == 0:
            linestyle = '-'
            timesteps = [i * dt for i in range(q.shape[0])]
            for i in range(q.shape[-1]):

                axs[0].plot(timesteps, q[:, i], label=str(i), linestyle=linestyle)
                axs[1].plot(timesteps, qd[:, i], label=str(i), linestyle=linestyle)

        else:
            linestyle = '--'
            timesteps = [i * dt for i in range(q.shape[0])]
            for i in range(q.shape[-1] - 2):
                axs[0].plot(timesteps, q[:, i], label=str(i), linestyle=linestyle)
                axs[1].plot(timesteps, qd[:, i], label=str(i), linestyle=linestyle)

            for i in range(q.shape[-1] - 2, q.shape[-1]):
                axs[0].plot(timesteps, q[:, i] * 50, label=str(i), linestyle=linestyle)
                axs[1].plot(timesteps, qd[:, i] * 50, label=str(i), linestyle=linestyle)

    plt.legend()
    plt.show()
