
import numpy as np
import torch
import copy
import time
import os
import trimesh.transformations as tr

# cuRobo
from curobo.geom.types import WorldConfig, Cuboid, Mesh
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

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video, create_gripper_marker, plot_trajs
from isaacgymenvs.tasks.fetch.fetch_solution_base import FetchSolutionBase
from isaacgymenvs.tasks.fetch.utils.mppi_utils import MPPIPolicy


class FetchPtdCabinet(FetchPointCloudBase, FetchSolutionBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        self.tensor_args = TensorDeviceType()

        # GT IK solver for grasp filtering
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

        self.mppi_log_dir = f'./logs/mppi_log/{self.cfg["experiment_name"]}'
        if not os.path.exists(self.mppi_log_dir):
            os.makedirs(self.mppi_log_dir)

        self.mppi_policy = MPPIPolicy(device=self.device, ik_kwargs=self.cfg["solution"]["cuRobo"],
                                      log_file=f'{self.mppi_log_dir}/mppi_log', **self.cfg["solution"]["mppi"])
        self.current_plan = []

        assert self.arm_control_type == 'joint'
        assert self.num_envs == 1

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

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)

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
    Motion Gen & IK Config
    """

    def update_ik_world_collider_pose(self):
        pose = self._get_pose_in_robot_frame()
        scene_pose, table_pose, object_pose = pose['scene'], pose['table'], pose['object']

        for i in range(self.num_envs):
            dq, dt = table_pose['quat'], table_pose['pos']
            dq = torch.concat([dq[..., -1:], dq[..., :-1]], dim=-1)
            pose = Pose(dt[i:i+1], dq[i:i+1])
            self.ik_collision.update_obb_pose(w_obj_pose=pose, name=f'env_{i}_table', env_idx=i)

            sq, st = scene_pose['quat'], scene_pose['pos']
            sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
            pose = Pose(st[i:i+1], sq[i:i+1])

            for j, f in enumerate(self.scene_asset[i]['files']):
                self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_mesh_{j}', env_idx=i)

            oq, ot = object_pose['quat'], object_pose['pos']
            oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
            for j in range(self.num_objs):
                pose = Pose(ot[i:i+1, j], oq[i:i+1, j])
                self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'env_{i}_obj_{j}', env_idx=i)

    """
    Sample Grasps
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

    def sample_goal_obj_collision_free_grasp_pose(self):
        # Use IK solver to solve for candidate grasp pose
        annotated_grasp_pose = self._sample_goal_obj_annotated_grasp_pose()

        result_holder = torch.ones((self.num_envs, 1), device=self.tensor_args.device, dtype=torch.bool)

        grasp_poses, pre_grasp_poses = [], []
        grasp_success = []

        # Check collision-free IK at grasp pose (disable goal obj)
        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(False)

        # Check collision-free IK at pre-grasp pose
        for i in range(50):
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

            ik_result = self.ik_solver.solve_batch_env(grasp_pose)
            torch.cuda.synchronize()

            grasp_success.append(result_holder & ik_result.success)

        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(True)

        grasp_poses, pre_grasp_poses = Pose.vstack(grasp_poses, dim=1), Pose.vstack(pre_grasp_poses, dim=1)
        grasp_success = torch.cat(grasp_success, dim=1)

        res = {
            'grasp_poses': grasp_poses,
            'pre_grasp_poses': pre_grasp_poses,
            'grasp_success': grasp_success
        }

        return res

    """
    Obs Utils
    """
    def update_mppi_state(self, add_goal_obj=False):
        ptc = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=True, segmented_ptd=True)['camera_pointcloud_seg'][0]
        for k, v in ptc.items():
            ptc[k] = v.cpu().numpy()

        if add_goal_obj:
            scene_ptc = ptc['scene']
            obj_ptc = ptc['goal']
        else:
            scene_ptc = np.concatenate([ptc['scene'], ptc['goal']], axis=0)
            obj_ptc = []

        self.mppi_policy.update_context({
            "q": self.states["q"].cpu().numpy()[0],
            "scene_ptc": scene_ptc,
            "obj_ptc": obj_ptc
        })

    def update_mppi_model_center(self):
        ptc = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=True, segmented_ptd=True)['camera_pointcloud_seg'][0]
        goal_ptc = ptc['goal'].cpu().numpy()
        if len(goal_ptc) > 0:
            goal_h = goal_ptc.mean(axis=0)[-1]
            if self.cfg["solution"]["mppi"]["scene_coll_nn"] == 'CBN':
                offset = 0.2
            elif self.cfg["solution"]["mppi"]["scene_coll_nn"] == 'SCN':
                offset = 0.35
            else:
                offset = 0.0
            robot_to_model = [-0.5, 0.0, - goal_h + offset]
        else:
            robot_to_model = self.mppi_policy.robot_to_model

        self.mppi_policy.update_model_center(robot_to_model)

    def get_scene_waypoints(self):
        pose = self._get_pose_in_robot_frame()
        ptc = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=True, segmented_ptd=True)['camera_pointcloud_seg'][0]['scene']
        return self.mppi_policy.get_waypoint_pose(pose['eef']['pos'][0], pose['eef']['quat'][0], ptc)

    """
    Arm Control
    """

    def _set_plan(self, plan):
        self.current_plan = [copy.deepcopy(q) for q in plan] if plan is not None else None
        self.current_plan_index = 0
        self.num_steps_for_q = 0

    def _get_next_q_in_plan(self, cur_q, max_steps_per_q=40):
        if self.current_plan is None:
            return None
        if self.current_plan_index >= len(self.current_plan):
            return None
        self.num_steps_for_q += 1

        while True:
            if isinstance(self.current_plan[self.current_plan_index], tuple):
                current_action = self.current_plan[self.current_plan_index][0]
                super_accurate = self.current_plan[self.current_plan_index][1]
            else:
                current_action = self.current_plan[self.current_plan_index]
                super_accurate = (
                    self.current_plan_index == len(self.current_plan) - 1
                )

            delta_q = cur_q[:7] - current_action[:7]
            delta_q = np.max(np.abs(delta_q))

            if super_accurate:
                move_forward = delta_q < 0.001
            else:
                move_forward = delta_q < 0.01

            if not move_forward and not super_accurate:
                move_forward = self.num_steps_for_q > max_steps_per_q

            if move_forward:
                self.current_plan_index += 1
                self.num_steps_for_q = 0
                if self.current_plan_index == len(self.current_plan):
                    self.current_plan_index -= 1
                    break
            else:
                break

        return current_action

    def step_q(self, q, gripper_state):
        # follow the traj
        if q is None:
            traj_command = {"joint_state": self.states['q'][:, :-2].clone()}
        else:
            traj_command = {"joint_state": torch.from_numpy(q).to(self.device).unsqueeze(0)[:, :-2]}

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

            delta_pose = t_pose @ tr.inverse_matrix(eef_pose)
            err_pos = np.linalg.norm(delta_pose[:, :3, 3], axis=-1)
            err_rot = np.arccos((np.trace(delta_pose[:, :3, :3], axis1=1, axis2=2) - 1) / 2.)
            err_idx = np.argmin(err_pos)

            err_pose.append({'pos_err': err_pos[err_idx], 'rot_err': err_rot[err_idx]})

        return err_pose

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

        self.update_ik_world_collider_pose()
        ik_result = self.sample_goal_obj_collision_free_grasp_pose()

        pre_grasp_poses = ik_result['pre_grasp_poses'][0].get_matrix()[ik_result['grasp_success'][0]]
        grasp_poses = ik_result['grasp_poses'][0].get_matrix()[ik_result['grasp_success'][0]]

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

        self.current_plan = None
        free_space_pose = np.array([[[0.,   -1.,    0.,   -0.2], [-1.,    0.,   0.,   -0.25],
                                     [0.,    0.,   -1.,    0.66], [0.,    0.,    0.,    1.]],
                                    [[0.,    1.,    0.,   -0.2], [1.,    0.,    0.,    0.25],
                                     [0.,    0.,   -1.,    0.66], [0.,    0.,    0.,    1.]]])

        if hasattr(self.mppi_policy, 'waypoint_pred') and self.cfg["solution"]["mppi"]["sample_waypoints"]:
            waypoint_pose = self.get_scene_waypoints()  # a maximal of 10 x 2 paris

            if waypoint_pose is not None:
                self.mppi_policy.reset(state=2)
                waypoint_pose = waypoint_pose[:10]
                r_waypoint_pose = waypoint_pose.repeat(len(free_space_pose), axis=0)
                r_free_space_pose = free_space_pose.repeat(len(waypoint_pose), axis=0)
                self.mppi_policy.update_goal(r_waypoint_pose, r_free_space_pose)
            else:
                self.mppi_policy.reset(state=3)
                self.mppi_policy.update_goal(free_space_pose, free_space_pose)
        else:
            self.mppi_policy.reset(state=3)
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
