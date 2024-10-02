import logging
from timeit import default_timer as timer

import numpy as np
import scipy.spatial
import trimesh.transformations
import trimesh.transformations as tra
from autolab_core import Logger

import sys
sys.path.append('../third_party/SceneCollisionNet/scenecollisionnet')
sys.path.append('../third_party/cabinet/src')
from policy.collision_checker_custom import (
    FCLMultiSceneCollisionChecker,
    FCLSceneCollisionChecker,
    FCLSelfCollisionChecker,
    NNSceneCollisionChecker,
    NNSelfCollisionChecker,
    CabinetSceneCollisionChecker
)
from cabi_net.model.waypoint_custom import load_cabinet_model_for_inference
from policy.robot import Robot
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
import torch
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

np.set_printoptions(suppress=True)


class CuroboIK(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tensor_args = TensorDeviceType()
        ik_config = IKSolverConfig.load_from_robot_config(
            self._get_cuRobo_robot_config(),
            rotation_threshold=self.cfg["ik_rot_th"],
            position_threshold=self.cfg["ik_pos_th"],
            num_seeds=self.cfg["ik_num_seed"],
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=self.cfg["collision_activation_dist"]
        )
        self.ik_solver = IKSolver(ik_config)

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)

        return robot_cuRobo_cfg

    def solve_ik_curobo(self, pose):
        quat = tra.quaternion_from_matrix(pose[:3, :3])
        quat = torch.tensor(quat, device=self.tensor_args.device, dtype=torch.float)
        pos = torch.tensor(pose[:3, 3], device=self.tensor_args.device, dtype=torch.float)

        ik_pose = Pose(pos, quat)
        ik_result = self.ik_solver.solve(ik_pose)

        if bool(ik_result.success.reshape(-1).cpu().numpy()):
            return ik_result.solution[:7].cpu().numpy().reshape(-1)

        return None


class MPPIPolicy:
    def __init__(
        self,
        num_rollouts,
        horizon,
        q_noise_std,
        num_iks: 10,
        max_step=0.02,
        collision_steps=10,
        transition_threshold=0.05,
        self_coll_nn=None,
        scene_coll_nn=None,
        device=0,
        log_file=None,
        ik_kwargs={},
        **kwargs
    ):
        """
        Args:
          mode: string, 'qspace' or 'tspace' for configuration space and task space.
          num_rollouts: number of desired paths.
          horizon: int, horizon of unrolling
          q_noise_std: standard deviation of noise in qspace.
          max_step: float, distance to move along trajectory at each timestep in rollout
          lift_height: height in m to lift after grasping, placing
          lift_steps: int, number of steps in lifting plan
          collision_steps: int, number of discrete intervals to check collisions for between each timestep
          transition_threshold: float, threshold for moving between states
          self_coll_nn: path to a network for checking self collisions, defaults to FCL checker
          scene_coll_nn: path to scene collision network, defaults to FCL checker
          cam_type: "ws" or "hand" determines which camera to use
          device: int, compute device for rollouts
          safe: bool, pause before grasping and placing and wait for confirmation
          log_file: str, path to a logging file
        """
        self.n_iks = num_iks
        self.num_path = num_rollouts
        self.horizon = horizon
        self.noise_std = q_noise_std
        self.max_step = max_step
        self.collision_steps = collision_steps
        self.transition_threshold = transition_threshold
        self.device = device

        self.ik_proc = CuroboIK(ik_kwargs)

        CODE_PATH = '../third_party'
        self.robot = Robot(
            f"{CODE_PATH}/SceneCollisionNet/data/panda/panda.urdf",
            "right_gripper",  # same as panda_hand
            device=device,
        )

        # Set up self collision checkers
        if self_coll_nn == 'FCL':
            self.self_collision_checker = FCLSelfCollisionChecker(self.robot)
        elif self_coll_nn == 'SCN':
            self.self_collision_checker = NNSelfCollisionChecker(f'{CODE_PATH}/SceneCollisionNet/weights/self_coll_nn', device=device)
        else:
            raise NotImplementedError

        for i in range(len(self.robot.links) - 1):
            self.self_collision_checker.set_allowed_collisions(
                self.robot.links[i].name, self.robot.links[i + 1].name
            )
        self.self_collision_checker.set_allowed_collisions("panda_hand", "panda_rightfinger")
        self.self_collision_checker.set_allowed_collisions( "panda_link7", "panda_hand")

        if scene_coll_nn == 'FCL':
            self.scene_collision_checker = FCLMultiSceneCollisionChecker(self.robot, use_scene_pc=True)
            self.robot_to_model = [0.0, 0.0, 0.0]
        elif scene_coll_nn == 'SCN':
            self.scene_collision_checker = NNSceneCollisionChecker(f'{CODE_PATH}/SceneCollisionNet/weights/scene_coll_nn', self.robot, device=device, use_knn=False)
            self.robot_to_model = [-0.5, 0.0, 0.2]
        elif scene_coll_nn == 'CBN':
            self.scene_collision_checker = CabinetSceneCollisionChecker(f'{CODE_PATH}/cabinet/checkpoints/cabinet_collision', self.robot, device=device)
            self.robot_to_model = [-0.5, 0.0, -0.1]
            self.waypoint_pred, _ = load_cabinet_model_for_inference(f'{CODE_PATH}/cabinet/checkpoints/cabinet_waypoint/weights/last.ckpt',
                                                                     f'{CODE_PATH}/cabinet/checkpoints/cabinet_waypoint/inference.yaml')
        else:
            raise NotImplementedError

        self.logger = Logger.get_logger("MPPIPolicy", log_file=log_file, log_level=logging.DEBUG)

        self.state = 0
        self.poses = None
        self.cur_grasps_init = None
        self.cur_grasps_final = None
        self.ik_cur_grasps_init = None
        self.ik_cur_grasps_final = None
        self.reach_plan = None
        self.prev_plan_ind = None
        self.ee_offset = None
        self.prev_rollout_lens = []
        self.reset()

    def reset(self, state=0):
        self.state = state
        self.poses = None
        self.cur_grasps_init = None
        self.cur_grasps_final = None
        self.ik_cur_grasps_init = None
        self.ik_cur_grasps_final = None
        self.reach_plan = None
        self.prev_plan_ind = None
        self.ee_offset = None
        self.prev_rollout_lens = []

    def set_state(self, state=0):
        self.state = state

    """
    Update Planning Context
    """
    def update_context(self, obs):
        """
        syncs up the scene with the latest observation.
        Args:
        obs: np.array, pointcloud of the scene
        state: dict, this is the gym_state_dict coming from scene managers.
          contains info about robot and object."""
        self.robot_q = obs["q"].astype(np.float64).copy()

        rtm = np.eye(4)
        rtm[:3, 3] = self.robot_to_model

        in_obs = {
            "scene_pc": obs["scene_ptc"],
            "object_pc": obs["obj_ptc"],
            "robot_to_model": rtm,
            "model_to_robot": np.linalg.inv(rtm),
        }

        self.robot.set_joint_cfg(self.robot_q)
        ee_pose = self.robot.ee_pose[0].cpu().numpy()

        if self.ik_cur_grasps_init is None:
            if len(obs["obj_ptc"]) > 0:
                obj_pose = trimesh.transformations.translation_matrix(obs["obj_ptc"].mean(axis=0))
                offset_pose = trimesh.transformations.inverse_matrix(ee_pose) @ obj_pose
                self.ee_offset = offset_pose[:3, 3]
            else:
                self.ee_offset = np.array([0, 0, 0])

            with torch.no_grad():
                self.scene_collision_checker.set_object(in_obs)

        assert len(obs["scene_ptc"]) > 0
        with torch.no_grad():
            self.scene_collision_checker.set_scene(in_obs)

    def update_model_center(self, robot_to_model):
        self.robot_to_model = robot_to_model

    def get_waypoint_pose(self, eef_pos, eef_quat, ptc):
        eef_quat = np.concatenate([eef_quat.cpu().numpy()[-1:], eef_quat.cpu().numpy()[:-1]], axis=-1)

        waypoints, _ = self.waypoint_pred.run_inference(
            ptc.cpu().numpy(),
            tra.translation_matrix(self.robot_to_model),
            tra.translation_matrix(eef_pos.cpu().numpy())
        )

        if len(waypoints) == 0:
            return None

        waypoint_poses = np.expand_dims(np.eye(4), axis=0).repeat(len(waypoints), axis=0)
        waypoint_poses[:, :3, 3] = waypoints
        waypoint_poses = waypoint_poses @ tra.quaternion_matrix(eef_quat)

        return waypoint_poses

    def update_goal(self, init_poses, final_poses):
        self.poses = {
            'init_poses': init_poses,
            'final_poses': final_poses
        }

    """
    Planning Utils
    """

    def _compute_batch_reward(self, qs):
        """
        Args:
          paths: tensor, shape (num_paths, horizon, dof)

        Returns:
          closest distance, tensor float, (batch_size)
          closest_grasp_index min index
        """
        if qs.shape[-1] != self.ik_cur_grasps_init.shape[-1]:
            raise ValueError(
                "last dim should be equal {} {}".format(
                    qs.shape, self.ik_cur_grasps_init.shape
                )
            )
        distance = torch.norm(
            qs[..., None, :7]
            - torch.from_numpy(self.ik_cur_grasps_init[:, :7]).to(self.device),
            dim=-1,
        )
        output = torch.min(distance, dim=-1)
        return -output[0], output[1]

    def _trim_rollouts(self, rewards, collisions):
        # Array of connected points collision free along path
        connected = torch.cat(
            (
                torch.ones(
                    self.num_path, 1, dtype=torch.bool, device=self.device
                ),
                collisions.sum(dim=-1).cumsum(dim=-1) == 0,
            ),
            dim=1,
        )
        rewards = rewards * connected - 10000 * ~connected
        rollout_values, rollout_lengths = rewards.max(dim=1)
        return rollout_values, rollout_lengths

    def _collect_rollouts(self):
        """
        Unrolls self.horizon steps and push the trajectory value + trajectory
        in output_queue.

        Rollouts are list of configurations or (configuration, bool) where the
        bool variable indicates whether you want the controller to accurately get
        to that way point. Otherwise the contoller will proceed to next waypoints
        if it is roughly close.

        For context, for one environment you can provide a list of waypoints and
        the controller only execs the first set of actions that can be done in
        1/--control-frequency.
        """
        init_q = self.robot_q.copy()

        # Setup rollouts array and initialize with first q
        rewards = torch.empty((self.num_path, self.horizon), device=self.device)
        rollouts = torch.empty((self.num_path, self.horizon, self.robot.dof), device=self.device)
        rollouts[:, 0] = (
            torch.from_numpy(init_q)
            .to(self.device)
            .reshape([1, -1])
            .repeat([self.num_path, 1])
        )

        # Find straight line trajectory to goal
        _, closest_g_ind = self._compute_batch_reward(rollouts[:, 0])
        closest_g_q = torch.from_numpy(self.ik_cur_grasps_init).to(self.device)[closest_g_ind]
        greedy_dir = torch.nn.functional.normalize(closest_g_q - rollouts[:, 0], dim=-1)

        # Perturb the greedy direction and renormalize (keep one greedy)
        noise_dir = torch.empty((self.num_path, self.robot.dof), device=self.device).normal_(mean=0, std=self.noise_std)
        noise_dir[0] = 0.0
        rollout_dirs = torch.nn.functional.normalize(greedy_dir + noise_dir)

        # Generate rollouts
        step_sizes = torch.empty((self.num_path, self.horizon - 1, 1), device=self.device).fill_(self.max_step)
        rollouts[:, 1:] = rollout_dirs[:, None] * step_sizes
        rollouts = torch.cumsum(rollouts.cpu(), dim=1).to(self.device)  # solve deterministic err

        # Clip actions to joint limits
        rollouts = torch.max(rollouts, self.robot.min_joints)
        rollouts = torch.min(rollouts, self.robot.max_joints)

        # Set fingers to open or closed depending on desired cfg
        rollouts[..., -2:] = closest_g_q[0, -2:]
        rewards = self._compute_batch_reward(rollouts)[0]
        return rollouts, rewards

    def _check_collisions(self, rollouts, check_obj=False):
        alpha = (
            torch.linspace(0, 1, self.collision_steps)
            .reshape([1, 1, -1, 1])
            .to(self.device)
        )
        waypoints = (
            alpha * rollouts[:, 1:, None]
            + (1.0 - alpha) * rollouts[:, :-1, None]
        ).reshape(-1, self.robot.dof)

        if isinstance(self.self_collision_checker, FCLSelfCollisionChecker):
            coll_mask = np.zeros(len(waypoints), dtype=np.bool)
            for i, q in enumerate(waypoints):
                coll_mask[i] = self.self_collision_checker(q)
            coll_mask = torch.from_numpy(coll_mask).to(self.device)
        else:
            coll_mask = self.self_collision_checker(waypoints)

        if isinstance(self.scene_collision_checker, FCLSceneCollisionChecker):
            scene_coll_mask = self.scene_collision_checker(waypoints)
            coll_mask |= torch.from_numpy(scene_coll_mask).to(self.device)
        else:
            coll_mask |= self.scene_collision_checker(waypoints, threshold=0.45)

        if check_obj:
            translation_matrix = trimesh.transformations.translation_matrix(self.ee_offset)
            obj_pose = self.robot.ee_pose @ torch.from_numpy(translation_matrix).float().to(self.device)
            obj_trs = torch.cat([obj_pose[:, :3, 3], torch.ones(len(self.robot.ee_pose), 1, device=self.device)], dim=-1)

            model_obj_trs = (
                self.scene_collision_checker.robot_to_model @ obj_trs.T
            )
            obj_coll = self.scene_collision_checker.check_object_collisions(
                model_obj_trs[:3].T, threshold=0.45
            )
            coll_mask |= obj_coll.reshape(coll_mask.shape)

        return coll_mask.reshape(
            self.num_path, self.horizon - 1, self.collision_steps
        )

    def _localize_in_plan(self, cur_q, plan):
        distances = np.linalg.norm(plan - cur_q, axis=1)
        return np.argmin(distances)

    def _set_pose_ik(self, two_stage=True, close_config=True, n_iks=None):
        cur_grasps_init = self.poses['init_poses'].copy()
        cur_grasps_final = self.poses['final_poses'].copy()

        num_grasps = len(cur_grasps_init)
        n_iks = n_iks if n_iks is not None else self.n_iks

        # collect computed iks
        ik_time = timer()
        cur_grasp_init_inds = []
        ik_cur_grasps_init = []
        for i in range(num_grasps):
            for n in range(n_iks):
                output = self.ik_proc.solve_ik_curobo(cur_grasps_init[i])
                if output is not None:
                    cur_grasp_init_inds.append(i)
                    ik_cur_grasps_init.append(output)

        cur_grasps_init = cur_grasps_init[cur_grasp_init_inds]
        cur_grasps_final = cur_grasps_final[cur_grasp_init_inds]

        if len(cur_grasps_init) == 0:
            self.logger.warning("No pre-grasp IKs found!")
            self.reset()
            return None

        # collect computed iks
        if two_stage:
            cur_grasp_final_inds = []
            ik_cur_grasps_final = []
            for i in range(len(cur_grasps_final)):
                output = self.ik_proc.solve_ik_curobo(cur_grasps_final[i])
                if output is not None:
                    cur_grasp_final_inds.append(i)
                    ik_cur_grasps_final.append(output)

            cur_grasps_final = cur_grasps_final[cur_grasp_final_inds]
            cur_grasps_init = cur_grasps_init[cur_grasp_final_inds]
            ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)[cur_grasp_final_inds]
            ik_cur_grasps_final = np.asarray(ik_cur_grasps_final)
        else:
            ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)
            ik_cur_grasps_final = ik_cur_grasps_init.copy()

        if len(cur_grasps_final) == 0:
            self.logger.warning("No grasp IKs found!")
            self.reset()
            return None

        cfree_cur_grasps_init = np.asarray(cur_grasps_init)
        cfree_ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)
        cfree_cur_grasps_final = np.asarray(cur_grasps_final)
        cfree_ik_cur_grasps_final = np.asarray(ik_cur_grasps_final)

        if close_config:
            init_final_mask = (
                    np.linalg.norm(
                        ik_cur_grasps_init - np.asarray(ik_cur_grasps_final),
                        axis=-1,
                        ord=np.inf,
                    )
                    < 0.25
            )

            self.logger.debug(
                "{}/{} grasps have close ik solution".format(
                    init_final_mask.sum(), num_grasps
                )
            )

            # Check collisions for each grasp with self and with scene
            cfree_cur_grasps_init = np.asarray(cur_grasps_init)[init_final_mask]
            cfree_ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)[init_final_mask]
            cfree_cur_grasps_final = np.asarray(cur_grasps_final)[init_final_mask]
            cfree_ik_cur_grasps_final = np.asarray(ik_cur_grasps_final)[init_final_mask]

        # import pdb; pdb.set_trace()
        self.logger.debug("ik_time = {}".format(timer() - ik_time))
        self.logger.debug(
            "{}/{} grasps have ik solution".format(
                len(cur_grasps_final), num_grasps
            )
        )

        if len(cfree_cur_grasps_init) == 0:
            self.logger.warning("No grasp IKs found!")
            self.reset()
            return None

        # add additional 2 joints
        gripper_joint = np.ones((len(cfree_ik_cur_grasps_init), 2), dtype=np.float32) * 0.04
        cfree_ik_cur_grasps_init = np.concatenate([cfree_ik_cur_grasps_init, gripper_joint], axis=-1)
        cfree_ik_cur_grasps_final = np.concatenate([cfree_ik_cur_grasps_final, gripper_joint], axis=-1)

        if isinstance(
                self.self_collision_checker, FCLSelfCollisionChecker
        ):
            self_coll_mask = np.zeros(
                len(cfree_cur_grasps_init), dtype=np.bool
            )
            for i in range(len(cfree_ik_cur_grasps_init)):
                self_coll_mask[i] = not self.self_collision_checker(
                    cfree_ik_cur_grasps_init[i]
                )
        else:
            self_coll_mask = (
                ~self.self_collision_checker(cfree_ik_cur_grasps_init)
                .cpu()
                .numpy()
            )
        cfree_cur_grasps_init = cfree_cur_grasps_init[self_coll_mask]
        cfree_cur_grasps_final = cfree_cur_grasps_final[self_coll_mask]
        cfree_ik_cur_grasps_init = cfree_ik_cur_grasps_init[self_coll_mask]
        cfree_ik_cur_grasps_final = cfree_ik_cur_grasps_final[self_coll_mask]

        if len(cfree_cur_grasps_init) == 0:
            self.logger.warning("No grasps found!")
            self.reset()
            return None

        if isinstance(self.scene_collision_checker, FCLSceneCollisionChecker):
            scene_coll_mask = ~self.scene_collision_checker(cfree_ik_cur_grasps_init, threshold=0.45)
        else:
            scene_coll_mask = ~self.scene_collision_checker(cfree_ik_cur_grasps_init, threshold=0.45).cpu().numpy()

        cfree_cur_grasps_init = cfree_cur_grasps_init[scene_coll_mask]
        cfree_cur_grasps_final = cfree_cur_grasps_final[scene_coll_mask]
        cfree_ik_cur_grasps_init = cfree_ik_cur_grasps_init[scene_coll_mask]
        cfree_ik_cur_grasps_final = cfree_ik_cur_grasps_final[scene_coll_mask]

        if len(cur_grasps_init) == 0 or len(cfree_cur_grasps_init) == 0:
            self.logger.warning("No grasps found!")
            self.reset()
            return None

        self.cur_grasps_init = cfree_cur_grasps_init
        self.cur_grasps_final = cfree_cur_grasps_final
        self.ik_cur_grasps_init = cfree_ik_cur_grasps_init
        self.ik_cur_grasps_final = cfree_ik_cur_grasps_final
        self.ik_cur_grasps_init[:, -2:] = 0.04  # Set fingers open
        self.ik_cur_grasps_final[:, -2:] = 0.04

        return len(self.ik_cur_grasps_init)

    """
    Plan
    """

    def get_plan(self):

        # Set IK Goals and reach Target Init Poses
        if self.state == 0:
            if self.ik_cur_grasps_init is None or self.ik_cur_grasps_final is None:
                res = self._set_pose_ik()
                if res is None:
                    return None

            if np.linalg.norm(self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1).min() < self.transition_threshold:
                self.state = 1
                return [(self.robot_q.copy(), True)]

        # Reach toward the final pose with linear motion
        elif self.state == 1:
            closest_init_grasp_ind = np.linalg.norm(self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1).argmin()

            init_q = self.robot_q.copy()
            final_q = self.ik_cur_grasps_final[closest_init_grasp_ind]
            alpha = np.linspace(0, 1, 20)[:, None]
            if self.reach_plan is None:
                self.reach_plan = alpha * final_q[None, :] + (1 - alpha) * init_q[None, :]
            plan_ind = self._localize_in_plan(self.robot_q, self.reach_plan)
            index = min(plan_ind + 1, len(self.reach_plan) - 1)
            self.logger.debug(
                "index = {}, reach_plan = {}".format(
                    index, len(self.reach_plan)
                )
            )

            if index == len(self.reach_plan) - 1 or (
                self.prev_plan_ind is not None and self.prev_plan_ind >= index
            ):
                self.state = 2
                return None
            else:
                self.prev_plan_ind = index

            # precision reaching for last 10% of plan
            return list(
                zip(
                    self.reach_plan[index:],
                    [True if index > 5 else False]
                    * len(self.reach_plan[index:]),
                )
            )

        elif self.state == 2:
            if self.ik_cur_grasps_init is None or self.ik_cur_grasps_final is None:
                res = self._set_pose_ik(two_stage=True, close_config=False, n_iks=5)
                if res is None:
                    return None

            if np.linalg.norm(self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1).min() < self.transition_threshold:
                self.ik_cur_grasps_init = self.ik_cur_grasps_final.copy()
                self.cur_grasps_init = self.cur_grasps_final.copy()
                self.state = 3
                return [(self.robot_q.copy(), True)]

        # Plan fetch to final pose
        elif self.state == 3:
            if self.ik_cur_grasps_init is None or self.ik_cur_grasps_final is None:
                res = self._set_pose_ik(two_stage=False)
                if res is None:
                    return None

            if np.linalg.norm(self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1).min() < self.transition_threshold:
                return None
        else:
            raise NotImplementedError

        self.logger.debug(
            "Joint Dist to Target: {:.4f}".format(
                np.linalg.norm(
                    self.robot_q[:7] - self.ik_cur_grasps_init[:, :7],
                    ord=np.inf,
                    axis=-1,
                ).min()
            )
        )

        # Collect rollouts and find best collision free trajectory
        self.logger.info("collecting data")
        rollouts, rewards = self._collect_rollouts()
        collisions = self._check_collisions(rollouts, check_obj=(self.state == 2))
        rollout_values, rollout_lengths = self._trim_rollouts(rewards, collisions)

        best_rollout_val, best_rollout_ind = rollout_values.max(dim=0)
        self.logger.debug(f"==============> best_rollout_val {best_rollout_val} <================")
        best_rollout_len = rollout_lengths[best_rollout_ind]
        self.prev_rollout_lens.append(best_rollout_len)
        if len(self.prev_rollout_lens) >= 5:
            self.prev_rollout_lens.pop(0)
            # mppi get stuck
            prev_len = sum(self.prev_rollout_lens)
            if self.state == 0 and prev_len <= 3:
                self.state = 1
            elif self.state == 2 and prev_len <= 5:
                self.ik_cur_grasps_init = self.ik_cur_grasps_final.copy()
                self.cur_grasps_init = self.cur_grasps_final.copy()
                self.state = 3
            elif prev_len == 0:
                return None

        best_rollout = rollouts[best_rollout_ind, : best_rollout_len + 1].cpu().numpy()

        self.logger.info("reward = {:.4f}".format(best_rollout_val))
        self.logger.info("best rollout length = {:d}".format(best_rollout_len))
        self.rollouts = (rollouts, rollout_values, rollout_lengths)

        return best_rollout
