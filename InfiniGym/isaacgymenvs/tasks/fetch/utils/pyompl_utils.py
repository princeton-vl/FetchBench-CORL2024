
import pybullet as pb
import pybullet_data
import trimesh.collision
import numpy as np
import os
import time

import trimesh as tr

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

import sys
sys.path.append('../third_party/pybullet_ompl')
import pb_ompl


class TrimeshWorld(object):
    def __init__(self):
        self.mesh_indicies = []
        self.collision_manager = trimesh.collision.CollisionManager()
        self.gripper_mesh = trimesh.load("assets/urdf/franka_description/meshes/franka_gripper_collision_mesh.stl")
        self.gripper_mesh = self.gripper_mesh.apply_transform(
            np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )

    def add_object(self, mesh, pos, quat, name):
        translation_matrix = tr.transformations.translation_matrix(pos)
        quat = np.concatenate([quat[-1:], quat[:-1]], axis=-1)
        rotation_matrix = tr.transformations.quaternion_matrix(quat)
        pose = translation_matrix @ rotation_matrix
        self.collision_manager.add_object(name, mesh=mesh, transform=pose)
        self.mesh_indicies.append(name)

    def reset_world(self):
        for idx in self.mesh_indicies:
            self.collision_manager.remove_object(idx)

        self.mesh_indicies = []

    def in_collision_with(self, mesh, transform, min_distance=0.0, epsilon=1.0 / 1e3):
        colliding = self.collision_manager.in_collision_single(mesh=mesh, transform=transform)
        if not colliding and min_distance > 0.0:
            distance = self.collision_manager.min_distance_single(mesh=mesh, transform=transform)
            if distance < min_distance - epsilon:
                colliding = True
        return colliding

    def check_grasp_collision(self, grasp_poses):
        T, success = [], []
        for t in grasp_poses:
            pos, quat = t[:3], t[3:]
            quat = np.concatenate([quat[-1:], quat[:-1]], axis=-1)
            translation = tr.transformations.translation_matrix(pos)
            rotation = tr.transformations.quaternion_matrix(quat)
            pose = translation @ rotation

            success.append(not self.in_collision_with(self.gripper_mesh, transform=pose))
            T.append(pose)

        return np.stack(T, axis=0), np.array(success)


class PyBulletOMPL():
    def __init__(self, config, debug_viz=False):
        self.obstacles = []
        self.attachment = []
        self.attach_offset = np.eye(4)

        self.cfg = config
        self.debug_viz = debug_viz

        self._total_timeout = config["total_timeout"]
        self._single_timeout = config["single_timeout"]

        pb.connect(pb.GUI if self.debug_viz else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1. / 240.)

        # load robot
        robot_id = pb.loadURDF(
            "assets/urdf/franka_description/robots/franka_r3_cvx_ompl.urdf",
            basePosition=(0, 0, 0), baseOrientation=(0, 0, 0, 1), useFixedBase=1,
            flags=pb.URDF_IGNORE_VISUAL_SHAPES | pb.URDF_USE_INERTIA_FROM_FILE
        )
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        self.trimesh_collision_interface = TrimeshWorld()
        # self.trac_ik_interface = TracIKSolver(
        #     "assets/urdf/franka_description/robots/franka_r3_cvx_ompl.urdf",
        #     "panda_link0",
        #     "panda_hand"
        # )

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

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.attachment, self.obstacles,
                                                valid_seg_frac=self.cfg['valid_seg_frac'])
        self.pb_ompl_interface.set_planner(config["planner"])

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)

        return robot_cuRobo_cfg

    def check_ik_collision(self, state):
        return self.pb_ompl_interface.is_state_valid(state)

    def solve_ik_curobo(self, pose):
        quat = tr.transformations.quaternion_from_matrix(pose[:3, :3])
        quat = torch.tensor(quat, device=self.tensor_args.device, dtype=torch.float)
        pos = torch.tensor(pose[:3, 3], device=self.tensor_args.device, dtype=torch.float)

        ik_pose = Pose(pos, quat)
        ik_result = self.ik_solver.solve(ik_pose)

        if bool(ik_result.success.reshape(-1).cpu().numpy()):
            return ik_result.solution[:7].cpu().numpy().reshape(-1)

        return None

    def reset_scene(self):
        for obstacle in self.obstacles:
            pb.removeBody(obstacle)
        for attach in self.attachment:
            pb.removeBody(attach)

        self.obstacles = []
        self.attachment = []
        self.pb_ompl_interface.obstacles = []
        self.pb_ompl_interface.attachment = []
        self.pb_ompl_interface.ss.clear()
        self.pb_ompl_interface.ss.clearStartStates()
        self.trimesh_collision_interface.reset_world()

    def update_scene(self, scene_info, goal_obj_idx, attach_goal_obj=False):
        self.reset_scene()

        dq, dt = scene_info['table_quat'], scene_info['table_pos']
        table_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=scene_info['table_dim'] / 2.)
        tableID = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_id,
                                     basePosition=dt, baseOrientation=dq)

        sq, st = scene_info['scene_quat'], scene_info['scene_pos']
        scene_file = f'{scene_info["scene_asset"]["asset_root"]}/{scene_info["scene_asset"]["urdf_file"]}'
        sceneID = pb.loadURDF(scene_file, basePosition=st, baseOrientation=sq, useFixedBase=True,
                              flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL | pb.URDF_MERGE_FIXED_LINKS)

        oq, ot = scene_info['object_quat'], scene_info['object_pos']
        objIDs = []
        for j in range(oq.shape[0]):
            obj_file = f'{scene_info["object_asset"][j]["asset_root"]}/{scene_info["object_asset"][j]["urdf_file"]}'
            obj_id = pb.loadURDF(obj_file, basePosition=ot[j], baseOrientation=oq[j])
            if attach_goal_obj and j == goal_obj_idx:
                self.attachment.append(obj_id)
            else:
                objIDs.append(obj_id)

        self.obstacles = [tableID, sceneID] + objIDs

        eq, et = scene_info['eef_quat'], scene_info['eef_pos']
        translation_matrix = tr.transformations.translation_matrix(et)
        eq = np.concatenate([eq[-1:], eq[:-1]], axis=-1)
        rotation_matrix = tr.transformations.quaternion_matrix(eq)
        eef_pose = translation_matrix @ rotation_matrix

        gq, gt = oq[goal_obj_idx], ot[goal_obj_idx]
        translation_matrix = tr.transformations.translation_matrix(gt)
        gq = np.concatenate([gq[-1:], gq[:-1]], axis=-1)
        rotation_matrix = tr.transformations.quaternion_matrix(gq)
        goal_obj_pose = translation_matrix @ rotation_matrix

        offset_pose = tr.transformations.translation_matrix([0, 0, self.cfg["attach_offset_z"]])

        attach_pose = np.linalg.inv(eef_pose) @ offset_pose @ goal_obj_pose

        self.pb_ompl_interface.set_planning_env(self.obstacles, self.attachment, attach_pose)

        for i, m in enumerate(scene_info['scene_asset']['meshes']):
            m = m.copy()
            m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]]))
            self.trimesh_collision_interface.add_object(m, st, sq, f'scene_{i}')

        for i, obj in enumerate(scene_info['object_asset']):
            if attach_goal_obj and i == goal_obj_idx:
                continue
            self.trimesh_collision_interface.add_object(obj['mesh'].copy(), ot[i], oq[i], f'obj_{i}')

    def check_collision_free_grasps(self, grasp_poses):
        return self.trimesh_collision_interface.check_grasp_collision(grasp_poses)

    def plan_goalset(self, start, goalset, ret_goal=False):
        self.robot.set_state(start)

        res, path = False, None
        start_time = time.time()
        while True:
            if time.time() - start_time > self._total_timeout:
                if ret_goal:
                    return False, None, None
                else:
                    return False, None

            self.pb_ompl_interface.ss.clear()
            self.pb_ompl_interface.ss.clearStartStates()
            random_idx = np.random.randint(len(goalset))
            goal = goalset[random_idx]

            goal_ik = self.solve_ik_curobo(goal)
            if goal_ik is None:
                print("IK not found.")
                continue

            if not self.check_ik_collision(goal_ik):
                print("IK Collision Detected.")
                continue

            res, path = self.pb_ompl_interface.plan_start_goal(start.tolist(), goal_ik.tolist(),
                                                               allowed_time=self._single_timeout,
                                                               threshold=self.cfg["goal_threshold"])
            print("Goal IK:", goal_ik)
            print("SS Goal:", self.pb_ompl_interface.ss.getGoal())

            if res:
                if ret_goal:
                    return res, path, goal
                else:
                    return res, path


class PyBulletOMPLPCD():
    def __init__(self, config, debug_viz=False, tmp_file_path='.'):

        self.obstacles = []

        self.cfg = config
        self.debug_viz = debug_viz

        self._total_timeout = config["total_timeout"]
        self._single_timeout = config["single_timeout"]
        self._occ_grid_size = config['occ_grid_size']

        self._mesh_file_path = tmp_file_path

        pb.connect(pb.GUI if self.debug_viz else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1. / 240.)

        # load robot
        robot_id = pb.loadURDF(
            "assets/urdf/franka_description/robots/franka_r3_cvx_ompl.urdf",
            basePosition=(0, 0, 0), baseOrientation=(0, 0, 0, 1), useFixedBase=1,
            flags=pb.URDF_IGNORE_VISUAL_SHAPES | pb.URDF_USE_INERTIA_FROM_FILE
        )
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        self.trimesh_collision_interface = TrimeshWorld()

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

        # setup pb_ompl
        robot_mesh_path = ['franka_description/meshes/collision_cvx/link0.obj',
                           'franka_description/meshes/collision_cvx/link1.obj',
                           'franka_description/meshes/collision_cvx/link2.obj',
                           'franka_description/meshes/collision_cvx/link3.obj',
                           'franka_description/meshes/collision_cvx/link4.obj',
                           'franka_description/meshes/collision_cvx/link5.obj',
                           'franka_description/meshes/collision_cvx/link6.obj',
                           'franka_description/meshes/collision_cvx/link7.obj',
                           'franka_description/meshes/collision_cvx/hand.obj',
                           'franka_description/meshes/collision_cvx/finger_4part.obj']
        robot_meshes = []
        for path in robot_mesh_path:
            mesh = trimesh.load(f'assets/urdf/{path}', force='mesh')
            robot_meshes.append(mesh)
            if path == robot_mesh_path[-1]:
                mesh2 = mesh.copy().apply_transform(tr.transformations.rotation_matrix(np.pi, [0, 0, 1]))
                robot_meshes.append(mesh2)

        self.pb_ompl_interface = pb_ompl.PbTmOMPL(self.robot, robot_meshes, valid_seg_frac=self.cfg['valid_seg_frac'])
        self.pb_ompl_interface.set_planner(config["planner"])

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)

        return robot_cuRobo_cfg

    def check_ik_collision(self, state):
        return self.pb_ompl_interface.is_state_valid(state)

    def solve_ik_curobo(self, pose):
        try:
            quat = tr.transformations.quaternion_from_matrix(pose[:3, :3])
            quat = torch.tensor(quat, device=self.tensor_args.device, dtype=torch.float)
            pos = torch.tensor(pose[:3, 3], device=self.tensor_args.device, dtype=torch.float)
        except:
            return None

        ik_pose = Pose(pos, quat)
        ik_result = self.ik_solver.solve(ik_pose)

        if bool(ik_result.success.reshape(-1).cpu().numpy()):
            return ik_result.solution[:7].cpu().numpy().reshape(-1)

        return None

    def reset_scene(self):
        for obstacle in self.obstacles:
            pb.removeBody(obstacle)

        self.obstacles = []
        self.pb_ompl_interface.reset()
        self.trimesh_collision_interface.reset_world()

    def update_scene(self, scene_info, goal_obj_idx, disable_goal_obj=False, attach_goal_obj=False):
        self.reset_scene()

        dq, dt = scene_info['table_quat'], scene_info['table_pos']
        table_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=scene_info['table_dim'] / 2.)
        tableID = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_id, basePosition=dt, baseOrientation=dq)
        pb_obstacles = [tableID]
        self.obstacles = pb_obstacles

        scene_mesh, goal_mesh = [], []
        if len(scene_info['scene_cloud']) > 0:
            pc = trimesh.PointCloud(scene_info['scene_cloud'])
            scene_mesh = [trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=self._occ_grid_size)]

        if len(scene_info['goal_cloud']) > 0:
            pc = trimesh.PointCloud(scene_info['goal_cloud'])
            goal_mesh = [trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=self._occ_grid_size)]

        eq, et = scene_info['eef_quat'], scene_info['eef_pos']
        translation_matrix = tr.transformations.translation_matrix(et)
        rotation_matrix = tr.transformations.quaternion_matrix(np.concatenate([eq[-1:], eq[:-1]], axis=-1))
        eef_pose = translation_matrix @ rotation_matrix

        offset_pose = tr.transformations.translation_matrix([0, 0, self.cfg["attach_offset_z"]])
        attach_pose = np.linalg.inv(eef_pose) @ offset_pose @ np.eye(4)

        if attach_goal_obj and not disable_goal_obj:
            tm_obstacles = scene_mesh
            attachment = goal_mesh
        elif attach_goal_obj and disable_goal_obj:
            tm_obstacles = scene_mesh
            attachment = []
        elif not attach_goal_obj and not disable_goal_obj:
            tm_obstacles = scene_mesh + goal_mesh
            attachment = []
        else:
            tm_obstacles = scene_mesh
            attachment = []

        self.pb_ompl_interface.set_planning_env(pb_obstacles, tm_obstacles, attachment, attach_pose)

        sq, st = scene_info['scene_quat'], scene_info['scene_pos']
        oq, ot = scene_info['object_quat'], scene_info['object_pos']

        for i, m in enumerate(scene_info['scene_asset']['meshes']):
            m = m.copy()
            m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]]))
            self.trimesh_collision_interface.add_object(m, st, sq, f'scene_{i}')

        for i, obj in enumerate(scene_info['object_asset']):
            if attach_goal_obj and i == goal_obj_idx:
                continue
            self.trimesh_collision_interface.add_object(obj['mesh'].copy(), ot[i], oq[i], f'obj_{i}')

    def check_collision_free_grasps(self, grasp_poses):
        return self.trimesh_collision_interface.check_grasp_collision(grasp_poses)

    def plan_goalset(self, start, goalset, ret_goal=False):
        self.robot.set_state(start)

        res, path = False, None
        start_time = time.time()
        while True:
            if time.time() - start_time > self._total_timeout:
                if ret_goal:
                    return False, None, None
                else:
                    return False, None

            self.pb_ompl_interface.ss.clear()
            self.pb_ompl_interface.ss.clearStartStates()
            random_idx = np.random.randint(len(goalset))
            goal = goalset[random_idx]

            goal_ik = self.solve_ik_curobo(goal)
            if goal_ik is None:
                continue

            if not self.check_ik_collision(goal_ik):
                print("IK Collision Detected.")
                continue

            res, path = self.pb_ompl_interface.plan_start_goal(start.tolist(), goal_ik.tolist(),
                                                               allowed_time=self._single_timeout,
                                                               threshold=self.cfg["goal_threshold"])
            print("Goal IK:", goal_ik)
            print("SS Goal:", self.pb_ompl_interface.ss.getGoal())

            if res:
                if ret_goal:
                    return res, path, goal
                else:
                    return res, path
