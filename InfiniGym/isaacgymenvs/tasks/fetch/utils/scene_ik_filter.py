from isaacgymenvs.utils.torch_jit_utils import (to_torch, get_axis_params, tensor_clamp,
                                                tf_vector, tf_combine, quat_mul, quat_conjugate,
                                                quat_to_angle_axis, tf_inverse, quat_apply,
                                                matrix_to_quaternion)
import numpy as np
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

import os
import torch
import trimesh
import trimesh.transformations as tr

from isaacgymenvs.tasks.fetch.load_utils import (get_env_config,
                                                 get_franka_panda_asset,
                                                 load_env_scene,
                                                 load_env_object,
                                                 load_env_object_combo,
                                                 InfiniSceneLoader)


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


def matrix_to_q_t(mtx):
    assert mtx.shape[1] == 4 and mtx.shape[2] == 4
    t = mtx[:, :3, 3]
    q = matrix_to_quaternion(mtx[:, :3, :3])
    q = torch.cat([q[..., 1:], q[..., :1]], dim=-1)
    return q, t


class CuRoboIKGraspChecker(object):
    def __init__(self, scene_path, max_pose_seed, debug_vis=False):
        self.loader = InfiniSceneLoader(scene_path)
        self.loader.load_env_config()

        task_configs = self.loader.load_task_config()
        self.task_actor_states = task_configs['task_init_state']
        self.task_obj_idx = task_configs['task_obj_index']
        self.task_obj_grasps, self.task_obj_assets = [], []
        self.task_scene_asset = load_env_scene(self.loader.scene_asset_config)
        self.task_table_dim = self.loader.robot_asset_config['table_dim']

        # Init CuRobo
        self.tensor_args = TensorDeviceType()
        self.init_torch_state()

        self._max_pose_seed = max_pose_seed
        self.debug_vis = debug_vis

        ik_config = IKSolverConfig.load_from_robot_config(
            self._get_cuRobo_robot_config(),
            self._get_cuRobo_world_config(),
            rotation_threshold=0.02,
            position_threshold=0.002,
            num_seeds=50,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=0.025
        )
        self.ik_solver = IKSolver(ik_config)
        self.ik_collision = self.ik_solver.world_coll_checker

    def init_torch_state(self):
        self.robot_state = to_torch(self.task_actor_states[..., 0, :], device=self.tensor_args.device, dtype=torch.float)
        self.scene_state = to_torch(self.task_actor_states[..., 2, :], device=self.tensor_args.device, dtype=torch.float)
        self.table_state = to_torch(self.task_actor_states[..., 1, :], device=self.tensor_args.device, dtype=torch.float)
        self.object_state = to_torch(self.task_actor_states[..., 3:, :], device=self.tensor_args.device, dtype=torch.float)
        self.load_objects()

    def load_objects(self):
        object_configs = self.loader.object_asset_config
        for config in object_configs:
            asset = load_env_object(config)
            grasp_poses = asset['grasp_poses']['T']
            grasp_label = asset['grasp_poses']['isaac_label_default']['force_label']
            success_grasp_poses = grasp_poses[np.where(grasp_label)[0]]

            grasps = to_torch(success_grasp_poses, device=self.tensor_args.device, dtype=torch.float)
            self.task_obj_grasps.append(grasps)
            self.task_obj_assets.append(asset)

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)

        return robot_cuRobo_cfg

    def _get_cuRobo_world_config(self):
        table_pose, scene_pose, object_pose = self._get_pose_in_robot_frame()

        tq, oq, sq = table_pose['quat'], object_pose['quat'], scene_pose['quat']
        tq = torch.concat([tq[..., -1:], tq[..., :-1]], dim=-1)
        oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
        sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)

        tq, tt = tq.cpu().numpy(), table_pose['pos'].cpu().numpy()
        sq, st = sq.cpu().numpy(), scene_pose['pos'].cpu().numpy()
        oq, ot = oq.cpu().numpy(), object_pose['pos'].cpu().numpy()

        t_cube = Cuboid(
            name=f"table",
            pose=[*tt[0], *tq[0]],
            dims=self.task_table_dim,
        )

        scene_meshes = []
        for j, f in enumerate(self.task_scene_asset['files']):
            c_mesh = Mesh(
                name=f"mesh_{j}",
                pose=[*st[0], *sq[0]],
                file_path=f,
                scale=[1.0, 1.0, 1.0],
            )
            scene_meshes.append(c_mesh)


        # add object asset
        object_meshes = []
        oq_i, ot_i = oq[0], ot[0]

        for j, obj in enumerate(self.task_obj_assets):
            q, t = oq_i[j], ot_i[j]
            o_mesh = Mesh(
                name=f"obj_{j}",
                pose=[*t, *q],
                file_path=obj['file'],
                scale=[1.0, 1.0, 1.0]
            )
            object_meshes.append(o_mesh)

        world_config = WorldConfig(mesh=scene_meshes + object_meshes, cuboid=[t_cube])
        return world_config

    def _get_pose_in_robot_frame(self):
        rq, rt = tf_inverse(self.robot_state[..., 3:7], self.robot_state[..., :3])
        dq, dt = tf_combine(rq, rt, self.table_state[..., 3:7], self.table_state[..., :3])
        sq, st = tf_combine(rq, rt, self.scene_state[..., 3:7], self.scene_state[..., :3])
        oq, ot = tf_combine(rq.unsqueeze(1).repeat(1, self.object_state.shape[1], 1),
                            rt.unsqueeze(1).repeat(1, self.object_state.shape[1], 1),
                            self.object_state[..., 3:7], self.object_state[..., :3])

        return {'quat': dq, 'pos': dt}, {'quat': sq, 'pos': st}, {'quat': oq, 'pos': ot}

    def _update_cuRobo_world_collider_state(self, i):
        table_pose, scene_pose, object_pose = self._get_pose_in_robot_frame()

        tq, tt = table_pose['quat'], table_pose['pos']
        tq = torch.concat([tq[..., -1:], tq[..., :-1]], dim=-1)
        pose = Pose(tt[i:i+1], tq[i:i+1])
        self.ik_collision.update_obb_pose(w_obj_pose=pose, name=f'table', env_idx=0)

        sq, st = scene_pose['quat'], scene_pose['pos']
        sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
        pose = Pose(st[i:i+1], sq[i:i+1])

        for j, f in enumerate(self.task_scene_asset['files']):
            self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'mesh_{j}', env_idx=0)

        oq, ot = object_pose['quat'], object_pose['pos']
        oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
        for j in range(self.object_state.shape[1]):
            pose = Pose(ot[i:i+1, j], oq[i:i+1, j])
            self.ik_collision.update_mesh_pose(w_obj_pose=pose, name=f'obj_{j}', env_idx=0)

    def _sample_goal_obj_annotated_grasp_pose(self, i):
        _, _, object_pose = self._get_pose_in_robot_frame()
        oq, ot = object_pose['quat'], object_pose['pos']

        sample_grasps = []
        goal_idx = self.task_obj_idx[i]
        grasp_quat, grasp_pos = matrix_to_q_t(self.task_obj_grasps[goal_idx])
        random_batch = torch.randint(grasp_pos.shape[0], size=(self._max_pose_seed,))

        sample_quat, sample_pos = grasp_quat[random_batch], grasp_pos[random_batch]
        oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(self._max_pose_seed, 1),
                      ot[i:i+1, goal_idx].repeat(self._max_pose_seed, 1))
        gq, gt = tf_combine(oq_i, ot_i, sample_quat, sample_pos)
        gq = torch.concat([gq[..., -1:], gq[..., :-1]], dim=-1)
        return torch.concat([gt, gq], dim=-1)

    def _enable_goal_obj_collision_checking(self, i, enable=True):
        goal_idx = self.task_obj_idx[i]
        self.ik_collision.enable_obstacle(f'obj_{goal_idx}', enable=enable, env_idx=0)

    def check_goal_obj_collision_free_grasp_pose(self, i):
        # Use IK solver to solve for candidate grasp pose
        self._update_cuRobo_world_collider_state(i)
        grasp_candidate = self._sample_goal_obj_annotated_grasp_pose(i)

        # Check collision-free IK at grasp pose (disable goal obj)
        self._enable_goal_obj_collision_checking(i, False)

        grasp_pose = Pose(grasp_candidate[..., :3].unsqueeze(0), grasp_candidate[..., 3:7].unsqueeze(0))
        ik_result = self.ik_solver.solve_goalset(grasp_pose.clone())
        torch.cuda.synchronize()

        self.ik_collision.enable_all_obstacle()

        if ik_result.success.cpu()[0] and self.debug_vis:
            table_pose, scene_pose, object_poses = self._get_pose_in_robot_frame()
            ik_pose = self.ik_solver.fk(ik_result.solution[0]).ee_pose
            self.grasp_vis_debug(table_pose, scene_pose, object_poses, grasp_pose[0], ik_pose, i)

        return ik_result.success, ik_result.solution[0]

    def check_scene_task_ik_free_grasps(self):
        results = []
        for i in range(self.task_actor_states.shape[0]):
            success, ik = self.check_goal_obj_collision_free_grasp_pose(i)
            results.append(success.cpu().numpy()[0])

        return results

    def grasp_vis_debug(self, table_pose, scene_pose, object_poses, grasp_pose, ik_pose, k):
        scene = trimesh.Scene()

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        tq = torch.concat([table_pose['quat'][k, -1:], table_pose['quat'][k, :-1]], dim=-1)
        tq, tt = tq.cpu().numpy(), table_pose['pos'][k].cpu().numpy()

        table_translation = tr.translation_matrix(tt)
        table_rotation = tr.quaternion_matrix(tq)

        table = trimesh.creation.box(extents=self.task_table_dim, transform=table_translation @ table_rotation)
        scene.add_geometry(table)

        sq = torch.concat([scene_pose['quat'][k, -1:], scene_pose['quat'][k, :-1]], dim=-1)
        sq, st = sq.cpu().numpy(), scene_pose['pos'][k].cpu().numpy()

        # vis environment 0
        scene_translation = tr.translation_matrix(st)
        scene_rotation = tr.quaternion_matrix(sq)

        # vis scene
        for f in self.task_scene_asset['files']:
            mesh = trimesh.load(f)
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        oq = torch.concat([object_poses['quat'][k, :, -1:], object_poses['quat'][k, :, :-1]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_poses['pos'][k].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.task_obj_assets):
            trans = tr.translation_matrix(ot[i])
            rot = tr.quaternion_matrix(oq[i])
            mesh = o['mesh'].copy().apply_transform(trans @ rot)
            scene.add_geometry(mesh)

        # grasp pose
        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(min(grasp_pose.position.shape[0], 10)):
            trans = tr.translation_matrix(grasp_pose.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(grasp_pose.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        if ik_pose is not None:
            for i in range(ik_pose.position.shape[0]):
                trans = tr.translation_matrix(ik_pose.position[i].cpu().numpy())
                rot = tr.quaternion_matrix(ik_pose.quaternion[i].cpu().numpy())
                ik_grasp = trans @ rot @ vis_rot
                command_marker = create_gripper_marker([0, 255, 0]).apply_transform(ik_grasp)
                scene.add_geometry(command_marker)

        scene.show()
