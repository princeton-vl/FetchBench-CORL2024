
import numpy as np
import os
import torch
import imageio
import trimesh.transformations as tra

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse, matrix_to_quaternion
from isaacgymenvs.tasks.fetch.vec_task import VecTask
from isaacgymenvs.tasks.fetch.utils.load_utils import (get_env_config,
                                                 get_franka_panda_asset,
                                                 load_env_scene,
                                                 load_env_object,
                                                 load_env_object_combo,
                                                 InfiniSceneLoader)


def image_to_video(obs_buf):
    video = []
    for s, images in enumerate([obs_buf]):
        steps = []
        for e, imgs in enumerate(images):
            steps.append(np.concatenate(imgs, axis=0))
        video.append(np.concatenate(steps, axis=1))
    return video


class FetchBase(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.cfg["env"]["numObservations"] = 0
        self.cfg["env"]["numActions"] = 0
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.scene_config_path = self.cfg["task"]["scene_config_path"]
        if self.cfg["env"]["numEnvs"] != len(self.scene_config_path):
            self.cfg["env"]["numEnvs"] = len(self.scene_config_path)

        # Env params
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.arm_control_type = self.cfg["env"]["armControlType"]
        self.gripper_control_type = self.cfg["env"]["gripperControlType"]
        self.osc_control_repeat = self.cfg["env"]["oscControlRepeat"]

        # Values to be filled in at runtime
        self.states = {}                          # will be dict filled with relevant states to use for reward calculation
        self.robot_handles = {}                   # will be dict mapping names to relevant sim handles
        self.num_robot_dofs = None                # Total number of DOFs per env
        self.num_objs = self.cfg["env"]["numObjs"]

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None              # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None       # State of all joints       (n_envs, n_dof)
        self._contact_force_state = None    # Contact of all rigid bodies

        self.rigid_body_index_map = {}

        # Task configs
        self.task_actor_init_state = []
        self.task_camera_init_state = []
        self.task_obj_label = []
        self.task_obj_index = []
        self._task_idx = -1
        self.task_obj_color = gymapi.Vec3(1.0, 0.0, 0.0)
        self.default_obj_color = gymapi.Vec3(0.0, 0.0, 1.0)

        # Robot state
        self._q = None                      # Joint positions           (n_envs, n_dof)
        self._qd = None                     # State of all rigid bodies (n_envs, n_bodies, 13)
        self._robot_base_state = None
        self._table_base_state = None
        self._eef_state = None              # end effector state (at grasping point)
        self._eef_lf_state = None           # end effector state (at left fingertip)
        self._eef_rf_state = None           # end effector state (at left fingertip)
        self._left_finger_force = None
        self._right_finger_force = None
        self._j_eef = None                  # Jacobian for end effector
        self._mm = None                     # Mass matrix

        # Scene state
        self._scene_base_state = None

        # Object state
        self._obj_state = None
        self._obj_contact_force = None
        self.obj_ref_point = None

        # Cam:
        self.cam_params = None
        self._cam_config = None

        # control
        self._arm_control = None            # Tensor buffer for controlling arm
        self._gripper_control = None        # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._vel_control = None            # Vel actions

        # reset
        self.robot_effort_limits = None     # Actuator effort limits for robot
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        # Joint PD
        self.pd_gain = self.cfg["env"]["robot"]["joint_gain"]
        self.pd_damp = self.cfg["env"]["robot"]["joint_damp"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # OSC PD
        self.kp = to_torch([self.cfg["env"]["robot"]["osc_gain"]] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([0.] * 7, device=self.device)
        self.kd_null = to_torch([self.cfg["env"]["robot"]["osc_null_damp"]] * 7, device=self.device)

        self.robot_default_dof_pos = (
            to_torch([-0.31267092, -1.1996635, 0.05781832, -2.1767514,
                      0.06494738, 0.9786396, 0.53001183, 0.04, 0.04], device=self.device)
        )

        self.robot_joint_names = ["panda_joint1",
                                  "panda_joint2",
                                  "panda_joint3",
                                  "panda_joint4",
                                  "panda_joint5",
                                  "panda_joint6",
                                  "panda_joint7"]

        # Reset all environments
        self.reset_task(0)
        self._refresh()

        self._traj_length = to_torch([0 for _ in range(self.num_envs)], device=self.device, dtype=torch.float)

    """
    Load Assets
    """

    def load_env_asset(self, config):
        loader = InfiniSceneLoader(get_env_config(config))
        loader.load_env_config()

        # no robot
        scene_asset = self.load_scene_asset(loader.scene_asset_config)
        table_asset = self.load_table_asset(loader.robot_asset_config)
        object_asset = []
        for o in loader.object_asset_config:
            object_asset.append(self.load_object_asset(o))
        combo_asset = []
        for o in loader.combo_asset_config:
            combo_asset.append(self.load_combo_asset(o))

        assert loader.camera_config["hov"] == self.cfg["env"]["cam"]["hov"]
        # cam params
        cam_params = self.load_camera_asset(self.cfg["env"]["cam"])

        return scene_asset, table_asset, object_asset, combo_asset, cam_params, loader

    def load_robot_asset(self):
        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0                                  # default = 0.02
        asset_options.density = 1000.0                                 # default = 1000.0
        asset_options.armature = self.cfg["env"]["robot"]["armature"]  # default = 0.0
        asset_options.enable_gyroscopic_forces = True
        asset_options.use_physx_armature = True
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        if self.cfg["env"]["robot"]["add_damping"]:
            asset_options.linear_damping = 0.1              # default = 0.0; increased to improve stability
            asset_options.max_linear_velocity = 10.0        # default = 1000.0; reduced to prevent CUDA errors
            asset_options.angular_damping = 0.5             # default = 0.5; increased to improve stability
            asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            asset_options.linear_damping = 0.0                       # default = 0.0
            asset_options.max_linear_velocity = 1000.0               # default = 1000.0
            asset_options.angular_damping = 0.5                      # default = 0.5
            asset_options.max_angular_velocity = 2 * np.pi           # default = 64.0

        robot_asset_path = get_franka_panda_asset(type=self.cfg["env"]["robot"]["type"])
        robot_asset = self.gym.load_asset(self.sim, robot_asset_path['asset_root'],
                                          robot_asset_path['urdf_file'], asset_options)

        r_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for p in r_props:
            p.friction = self.cfg["env"]["robot"]["friction"]
            p.restitution = self.cfg["env"]["robot"]["restitution"]
            p.rolling_friction = 0.0  # default = 0.0
            p.torsion_friction = 0.0  # default = 0.0
            p.compliance = 0.0  # default = 0.0
            p.thickness = 0.0  # default = 0.0
            p.contact_offset = self.cfg["env"]["robot"]["contact_offset"]
            p.rest_offset = 0.0
        self.gym.set_asset_rigid_shape_properties(robot_asset, r_props)

        return robot_asset

    def load_scene_asset(self, config):
        # load scene asset
        scene = load_env_scene(config)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        s = self.gym.load_asset(self.sim, scene['asset_root'], scene['urdf_file'], asset_options)

        s_props = self.gym.get_asset_rigid_shape_properties(s)
        for p in s_props:
            p.friction = config["friction"]
            p.restitution = config["restitution"]
            p.rolling_friction = 0.0  # default = 0.0
            p.torsion_friction = 0.0  # default = 0.0
            p.compliance = 0.0  # default = 0.0
            p.thickness = 0.0  # default = 0.0
            p.rest_offset = config["rest_offset"]
            p.contact_offset = config["contact_offset"]
        self.gym.set_asset_rigid_shape_properties(s, s_props)
        scene['asset'] = s

        return scene

    def load_object_asset(self, config):
        obj = load_env_object(config)

        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.0
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.density = config['density']
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.use_mesh_materials = True
        asset_options.override_inertia = True
        asset_options.override_com = True
        if self.cfg["env"]["objects"]["add_damping"]:
            asset_options.linear_damping = 0.05             # default = 0.0; increased to improve stability
            asset_options.max_linear_velocity = 10.0        # default = 1000.0; reduced to prevent CUDA errors
            asset_options.angular_damping = 0.5             # default = 0.5; increased to improve stability
            asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            asset_options.linear_damping = 0.0                       # default = 0.0
            asset_options.max_linear_velocity = 1.0                  # default = 1000.0
            asset_options.angular_damping = 0.5                      # default = 0.5
            asset_options.max_angular_velocity = 2 * np.pi           # default = 64.0

        o = self.gym.load_asset(self.sim, obj['asset_root'], obj['urdf_file'], asset_options)
        o_props = self.gym.get_asset_rigid_shape_properties(o)
        for p in o_props:
            p.friction = config["friction"]
            p.restitution = config["restitution"]
            p.rolling_friction = self.cfg["env"]["objects"]["rolling_friction"]  # default = 0.0
            p.torsion_friction = self.cfg["env"]["objects"]["torsion_friction"]  # default = 0.0
            p.compliance = 0.0        # default = 0.0
            p.thickness = 0.0         # default = 0.0
            p.rest_offset = config["rest_offset"]
            p.contact_offset = config["contact_offset"]

        self.gym.set_asset_rigid_shape_properties(o, o_props)
        obj['asset'] = o

        return obj

    def load_combo_asset(self, config):
        combo = load_env_object_combo(config)

        assets = []
        for i, obj in enumerate(combo['urdf_file']):
            asset_options = gymapi.AssetOptions()
            asset_options.thickness = 0.0
            asset_options.fix_base_link = combo['fixed_base'][i]
            asset_options.collapse_fixed_joints = True
            asset_options.density = config['density']  # default = 1000.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.use_mesh_materials = True
            asset_options.override_inertia = True
            asset_options.override_com = True
            if self.cfg["env"]["objects"]["add_damping"]:
                asset_options.linear_damping = 0.05  # default = 0.0; increased to improve stability
                asset_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
                asset_options.angular_damping = 0.5  # default = 0.5; increased to improve stability
                asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
            else:
                asset_options.linear_damping = 0.0  # default = 0.0
                asset_options.max_linear_velocity = 1.0  # default = 1000.0
                asset_options.angular_damping = 0.5  # default = 0.5
                asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0

            o = self.gym.load_asset(self.sim, combo['asset_root'], obj, asset_options)
            o_props = self.gym.get_asset_rigid_shape_properties(o)
            for p in o_props:
                p.friction = config['friction']
                p.restitution = config['restitution']
                p.rolling_friction = self.cfg["env"]["objects"]['rolling_friction']  # default = 0.0
                p.torsion_friction = self.cfg["env"]["objects"]['torsion_friction']  # default = 0.0
                p.compliance = 0.0  # default = 0.0
                p.thickness = 0.0  # default = 0.0
                p.rest_offset = config["rest_offset"]
                p.contact_offset = config['contact_offset']

            self.gym.set_asset_rigid_shape_properties(o, o_props)
            assets.append(o)
        combo['asset'] = assets

        return combo

    def load_table_asset(self, config):

        dim = config["table_dim"]
        table_dims = gymapi.Vec3(*dim)

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.0
        asset_options.fix_base_link = True
        asset_options.thickness = 0.0
        asset_options.disable_gravity = True
        table_asset = self.gym.create_box(self.sim,
                                          table_dims.x,
                                          table_dims.y,
                                          table_dims.z,
                                          asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = config["friction"]
            p.restitution = config["restitution"]
            p.rolling_friction = 0.0
            p.torsion_friction = 0.0
            p.compliance = 0.0
            p.thickness = 0.0
            p.contact_offset = config["contact_offset"]
            p.rest_offset = config["rest_offset"]
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)

        asset = {
            'asset': table_asset,
            'dim': config["table_dim"]
        }
        return asset

    def load_camera_asset(self, config):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = config["hov"]
        camera_props.width = config["width"]
        camera_props.height = config["height"]
        camera_props.enable_tensors = True

        return camera_props

    def load_asset_grasp_poses(self, o):
        T = o['grasp_poses']['T']

        if self.cfg.get('solution', {}).get('grasp_label', None) is not None:
            label = np.ones_like(o['grasp_poses']['acronym_label']).astype(np.bool_)
            gripper_type = self.cfg["solution"]["grasp_label"]["gripper_type"]

            if self.cfg["solution"]["grasp_label"]["use_flex_label"]:
                label = label & o['grasp_poses']['acronym_label'].astype(np.bool_)
            if self.cfg["solution"]["grasp_label"]["use_isaac_force_label"]:
                label = label & o['grasp_poses'][f'isaac_label_{gripper_type}']['force_label'].astype(np.bool_)
            if self.cfg["solution"]["grasp_label"]["use_isaac_success_label"]:
                label = label & o['grasp_poses'][f'isaac_label_{gripper_type}']['success'].astype(np.bool_)
        else:
            label = o['grasp_poses']['acronym_label'].astype(np.bool_)

        success_T = T[np.where(label)[0]]
        if len(success_T) == 0:
            # add a place holder impossible grasp
            success_T = np.array([tra.translation_matrix([0, 0, 25.])])

        return to_torch(success_T, device=self.device, dtype=torch.float32)

    """
    Robot Sensors and Configs
    """

    def get_robot_dof_props(self, asset):
        robot_dof_props = self.gym.get_asset_dof_properties(asset)
        if self.arm_control_type == 'joint':
            robot_dof_stiffness = to_torch([self.pd_gain] * 7, dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([self.pd_damp] * 7, dtype=torch.float, device=self.device)

            for i in range(self.num_robot_dofs - 2):
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]

        elif self.arm_control_type == 'osc':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

            for i in range(self.num_robot_dofs - 2):
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]
        else:
            raise NotImplementedError

        if self.gripper_control_type == 'position':
            robot_dof_stiffness = to_torch([1e4, 1e4], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([4e2, 4e2], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_POS
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]
                robot_dof_props['effort'][i + self.num_robot_dofs - 2] = 400

        elif self.gripper_control_type == 'velocity':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([7e2, 7e2], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_VEL
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]
                robot_dof_props['effort'][i + self.num_robot_dofs - 2] = 140

        elif self.gripper_control_type == 'effort':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([0, 0], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]
                robot_dof_props['effort'][i + self.num_robot_dofs - 2] = 100
        else:
            raise NotImplementedError

        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_effort_limits = []
        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self.robot_effort_limits.append(robot_dof_props['effort'][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_effort_limits = to_torch(self.robot_effort_limits, device=self.device)

        return robot_dof_props

    def get_robot_contacts(self):
        contact_dicts = []
        for i in range(self.num_envs):
            contacts = self.gym.get_env_rigid_contacts(self.envs[i])
            relations = []
            for contact in contacts:
                relation = []
                for n in ['body0', 'body1']:
                    if contact[n] == -1:
                        name = 'ground'
                    elif contact[n] in self.rigid_body_index_map:
                        name = self.rigid_body_index_map[contact[n]]
                    else:
                        raise NotImplementedError
                    relation.append(name)
                relations.append(relation)
            contact_dicts.append(relations)

        return contact_dicts

    """
    Create Envs
    """

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        super()._create_gym_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        robot_asset = self.load_robot_asset()
        self.robot_asset = robot_asset

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        robot_dof_props = self.get_robot_dof_props(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        max_agg_bodies = self.cfg["env"]["aggregateBody"]
        max_agg_shapes = self.cfg["env"]["aggregateShape"]

        self.robots = []
        self.scenes = []
        self.objects = []
        self.tables = []
        self.cameras = []
        self.envs = []

        self.actor_init_state = []
        self.camera_init_state = []
        self.obj_init_label = []
        self.obj_ref_point = []
        self.obj_grasp_poses = []

        # setup all envs
        self.scene_asset = []
        self.table_asset = []
        self.object_asset = []
        self.combo_asset = []

        for n in range(self.num_envs):
            seg_idx, actor_init_state = 1, []
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            scene_asset, table_asset, object_assets, combo_assets, camera_props, loader = \
                self.load_env_asset(self.scene_config_path[n])

            self.scene_asset.append(scene_asset)
            self.table_asset.append(table_asset)
            self.combo_asset.append(combo_assets)
            self.object_asset.append(object_assets)

            scene_start_pose = vec_state_to_pose(loader.scene_pose[0])
            robot_start_pose = vec_state_to_pose(loader.robot_pose[0][:13])
            table_start_pose = vec_state_to_pose(loader.robot_pose[0][13:])

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # NOTE: franka should ALWAYS be loaded first in sim!
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", n,
                                                self.cfg["env"]["robot"]["disable_self_collision"], seg_idx)
            seg_idx += 1
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            table_actor = self.gym.create_actor(env_ptr, table_asset['asset'], table_start_pose, "table", n, 0, seg_idx)
            seg_idx += 1
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.0))

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            scene_actor = self.gym.create_actor(env_ptr, scene_asset['asset'], scene_start_pose, "scene", n, 0, seg_idx)
            seg_idx += 1

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            object_actors, object_ref_points, object_init_states, object_init_labels, object_grasp_poses = [], [], [], [], []

            # add combo objects
            combo_transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32).T
            for k, cbo in enumerate(combo_assets):
                assert f'obj_combo_{k}' == cbo['name']  # make sure the order is the same
                org_start_pose = vec_state_to_pose(loader.object_poses[0][2*k])
                org_actor = self.gym.create_actor(env_ptr, cbo['asset'][0], org_start_pose,
                                                  f"obj_combo_{k}_org", n, 0, seg_idx)
                seg_idx += 1

                object_actors.append(org_actor)
                object_ref_points.append(apply_transform(cbo['metadata']['organizer_com'], combo_transform))
                object_grasp_poses.append(None)

                self.gym.set_rigid_body_color(env_ptr, org_actor, 0, gymapi.MESH_VISUAL, self.default_obj_color)

                obj_start_pose = vec_state_to_pose(loader.object_poses[0][2*k+1])
                obj_actor = self.gym.create_actor(env_ptr, cbo['asset'][1], obj_start_pose,
                                                  f"obj_combo_{k}_obj", n, 0, seg_idx)
                seg_idx += 1
                object_actors.append(obj_actor)
                object_ref_points.append(apply_transform(cbo['metadata']['object_com'], combo_transform))
                object_grasp_poses.append(None)   # Todo: Add sample Grasp Poses.
                self.gym.set_rigid_body_color(env_ptr, obj_actor, 0, gymapi.MESH_VISUAL, self.default_obj_color)

            idx_offset = len(combo_assets) * 2
            # add rigid objects
            for k, o in enumerate(object_assets):
                assert f'obj_{k}' == o['name']  # make sure the order is the same
                obj_start_pose = vec_state_to_pose(loader.object_poses[0][idx_offset + k])
                object_actor = self.gym.create_actor(env_ptr, o['asset'], obj_start_pose,
                                                     f"obj_rigid_{k}", n, 0, seg_idx)
                seg_idx += 1
                object_actors.append(object_actor)
                object_ref_points.append(o['metadata']['com'])

                grasps = self.load_asset_grasp_poses(o)
                q, t = matrix_to_q_t(grasps)
                object_grasp_poses.append(torch.concat([t, q], dim=-1))

                # set default color
                self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL, self.default_obj_color)

            # add env cams
            cams = []
            for j in range(self.cfg["env"]["cam"]["num_cam"]):
                camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                cam_pose = loader.camera_poses[0][j]
                self.gym.set_camera_location(camera_handle, env_ptr,
                                             gymapi.Vec3(*cam_pose[:3]), gymapi.Vec3(*cam_pose[3:]))
                cams.append(camera_handle)

            # add vis cam
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(camera_handle, env_ptr,
                                         gymapi.Vec3(*[-2, 0, 2.5]), gymapi.Vec3(*[0, 0, 0.5]))
            cams.append(camera_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            assert len(object_actors) == self.num_objs == len(loader.object_labels[0]) == len(object_ref_points)
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.scenes.append(scene_actor)
            self.objects.append(object_actors)
            self.tables.append(table_actor)
            self.cameras.append(cams)

            self._create_env_tasks(loader)
            self.obj_ref_point.append(object_ref_points)
            self.obj_grasp_poses.append(object_grasp_poses)

        self.init_rb_index_map()
        self.init_torch_data()

    def _create_env_tasks(self, loader):
        task_config = loader.load_task_config()
        num_tasks = self.cfg["env"]["numTasks"]
        if num_tasks is None:
            num_tasks = len(task_config['task_init_state'])
        assert len(task_config['task_init_state']) >= num_tasks
        self.task_actor_init_state.append(task_config['task_init_state'][:num_tasks])
        self.task_camera_init_state.append(task_config['task_camera_pose'][:num_tasks])
        self.task_obj_index.append(task_config['task_obj_index'][:num_tasks])
        self.task_obj_label.append(task_config['task_obj_label'][:num_tasks].tolist())

    def init_rb_index_map(self):
        rb_map = {}

        link_names = list(self.gym.get_actor_rigid_body_dict(self.envs[0], self.robots[0]).keys())
        for link in link_names:
            rb_map[self.gym.find_actor_rigid_body_index(self.envs[0], self.robots[0], link, gymapi.DOMAIN_ENV)] = link

        scene_name = list(self.gym.get_actor_rigid_body_dict(self.envs[0], self.scenes[0]).keys())[0]
        rb_map[self.gym.find_actor_rigid_body_index(self.envs[0], self.scenes[0], scene_name, gymapi.DOMAIN_ENV)] = 'scene'
        table_name = list(self.gym.get_actor_rigid_body_dict(self.envs[0], self.tables[0]).keys())[0]
        rb_map[self.gym.find_actor_rigid_body_index(self.envs[0], self.tables[0], table_name, gymapi.DOMAIN_ENV)] = 'table'

        for obj_idx, obj_actor in enumerate(self.objects[0]):
            obj_name = list(self.gym.get_actor_rigid_body_dict(self.envs[0], obj_actor).keys())[0]
            rb_map[self.gym.find_actor_rigid_body_index(self.envs[0], obj_actor, obj_name, gymapi.DOMAIN_ENV)] = f'obj_{obj_idx}'

        self.rigid_body_index_map = rb_map

    def init_torch_data(self):
        env_ptr, robot_ptr = self.envs[0], self.robots[0]

        self.robot_handles = {
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, robot_ptr, "panda_hand"),
            "left_finger": self.gym.find_actor_rigid_body_handle(env_ptr, robot_ptr, "panda_leftfinger"),
            "right_finger": self.gym.find_actor_rigid_body_handle(env_ptr, robot_ptr, "panda_rightfinger"),
            "left_finger_id": self.gym.find_actor_rigid_body_index(env_ptr, robot_ptr,
                                                                   'panda_leftfinger', gymapi.DOMAIN_ENV),
            "right_finger_id": self.gym.find_actor_rigid_body_index(env_ptr, robot_ptr,
                                                                    'panda_rightfinger', gymapi.DOMAIN_ENV)
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        _robot_jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "robot"))
        _robot_mm_tensor = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, "robot"))

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._contact_force_state = gymtorch.wrap_tensor(_contact_force_tensor).view(self.num_envs, -1, 3)

        # robot states
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._robot_base_state = self._root_state[:, 0, :]
        self._table_base_state = self._root_state[:, 1, :]
        self._eef_state = self._rigid_body_state[:, self.robot_handles["hand"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.robot_handles["left_finger"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.robot_handles["right_finger"], :]

        # robot finger force
        self._left_finger_force = self._contact_force_state[:, self.robot_handles['left_finger_id'], 0:3]
        self._right_finger_force = self._contact_force_state[:, self.robot_handles['right_finger_id'], 0:3]

        # scene states
        self._scene_base_state = self._root_state[:, 2, :]

        # end-effector jacobian and inertia matrix for OSC
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, robot_ptr)['panda_hand_joint']
        self._j_eef = _robot_jacobian_tensor[:, hand_joint_index, :, :7]
        self._mm = _robot_mm_tensor[:, :7, :7]

        # object states
        self._obj_state = self._root_state[:, -self.num_objs:, :]
        self.obj_ref_point = to_torch(self.obj_ref_point, device=self.device, dtype=torch.float)
        self.task_actor_init_state = to_torch(self.task_actor_init_state, device=self.device, dtype=torch.float)
        self.task_obj_index = to_torch(self.task_obj_index, device=self.device, dtype=torch.long)

        self.init_control()

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (3 + self.num_objs), dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def init_control(self):
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros_like(self._pos_control)
        self._effort_control = torch.zeros_like(self._pos_control)

        if self.arm_control_type == 'osc':
            self._arm_control = self._effort_control[:, :7]
        elif self.arm_control_type == 'joint':
            self._arm_control = self._pos_control[:, :7]
        else:
            raise NotImplementedError

        if self.gripper_control_type == 'effort':
            self._gripper_control = self._effort_control[:, 7:9]
        elif self.gripper_control_type == 'position':
            self._gripper_control = self._pos_control[:, 7:9]
        elif self.gripper_control_type == 'velocity':
            self._gripper_control = self._vel_control[:, 7:9]
        else:
            raise NotImplementedError

    """
    Env Step
    """

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self.states.update({
            "q": self._q[:, :],
            "qd": self._qd[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            "eef_lf_force": self._left_finger_force[:],
            "eef_rf_force": self._right_finger_force[:],

            "obj_pos": self._obj_state[..., :3],
            "obj_quat": self._obj_state[..., 3:7],
            "obj_vel": self._obj_state[..., 7:]

        })

    def reset_idx(self, env_ids):
        pos = tensor_clamp(
            self.robot_default_dof_pos.unsqueeze(0),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits
        )

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.robot_default_dof_pos[-2:]
        pos = pos.repeat(len(env_ids), 1)

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

    def reset_cam(self):
        for i, cams in enumerate(self.cameras):
            env_ptr = self.envs[i]
            for j, c in enumerate(cams[:-1]):
                cam_pose = self.task_camera_init_state[i][self._task_idx][j]
                self.gym.set_camera_location(c, env_ptr, gymapi.Vec3(*cam_pose[:3]), gymapi.Vec3(*cam_pose[3:]))

    def post_phy_step(self):
        last_q = self.states["q"].clone()
        self._refresh()
        curr_q = self.states["q"].clone()
        step_length = torch.norm(curr_q - last_q, dim=-1)
        self._traj_length += step_length

        if self.viewer is not None and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            # Plot visualizations
            for i in range(self.num_envs):
                for k in range(self.num_objs):
                    pos, rot, com = self.states["obj_pos"][i][k], self.states["obj_quat"][i][k], self.obj_ref_point[i][k]
                    p0 = (pos + quat_apply(rot, com)).cpu().numpy()

                    px = (pos + quat_apply(rot, com + to_torch([1, 0, 0], device=self.device) * 0.15)).cpu().numpy()
                    py = (pos + quat_apply(rot, com + to_torch([0, 1, 0], device=self.device) * 0.15)).cpu().numpy()
                    pz = (pos + quat_apply(rot, com + to_torch([0, 0, 1], device=self.device) * 0.15)).cpu().numpy()

                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                       [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                       [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                       [0.1, 0.1, 0.85])
                    #self.gym.draw_env_rigid_contacts(self.viewer, self.envs[i], gymapi.Vec3(0.0, 0.0, 1.0), 5.0, False)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                px = (eef_pos[i] + quat_apply(eef_rot[i], to_torch([1, 0, 0], device=self.device) * 0.25)).cpu().numpy()
                py = (eef_pos[i] + quat_apply(eef_rot[i], to_torch([0, 1, 0], device=self.device) * 0.25)).cpu().numpy()
                pz = (eef_pos[i] + quat_apply(eef_rot[i], to_torch([0, 0, 1], device=self.device) * 0.25)).cpu().numpy()

                p0 = eef_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reset_cam()

    def reset_task(self, idx):
        self._task_idx = idx
        self._traj_length = to_torch([0 for _ in range(self.num_envs)], device=self.device, dtype=torch.float)
        self.reset()

    def switch_arm_control_type(self, type, impedance=False):
        self.arm_control_type = type
        self.set_arm_control_param(impedance)
        robot_dof_props = self.get_robot_dof_props(self.robot_asset)
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            robot_ptr = self.robots[i]
            self.gym.set_actor_dof_properties(env_ptr, robot_ptr, robot_dof_props)
        self.init_control()

    def set_arm_control_param(self, impedance):
        if impedance:
            # OSC PD
            self.kp = to_torch([self.cfg["env"]["robot"]["osc_gain"] / 5.] * 6, device=self.device)
            self.kd = 2 * torch.sqrt(self.kp)
            self.kp_null = to_torch([0.] * 7, device=self.device)
            self.kd_null = to_torch([self.cfg["env"]["robot"]["osc_null_damp"] / 5.] * 7, device=self.device)

            # Joint PD
            self.pd_gain = self.cfg["env"]["robot"]["joint_gain"] / 10.
            self.pd_damp = self.cfg["env"]["robot"]["joint_damp"] / 3.
        else:
            self.kp = to_torch([self.cfg["env"]["robot"]["osc_gain"]] * 6, device=self.device)
            self.kd = 2 * torch.sqrt(self.kp)
            self.kp_null = to_torch([0.] * 7, device=self.device)
            self.kd_null = to_torch([self.cfg["env"]["robot"]["osc_null_damp"]] * 7, device=self.device)

            # Joint PD
            self.pd_gain = self.cfg["env"]["robot"]["joint_gain"]
            self.pd_damp = self.cfg["env"]["robot"]["joint_damp"]

    """
    Control Arm
    """

    def compute_osc_torques(self, tpos, tquat, robot_base=False):
        # Assuming robot base ori is (0, 0, 0, 1) in world frame

        eef_pos = self.states['eef_pos'].clone()
        eef_quat = self.states['eef_quat'].clone()

        if robot_base:
            robot_base_pos = self._robot_base_state[..., :3]
            robot_base_quat = self._robot_base_state[..., 3:7]
            tquat, tpos = tf_combine(robot_base_quat, robot_base_pos, tquat, tpos)

        dpos = tpos - eef_pos
        dquat = quat_mul(tquat, quat_conjugate(eef_quat))
        daxis_angle = axis_angle_from_quat(dquat)
        dpose = torch.cat([dpos, daxis_angle], dim=-1)

        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.robot_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null
        u = tensor_clamp(u.squeeze(-1), -self.robot_effort_limits[:7].unsqueeze(0), self.robot_effort_limits[:7].unsqueeze(0))

        return u

    def pre_phy_step(self, command, robot_base=False):
        if self.arm_control_type == 'joint':
            self._arm_control[:, :] = command['joint_state']
        elif self.arm_control_type == 'osc':
            u_arm = self.compute_osc_torques(tpos=command['eef_pos'], tquat=command['eef_quat'], robot_base=robot_base)
            self._arm_control[:, :] = u_arm
        else:
            raise NotImplementedError

        u_gripper = command['gripper_state']
        self.gripper_step(u_gripper)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))

    def gripper_step(self, u_gripper):
        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        if u_gripper is None:
            if self.cfg["env"]["gripperControlType"] == 'position':
                u_fingers[:] = self.states["q"][:, -2:].clone()
            if self.cfg["env"]["gripperControlType"] == 'effort':
                # keep the hand open with effort control
                u_fingers[:] = torch.ones_like(self._gripper_control) * 5.
        elif self.cfg["env"]["gripperControlType"] == 'position':
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.robot_dof_upper_limits[-2].item(),
                                          self.robot_dof_lower_limits[-2].item())
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.robot_dof_upper_limits[-1].item(),
                                          self.robot_dof_lower_limits[-1].item())
        elif self.cfg["env"]["gripperControlType"] == 'velocity':
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, 0.1, -0.1)
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, 0.1, -0.1)
            u_fingers[torch.abs(u_gripper) < 1e-2] = 0.  # no movement when u_gripper = 0
        elif self.cfg["env"]["gripperControlType"] == 'effort':
            # tune the controller for stability
            r = self.cfg["env"]["robot"]["gripper_force_damp_ratio"]
            delta_q_0 = self.states["q"][:, -2] - self.robot_dof_upper_limits[-2]
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, 20., -100. - r * delta_q_0)
            delta_q_0 = self.states["q"][:, -1] - self.robot_dof_upper_limits[-1]
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, 20.,  -100. - r * delta_q_0)
        else:
            raise NotImplementedError

        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

    """
    Solution
    """

    def teleport_joint_state(self, state):
        pos = self.robot_default_dof_pos.unsqueeze(0).clone()
        pos = pos.repeat(self.num_envs, 1)
        pos[:] = state

        # Reset the internal obs accordingly
        self._q[:, :] = pos
        self._qd[:, :] = torch.zeros_like(self._qd)

        self._pos_control[:, :] = pos
        self._effort_control[:, :] = torch.zeros_like(pos)
        self._vel_control[:, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[:, 0].flatten().contiguous()
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

    def solve(self):
        # set goal obj color
        self.set_target_color()

        for _ in range(60):
            self.env_physics_step()
            self.post_phy_step()

        contact = self.get_robot_contacts()
        rgb, seg = self.get_camera_image(rgb=True, seg=False)

        # set to default color
        self.set_default_color()

        return image_to_video(rgb), None

    """
    Eval
    """

    def eval(self):
        curr_pos = self.states["obj_pos"].clone()
        curr_rot = self.states["obj_quat"].clone()
        com = self.obj_ref_point.clone()

        init_pos = self.task_actor_init_state[:, self._task_idx, 3:, :3].clone()
        init_rot = self.task_actor_init_state[:, self._task_idx, 3:, 3:7].clone()

        curr_obj_to_world = curr_pos.reshape(-1, 3) + quat_apply(curr_rot.reshape(-1, 4), com.reshape(-1, 3))
        init_obj_to_world = init_pos.reshape(-1, 3) + quat_apply(init_rot.reshape(-1, 4), com.reshape(-1, 3))
        curr_obj_to_world = curr_obj_to_world.reshape(self.num_envs, -1, 3)
        init_obj_to_world = init_obj_to_world.reshape(self.num_envs, -1, 3)

        mask = 1. - torch.eye(self.num_objs, device=self.device)[self.task_obj_index[:, self._task_idx]]

        # Task Obj in Free Space
        task_obj_to_world = curr_obj_to_world \
        [torch.arange(self.num_envs, device=self.device), self.task_obj_index[:, self._task_idx]]
        robot_to_world = self._robot_base_state.clone()[..., :3]
        task_obj_to_robot = task_obj_to_world - robot_to_world

        # Other Obj in Original Pos
        err = (curr_obj_to_world - init_obj_to_world).norm(dim=-1) * mask
        err = (~(err < self.cfg["eval"]["e_threshold"])).any(dim=-1)

        obj_dist = ((curr_obj_to_world - init_obj_to_world).norm(dim=-1) * (1. - mask)).sum(dim=-1)

        z_results = task_obj_to_robot[..., -1] >= self.cfg["eval"]["z_threshold"]
        x_results = task_obj_to_robot[..., 0] <= self.cfg["eval"]["x_threshold"]
        e_results = ~err

        task_repeat = (obj_dist <= self.cfg["eval"]["d_threshold"]) & e_results

        result = {
            'z_threshold': z_results.cpu().numpy(),
            'x_threshold': x_results.cpu().numpy(),
            'e_threshold': e_results.cpu().numpy(),
            'task_repeat': task_repeat.cpu().numpy(),
            'success': (z_results & x_results & e_results).cpu().numpy(),
            'label': self.get_task_label()
        }

        return result

    def exit(self):
        pass

    """
    Camera
    """

    def set_target_color(self):
        for i, env_ptr in enumerate(self.envs):
            self.gym.set_rigid_body_color(env_ptr,
                                          self.objects[i][self.task_obj_index[i][self._task_idx].cpu()],
                                          0, gymapi.MESH_VISUAL, self.task_obj_color)

    def set_default_color(self):
        # set to default color
        for i, env_ptr in enumerate(self.envs):
            self.gym.set_rigid_body_color(env_ptr,
                                          self.objects[i][self.task_obj_index[i][self._task_idx].cpu()],
                                          0, gymapi.MESH_VISUAL, self.default_obj_color)

    def get_camera_image(self, rgb=False, seg=False):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        render_rgb_obs_buf, render_seg_obs_buf = self.get_numpy_images(self.cameras, rgb=rgb, seg=seg)
        self.gym.end_access_image_tensors(self.sim)

        return render_rgb_obs_buf, render_seg_obs_buf

    def get_numpy_images(self, camera_handles, rgb=False, seg=False):
        rgb_obs_buf, rgb_seg_buf = [], []
        for cam_handles, env in zip(camera_handles, self.envs):

            if isinstance(cam_handles, list):
                cam_ob, cam_seg = [], []
                for cam_handle in cam_handles:

                    if rgb:
                        color_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                        color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                        cam_ob.append(color_image)
                    if seg:
                        seg_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                        seg_image = seg_image.reshape(seg_image.shape[0], -1, 1)
                        cam_seg.append(seg_image)

                if rgb:
                    rgb_obs_buf.append(cam_ob)
                if seg:
                    rgb_seg_buf.append(cam_seg)

            else:
                if rgb:
                    color_image = self.gym.get_camera_image(self.sim, env, cam_handles, gymapi.IMAGE_COLOR)
                    color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                    rgb_obs_buf.append([color_image])
                if seg:
                    seg_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    seg_image = seg_image.reshape(seg_image.shape[0], -1, 1)
                    rgb_seg_buf.append([seg_image])

        return rgb_obs_buf, rgb_seg_buf

    """
    API Function
    """

    def get_task_idx(self):
        return self._task_idx

    def get_task_label(self):
        labels = []
        for i in range(self.num_envs):
            labels.append(self.task_obj_label[i][self._task_idx])

        return labels


#####################################################################
###=========================jit functions=========================###
#####################################################################


def vec_state_to_pose(vec):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*vec[:3])
    pose.r = gymapi.Quat(*vec[3:7])

    return pose


def matrix_to_q_t(mtx):
    assert mtx.shape[1] == 4 and mtx.shape[2] == 4
    t = mtx[:, :3, 3]
    q = matrix_to_quaternion(mtx[:, :3, :3])
    q = torch.cat([q[..., 1:], q[..., :1]], dim=-1)
    return q, t


def matrix_to_pose(mtx, transform=None):
    if transform is not None:
        mtx = transform @ mtx

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*mtx[:3, 3])
    q = tra.quaternion_from_matrix(mtx[:3, :3])
    pose.r = gymapi.Quat(*[*q[1:], q[0]])
    return pose


def apply_transform(xyz, T):
    if isinstance(xyz, list):
        xyz = np.array([*xyz, 1])
    elif isinstance(xyz, np.ndarray):
        xyz = np.concatenate([xyz, [1]])

    xyz = xyz.reshape(-1, 1)
    transform = (T @ xyz).reshape(-1)[:-1]

    return transform


@torch.jit.script
def axis_angle_from_quat(quat, eps=1.0e-6):
    # type: (Tensor, float) -> Tensor
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    mag = torch.linalg.norm(quat[:, 0:3], dim=1)
    com = quat[:, 3]
    rev = com > 0.
    half_angle = torch.atan2(mag, torch.abs(com))
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = (
        torch.where(torch.abs(angle) > eps, torch.sin(half_angle) / angle, 1 / 2 - angle ** 2.0 / 48))
    axis_angle = quat[:, 0:3] / sin_half_angle_over_angle.unsqueeze(-1) * (rev.float() * 2. - 1.).unsqueeze(-1)

    return axis_angle


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat
