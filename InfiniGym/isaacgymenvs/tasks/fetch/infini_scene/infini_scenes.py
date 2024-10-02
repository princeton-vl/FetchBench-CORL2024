
import numpy as np
import os
import torch
import imageio
import trimesh.transformations as tra

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.vec_task import VecTask
from isaacgymenvs.tasks.fetch.trimesh_scene import TrimeshRearrangeScene
from isaacgymenvs.tasks.fetch.utils.load_utils import (sample_random_scene,
                                                 get_franka_panda_asset,
                                                 sample_random_objects,
                                                 sample_random_combos,
                                                 InfiniSceneLoader)


class InfiniScene(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.scene_config_path = f'{self.cfg["sceneConfigPath"]}'
        self.loader = InfiniSceneLoader(self.scene_config_path)

        self.cfg["env"]["numObservations"] = 0  # Not RL Envs
        self.cfg["env"]["numActions"] = 0
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Env params
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # Env Asset
        self.robot_asset = None
        self.scene_asset = None
        self.object_asset = None
        self.object_combo_asset = None

        # Values to be filled in at runtime
        self.states = {}                    # will be dict filled with relevant states to use for reward calculation
        self.robot_handles = {}             # will be dict mapping names to relevant sim handles
        self.num_dofs = None                # Total number of DOFs per env
        self.num_objs = None

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None              # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None       # State of all joints       (n_envs, n_dof)
        self._contact_force_state = None    # Contact of all rigid bodies

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
        self._obj_init_state = None
        self._obj_init_label = None
        self._obj_contact_force = None

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
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.robot_default_dof_pos = (to_torch([-0.31267092, -1.1996635, 0.05781832, -2.1767514,
                      0.06494738, 0.9786396, 0.53001183, 0.04, 0.04], device=self.device))

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def sample_random_asset(self):
        asset_config = {
            'scene_category': self.cfg["env"].get("sceneCategory", None),
            'scene_idx': self.cfg["env"].get("sceneIdx", None),
            'num_objects': self.cfg["env"].get("numObjs", None),
            'num_combos': self.cfg["env"].get("numCombos", None),
            'combo_category': self.cfg["env"].get("comboCategory", None)
        }
        # Each Scene contains 15 assets
        assert asset_config['num_objects'] + asset_config['num_combos'] * 2 == 15

        mode = 'ws' if not self.cfg['benchmark'] else 'benchmark'

        self.scene_asset = sample_random_scene(asset_config['scene_category'], asset_config['scene_idx'], mode=mode)
        self.object_asset = sample_random_objects(asset_config['num_objects'], eval_only=self.cfg['use_eval'], mode=mode)
        self.object_combo_asset = sample_random_combos(asset_config['num_combos'], asset_config['combo_category'], mode=mode)

    def load_robot_asset(self):
        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0   # default = 0.02
        asset_options.density = 1000.0  # default = 1000.0
        asset_options.armature = self.cfg["env"]["robot"]["armature"]  # default = 0.0
        asset_options.enable_gyroscopic_forces = True
        asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        if self.cfg["env"]["robot"]["add_damping"]:
            asset_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
            asset_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
            asset_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
            asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            asset_options.linear_damping = 0.0                       # default = 0.0
            asset_options.max_linear_velocity = 1.0                  # default = 1000.0
            asset_options.angular_damping = 0.5                      # default = 0.5
            asset_options.max_angular_velocity = 2 * np.pi           # default = 64.0

        robot_asset_path = get_franka_panda_asset()
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

        # save configs
        self.loader.robot_asset_config.update({
            'name': 'robot_000',
            'asset_root':  robot_asset_path['asset_root'],
            'urdf_file': robot_asset_path['urdf_file'],
            'density': 1000.0,
            'armature': self.cfg["env"]["robot"]["armature"],
            'friction': self.cfg["env"]["robot"]["friction"],
            'restitution': self.cfg["env"]["robot"]["restitution"],
            'rolling_friction': 0,
            'torsion_friction': 0,
            'contact_offset': self.cfg["env"]["robot"]["contact_offset"],
            'rest_offset': 0.0
        })

        return robot_asset

    def load_scene_asset(self, scene):
        # load scene asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        s = self.gym.load_asset(self.sim, scene['asset_root'],
                                scene['urdf_file'], asset_options)

        s_props = self.gym.get_asset_rigid_shape_properties(s)
        for p in s_props:
            p.friction = self.cfg["env"]["scene"]["friction"]
            p.restitution = self.cfg["env"]["scene"]["restitution"]
            p.rolling_friction = 0.0  # default = 0.0
            p.torsion_friction = 0.0  # default = 0.0
            p.compliance = 0.0  # default = 0.0
            p.thickness = 0.0  # default = 0.0
            p.rest_offset = 0.0
            p.contact_offset = self.cfg["env"]["scene"]["contact_offset"]
        self.gym.set_asset_rigid_shape_properties(s, s_props)
        scene['asset'] = s

        self.loader.scene_asset_config.update({
            'name': scene['name'],
            'asset_root': scene['asset_root'],
            'urdf_file': scene['urdf_file'],
            'density': 1000.0,
            'friction': self.cfg["env"]["scene"]["friction"],
            'rolling_friction': 0,
            'torsion_friction': 0,
            'restitution': self.cfg["env"]["scene"]["restitution"],
            'contact_offset': self.cfg["env"]["scene"]["contact_offset"],
            'rest_offset': 0.0
        })

        return scene

    def load_object_asset(self, objects):

        for obj in objects:
            if self.cfg["env"]["objects"]["randomize_config"]:
                density = np.random.uniform(0.1, 0.4) * 1e3
                friction = np.random.uniform(0.6, 1.2)
                restitution = np.random.uniform(0.0, 0.1)
            else:
                density = self.cfg["env"]["objects"]["density"]
                friction = self.cfg["env"]["objects"]["friction"]
                restitution = self.cfg["env"]["objects"]["restitution"]

            asset_options = gymapi.AssetOptions()
            asset_options.thickness = 0.0
            asset_options.fix_base_link = False
            asset_options.collapse_fixed_joints = True
            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.use_mesh_materials = True
            asset_options.override_inertia = True
            asset_options.override_com = True
            if self.cfg["env"]["objects"]["add_damping"]:
                asset_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
                asset_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
                asset_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
                asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
            else:
                asset_options.linear_damping = 0.0  # default = 0.0
                asset_options.max_linear_velocity = 1.0  # default = 1000.0
                asset_options.angular_damping = 0.5  # default = 0.5
                asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0

            o = self.gym.load_asset(self.sim, obj['asset_root'], obj['urdf_file'], asset_options)
            o_props = self.gym.get_asset_rigid_shape_properties(o)
            for p in o_props:
                p.friction = friction
                p.restitution = restitution
                p.rolling_friction = self.cfg["env"]["objects"]["rolling_friction"]
                p.torsion_friction = self.cfg["env"]["objects"]["torsion_friction"]
                p.compliance = 0.0  # default = 0.0
                p.thickness = 0.0  # default = 0.0
                p.rest_offset = 0.0
                p.contact_offset = self.cfg["env"]["objects"]["contact_offset"]

            self.gym.set_asset_rigid_shape_properties(o, o_props)
            obj['asset'] = o

            self.loader.object_asset_config.append({
                'name': obj['name'],
                'asset_root': obj['asset_root'],
                'urdf_file': obj['urdf_file'],
                'density': density,
                'friction': friction,
                'restitution': restitution,
                'contact_offset': self.cfg["env"]["objects"]["contact_offset"],
                'rolling_friction': self.cfg["env"]["objects"]["rolling_friction"],
                'torsion_friction': self.cfg["env"]["objects"]["torsion_friction"],
                'rest_offset': 0.0
            })

        return objects

    def load_object_combo_asset(self, combos):

        for combo in combos:
            assets = []
            if self.cfg["env"]["objects"]["randomize_config"]:
                density = np.random.uniform(0.25, 0.4) * 1e3
                friction = np.random.uniform(0.6, 0.9)
                restitution = np.random.uniform(0.0, 0.05)
            else:
                density = self.cfg["env"]["objects"]["density"]
                friction = self.cfg["env"]["objects"]["friction"]
                restitution = self.cfg["env"]["objects"]["restitution"]

            for i, obj in enumerate(combo['urdf_file']):
                asset_options = gymapi.AssetOptions()
                asset_options.thickness = 0.0
                asset_options.fix_base_link = combo['fixed_base'][i]
                asset_options.collapse_fixed_joints = True
                asset_options.density = density
                asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                asset_options.use_mesh_materials = True
                asset_options.override_inertia = True
                asset_options.override_com = True
                if self.cfg["env"]["objects"]["add_damping"]:
                    asset_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
                    asset_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
                    asset_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
                    asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0; reduced to prevent CUDA errors
                else:
                    asset_options.linear_damping = 0.0  # default = 0.0
                    asset_options.max_linear_velocity = 1.0  # default = 1000.0
                    asset_options.angular_damping = 0.5  # default = 0.5
                    asset_options.max_angular_velocity = 2 * np.pi  # default = 64.0

                o = self.gym.load_asset(self.sim, combo['asset_root'], obj, asset_options)
                o_props = self.gym.get_asset_rigid_shape_properties(o)
                for p in o_props:
                    p.friction = friction
                    p.restitution = restitution
                    p.rolling_friction = self.cfg["env"]["objects"]["rolling_friction"]  # default = 0.0
                    p.torsion_friction = self.cfg["env"]["objects"]["torsion_friction"]  # default = 0.0
                    p.compliance = 0.0  # default = 0.0
                    p.thickness = 0.0  # default = 0.0
                    p.rest_offset = 0.0
                    p.contact_offset = self.cfg["env"]["objects"]["contact_offset"]

                self.gym.set_asset_rigid_shape_properties(o, o_props)
                assets.append(o)
            combo['asset'] = assets

            self.loader.combo_asset_config.append({
                'name': combo['name'],
                'asset_root': combo['asset_root'],
                'urdf_file': combo['urdf_file'],
                'density': density,
                'friction': friction,
                'restitution': restitution,
                'contact_offset': self.cfg["env"]["objects"]["contact_offset"],
                'rest_offset': 0.0
            })

        return combos

    def load_table_asset(self, dim):
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.0
        asset_options.fix_base_link = True
        asset_options.thickness = 0.0
        asset_options.disable_gravity = True
        table_dims = gymapi.Vec3(*dim)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y,
                                          table_dims.z, asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = self.cfg["env"]["robot"]["friction"]
            p.restitution = self.cfg["env"]["robot"]["restitution"]
            p.rolling_friction = 0.0
            p.torsion_friction = 0.0
            p.compliance = 0.0
            p.thickness = 0.0
            p.contact_offset = self.cfg["env"]["robot"]["contact_offset"]
            p.rest_offset = 0.0
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)

        self.loader.robot_asset_config.update({
            'table_dim': dim
        })
        return table_asset

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        super()._create_gym_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.sample_random_asset()
        robot_asset = self.load_robot_asset()
        scene_assets = self.load_scene_asset(self.scene_asset)
        # load object asset
        object_assets = self.load_object_asset(self.object_asset)
        object_combo_assets = self.load_object_combo_asset(self.object_combo_asset)

        self.num_objs = len(object_combo_assets) * 2 + len(object_assets)

        self.robot_asset = robot_asset
        self.scene_asset = scene_assets
        self.object_asset = object_assets
        self.object_combo_asset = object_combo_assets

        trimesh_scene = TrimeshRearrangeScene(scene_assets['meshes'],
                                              scene_assets['support'],
                                              scene_assets['robot_cam_config'],
                                              spacing=spacing,
                                              **self.cfg["env"]["scene"])
        table_dim = trimesh_scene.table_dim
        table_asset = self.load_table_asset(table_dim)

        scene_asset = scene_assets['asset']
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_scene_bodies = self.gym.get_asset_rigid_body_count(scene_asset)
        self.num_scene_dofs = self.gym.get_asset_dof_count(scene_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)
        print("num scene bodies: ", self.num_scene_bodies)
        print("num scene dofs: ", self.num_scene_dofs)

        robot_dof_props = self.get_robot_dof_props(robot_asset)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(scene_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(scene_asset)

        num_objects_bodies = 0
        num_objects_shapes = 0
        for c in object_combo_assets:
            num_objects_bodies += self.gym.get_asset_rigid_body_count(c['asset'][0])
            num_objects_shapes += self.gym.get_asset_rigid_shape_count(c['asset'][0])
            num_objects_bodies += self.gym.get_asset_rigid_body_count(c['asset'][1])
            num_objects_shapes += self.gym.get_asset_rigid_shape_count(c['asset'][1])
        for o in object_assets:
            num_objects_bodies += self.gym.get_asset_rigid_body_count(o['asset'])
            num_objects_shapes += self.gym.get_asset_rigid_shape_count(o['asset'])

        max_agg_bodies = num_franka_bodies + num_cabinet_bodies + num_objects_bodies + 1
        max_agg_shapes = num_franka_shapes + num_cabinet_shapes + num_objects_shapes + 1

        self.cam_params = self.get_camera_params()

        # handle holders
        self.robots = []
        self.scenes = []
        self.objects = []
        self.tables = []
        self.envs = []
        self.cameras = []
        self._cam_config = []
        self._obj_com = []
        self._obj_init_state = []
        self._obj_init_label = []

        for i in range(self.num_envs):
            seg_idx = 0
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            # NOTE: franka should ALWAYS be loaded first in sim!

            gym_transform = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            num_discarded_objs = len(object_assets) - self.cfg["env"]["numSceneObjs"]
            num_discarded_combos = len(object_combo_assets) - self.cfg["env"]["numSceneCombos"]

            n_combos, n_objects = trimesh_scene.random_arrangement(object_assets, object_combo_assets,
                                                                   num_obj_discarded=num_discarded_objs,
                                                                   num_combo_discarded=num_discarded_combos)

            # towards +x
            robot_base_pos, robot_table_pos = trimesh_scene.sample_robot_base()
            robot_start_pose = gymapi.Transform()
            robot_start_pose.p = gymapi.Vec3(*apply_transform(robot_base_pos, gym_transform))
            robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            table_start_pose = gymapi.Transform()
            table_start_pose.p = gymapi.Vec3(*apply_transform(robot_table_pos, gym_transform))
            table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            scene_start_pose = gymapi.Transform()
            scene_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            quat = quat_mul(to_torch([0.0, 0.0, 1.0, 0.0], device='cpu'),
                            to_torch([0.707, 0.0, 0.0, 0.707], device='cpu'))
            scene_start_pose.r = gymapi.Quat(*quat.numpy())

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, seg_idx)
            seg_idx += 1
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, seg_idx)
            seg_idx += 1
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.0))

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            scene_actor = self.gym.create_actor(env_ptr, scene_asset, scene_start_pose, "scene", i, 0, seg_idx)
            seg_idx += 1

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            object_actors, object_init_states, object_init_labels, object_coms = [], [], [], []
            for k, cbo in enumerate(n_combos):
                assert f'obj_combo_{k}' == cbo['name']  # make sure the order is the same
                combo_transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                            [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
                combo_start_pose = matrix_to_pose(cbo['placement_pose'], transform=gym_transform,
                                                  pre_transform=combo_transform)
                organizer_actor = self.gym.create_actor(env_ptr, cbo['asset'][0], combo_start_pose,
                                                        f"obj_combo_{k}_organizer", i, 0, seg_idx)
                seg_idx += 1

                object_actors.append(organizer_actor)
                object_coms.append(apply_transform(cbo['metadata']['organizer_com'], combo_transform.T))

                object_init_states.append([combo_start_pose.p.x, combo_start_pose.p.y,
                                           combo_start_pose.p.z, combo_start_pose.r.x,
                                           combo_start_pose.r.y, combo_start_pose.r.z,
                                           combo_start_pose.r.w, 0, 0, 0, 0, 0, 0])
                object_actor = self.gym.create_actor(env_ptr, cbo['asset'][1], combo_start_pose,
                                                     f"obj_combo_{k}_object", i, 0, seg_idx)
                object_init_labels.append(f'combo_org_{cbo["combo_type"]}_{cbo["placement_label"]}')

                seg_idx += 1
                object_actors.append(object_actor)
                object_coms.append(apply_transform(cbo['metadata']['object_com'], combo_transform.T))
                object_init_states.append([combo_start_pose.p.x, combo_start_pose.p.y,
                                           combo_start_pose.p.z, combo_start_pose.r.x,
                                           combo_start_pose.r.y, combo_start_pose.r.z,
                                           combo_start_pose.r.w, 0, 0, 0, 0, 0, 0])
                object_init_labels.append(f'combo_obj_{cbo["combo_type"]}_{cbo["placement_label"]}')

            for k, o in enumerate(n_objects):
                assert f'obj_{k}' == o['name']  # make sure the order is the same

                obj_start_pose = matrix_to_pose(o['placement_pose'], transform=gym_transform)
                object_actor = self.gym.create_actor(env_ptr, o['asset'], obj_start_pose, f"obj_{k}", i, 0, seg_idx)
                seg_idx += 1
                object_actors.append(object_actor)
                object_coms.append(o['metadata']['com'])

                object_init_states.append([obj_start_pose.p.x, obj_start_pose.p.y,
                                           obj_start_pose.p.z, obj_start_pose.r.x,
                                           obj_start_pose.r.y, obj_start_pose.r.z,
                                           obj_start_pose.r.w, 0, 0, 0, 0, 0, 0])
                object_init_labels.append(f'rigid_obj_{o["placement_label"]}')

            # add cams
            cams, cam_configs = [], []
            for i in range(self.cfg["env"]["cam"]["num_cam"]):
                cam_config = trimesh_scene.sample_camera_pose(i=i)
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.cam_params)
                self.gym.set_camera_location(camera_handle, env_ptr,
                                             gymapi.Vec3(*apply_transform(cam_config['pos'], gym_transform)),
                                             gymapi.Vec3(*apply_transform(cam_config['focus'], gym_transform)))
                cams.append(camera_handle)
                cam_configs.append(np.concatenate([apply_transform(cam_config['pos'], gym_transform),
                                                   apply_transform(cam_config['focus'], gym_transform)]))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.scenes.append(scene_actor)
            self.objects.append(object_actors)
            self.tables.append(table_actor)
            self.cameras.append(cams)
            self._cam_config.append(cam_configs)
            self._obj_com.append(object_coms)
            self._obj_init_state.append(object_init_states)
            self._obj_init_label.append(object_init_labels)

        self.init_torch_data()

    def get_robot_dof_props(self, asset):

        robot_dof_props = self.gym.get_asset_dof_properties(asset)
        if self.cfg["env"]["armControlType"] == 'joint':
            robot_dof_stiffness = to_torch([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([50, 50, 50, 50, 50, 50, 50], dtype=torch.float, device=self.device)

            for i in range(self.num_robot_dofs - 2):
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]

        elif self.cfg["env"]["armControlType"] == 'osc':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

            for i in range(self.num_robot_dofs - 2):
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]
        else:
            raise NotImplementedError

        if self.cfg["env"]["gripperControlType"] == 'position':
            robot_dof_stiffness = to_torch([1e4, 1e4], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([4e2, 4e2], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_POS
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]
                robot_dof_props['effort'][i + self.num_robot_dofs - 2] = 400

        elif self.cfg["env"]["gripperControlType"] == 'velocity':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([7e2, 7e2], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_VEL
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]

        elif self.cfg["env"]["gripperControlType"] == 'effort':
            # set robot dof properties
            robot_dof_stiffness = to_torch([0, 0], dtype=torch.float, device=self.device)
            robot_dof_damping = to_torch([0, 0], dtype=torch.float, device=self.device)

            for i in range(2):
                robot_dof_props['driveMode'][i + self.num_robot_dofs - 2] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i + self.num_robot_dofs - 2] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i + self.num_robot_dofs - 2] = robot_dof_damping[i]
        else:
            raise NotImplementedError

        # upscale gripper effort
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

    def get_camera_params(self):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.cfg["env"]["cam"]["hov"]
        camera_props.width = self.cfg["env"]["cam"]["width"]
        camera_props.height = self.cfg["env"]["cam"]["height"]
        camera_props.enable_tensors = True

        self.loader.camera_config.update({
            'hov': camera_props.horizontal_fov,
            'width': camera_props.width,
            'height': camera_props.height
        })
        return camera_props

    def get_numpy_rgb_images(self, camera_handles):
        rgb_obs_buf = []
        for cam_handles, env in zip(camera_handles, self.envs):
            cam_ob = []
            if isinstance(cam_handles, list):
                for cam_handle in cam_handles:
                    color_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                    cam_ob.append(color_image)
                rgb_obs_buf.append(cam_ob)
            else:
                color_image = self.gym.get_camera_image(self.sim, env, cam_handles, gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                rgb_obs_buf.append([color_image])
        return rgb_obs_buf

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
        self._obj_init_state = (to_torch(self._obj_init_state, device=self.device, dtype=torch.float)
                                .view(self.num_envs, self.num_objs, 13))

        self.init_control()

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (3 + self.num_objs), dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def init_control(self):
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros_like(self._pos_control)
        self._effort_control = torch.zeros_like(self._pos_control)

        if self.cfg["env"]["armControlType"] == 'osc':
            self._arm_control = self._effort_control[:, :7]
        elif self.cfg["env"]["armControlType"] == 'joint':
            self._arm_control = self._pos_control[:, :7]
        else:
            raise NotImplementedError

        if self.cfg["env"]["gripperControlType"] == 'effort':
            self._gripper_control = self._effort_control[:, 7:9]
        elif self.cfg["env"]["gripperControlType"] == 'position':
            self._gripper_control = self._pos_control[:, 7:9]
        elif self.cfg["env"]["gripperControlType"] == 'velocity':
            self._gripper_control = self._vel_control[:, 7:9]
        else:
            raise NotImplementedError

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
        # Todo: Add init joint angle noise.
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

        multi_env_ids_objects_int32 = self._global_indices[env_ids, 3:].flatten()
        self._obj_state[env_ids, :] = self._obj_init_state[env_ids, :]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_objects_int32), len(multi_env_ids_objects_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0

        self.env_physics_step()
        self._refresh()

    def post_phy_step(self):
        self._refresh()

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            # Plot visualizations
            for i in range(self.num_envs):
                for k in range(self.num_objs):
                    pos, rot, com = self.states["obj_pos"][i][k], self.states["obj_quat"][i][k], self._obj_com[i][k]
                    com = to_torch(com, device=self.device)
                    p0 = (pos + quat_apply(rot, com)).cpu().numpy()

                    px = (pos + quat_apply(rot, com + to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos + quat_apply(rot, com + to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos + quat_apply(rot, com + to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
                    #self.gym.draw_env_rigid_contacts(self.viewer, self.envs[i], gymapi.Vec3(0.0, 0.0, 1.0), 5.0, False)

                px = (self.states["eef_pos"][i] + quat_apply(self.states["eef_quat"][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.states["eef_pos"][i] + quat_apply(self.states["eef_quat"][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.states["eef_pos"][i] + quat_apply(self.states["eef_quat"][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.states["eef_pos"][i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])

    def step(self, action=None):
        for _ in range(120):
            self.env_physics_step()
            self.post_phy_step()

        saved = self.save_env()
        self.log_camera_view_image(saved)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def check_env_status(self, idx):
        obj_curr_pos = self.states["obj_pos"][idx]
        obj_curr_quat = self.states["obj_quat"][idx]
        obj_com = to_torch(self._obj_com[idx], device=self.device)
        obj_init_pos = self._obj_init_state[..., :3][idx]
        obj_init_quat = self._obj_init_state[..., 3:7][idx]

        p_init = (obj_init_pos + quat_apply(obj_init_quat, obj_com)).cpu().numpy()
        p_curr = (obj_curr_pos + quat_apply(obj_curr_quat, obj_com)).cpu().numpy()

        delta_pos = np.linalg.norm(p_init - p_curr, axis=-1)

        obj_vel = self.states["obj_vel"][idx].cpu().numpy()
        delta_v = np.linalg.norm(obj_vel[..., :3], axis=-1)
        delta_w = np.linalg.norm(obj_vel[..., 3:], axis=-1)

        res = True
        for i, o in enumerate(self._obj_init_label[idx]):
            if 'on_floor' in o:
                res &= (delta_v[i] < 1e-1)
                continue

            res &= (delta_v[i] < 5e-2)
            res &= (delta_pos[i] < 5e-2)

        return res

    def log_camera_view_image(self, save):

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        render_rgb_obs_buf = self.get_numpy_rgb_images(self.cameras)
        self.gym.end_access_image_tensors(self.sim)

        for i, images in enumerate(render_rgb_obs_buf):
            if save[i]:
                for j, img in enumerate(images):
                    imageio.imwrite(f'{self.scene_config_path}/env_{i}_{j}.png', img)

    def save_env(self, max_envs=1):
        saved = []
        for i in range(self.num_envs):
            robot_state = self._robot_base_state[i].cpu().numpy()
            table_state = self._table_base_state[i].cpu().numpy()
            scene_state = self._scene_base_state[i].cpu().numpy()

            object_state = self._obj_state[i].cpu().numpy()
            obj_init_label = self._obj_init_label[i]

            res = self.check_env_status(i)
            saved.append(res)
            if not res:
                continue

            robot_full_state = np.concatenate([robot_state, table_state], axis=0)
            self.loader.append_pose(robot_full_state, cat='robot')
            self.loader.append_pose(scene_state, cat='scene')
            self.loader.append_pose(object_state, cat='object')

            self.loader.append_pose(np.stack(self._cam_config[i]), cat='camera')
            self.loader.object_labels.append(obj_init_label)

        # dump env configs
        self.loader.save_env_config()
        return saved


#####################################################################
###=========================jit functions=========================###
#####################################################################


def matrix_to_pose(mtx, transform=None, pre_transform=None):
    if transform is not None:
        mtx = transform @ mtx
    if pre_transform is not None:
        mtx = mtx @ pre_transform

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*mtx[:3, 3])
    q = tra.quaternion_from_matrix(mtx[:3, :3])
    pose.r = gymapi.Quat(*[*q[1:], q[0]])
    return pose


def apply_transform(xyz, T):
    if isinstance(xyz, list):
        xyz = np.array([*xyz, 1])
    elif isinstance(xyz, np.ndarray):
        xyz = np.concatenate([xyz, np.array([1.])])

    xyz = xyz.reshape(-1, 1)
    transform = (T @ xyz).reshape(-1)[:-1]

    return transform


@torch.jit.script
def axis_angle_from_quat(quat, eps=1.0e-6):
    # type: (Tensor, float) -> Tensor
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    mag = torch.linalg.norm(quat[:, 0:3], dim=1)
    half_angle = torch.atan2(mag, quat[:, 3])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = (
        torch.where(torch.abs(angle) > eps, torch.sin(half_angle) / angle, 1 / 2 - angle ** 2.0 / 48))
    axis_angle = quat[:, 0:3] / sin_half_angle_over_angle.unsqueeze(-1)

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
