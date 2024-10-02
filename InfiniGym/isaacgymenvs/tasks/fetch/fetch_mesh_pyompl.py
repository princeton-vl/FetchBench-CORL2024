
import numpy as np
import os
import torch
import trimesh.transformations as tr
import trimesh
import time

from isaacgymenvs.tasks.fetch.utils.pyompl_utils import PyBulletOMPL
from curobo.types.math import Pose

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import (to_torch, get_axis_params, tensor_clamp,
                                                tf_vector, tf_combine, quat_mul, quat_conjugate,
                                                quat_to_angle_axis, tf_inverse, quat_apply,
                                                matrix_to_quaternion)

from isaacgymenvs.tasks.fetch.fetch_solution_base import FetchSolutionBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video


class FetchMeshPyompl(FetchSolutionBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        assert self.num_envs == 1  # Buggy Pybullet, only one env at a time
        # Multiple Pybullet Server may cause error
        self.motion_generator = \
            [PyBulletOMPL(self.cfg["solution"]["pyompl"], self.debug_viz) for i in range(self.num_envs)]

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
            'scene': {'quat': sq.cpu(), 'pos': st.cpu()},
            'table': {'quat': dq.cpu(), 'pos': dt.cpu()},
            'object': {'quat': oq.cpu(), 'pos': ot.cpu()},
            'eef': {'quat': eq.cpu(), 'pos': et.cpu()}
        }

        return pose

    def _get_batch_joint_trajs(self, trajs):
        max_len = 0

        for t in trajs:
            if t is None or len(t) == 0:
                continue
            max_len = max(max_len, len(t))

        max_len = max_len + 10  # Stabilize the last state
        padded_trajs = []

        for i, tr in enumerate(trajs):
            if tr is None:
                halt_state = self.states["q"][i:i+1][..., :-2].clone()
                new_pos = torch.ones((max_len, *halt_state.shape[1:]), device=halt_state.device, dtype=halt_state.dtype) * halt_state
                padded_trajs.append(new_pos)

            else:
                # cut-off zeros
                if isinstance(tr, list):
                    tr = to_torch(tr, device=self.device)

                zero_pos = torch.sum(torch.abs(tr), dim=-1) < 1e-3
                zero_pos = torch.argmax(zero_pos.float(), dim=-1)

                if zero_pos.cpu().numpy() == 0:
                    last_step = tr.shape[0] - 1
                else:
                    last_step = zero_pos - 1

                # position
                new_pos = torch.ones((max_len, *tr.shape[1:]), device=tr.device, dtype=tr.dtype) * tr[last_step]
                new_pos[:last_step] = tr[:last_step].clone()

                padded_trajs.append(new_pos)
        return torch.stack(padded_trajs, dim=0)

    """
    Pybullet State
    """

    def update_pyompl_world_collider_pose(self, attach_goal_obj=False):
        pose = self._get_pose_in_robot_frame()
        eef_pose, scene_pose, table_pose, object_pose = (pose['eef'], pose['scene'], pose['table'], pose['object'])

        for i in range(self.num_envs):
            # get scene info
            scene_info = {
                'eef_quat': eef_pose['quat'][i].numpy(),
                'eef_pos': eef_pose['pos'][i].numpy(),
                'table_quat': table_pose['quat'][i].numpy(),
                'table_pos': table_pose['pos'][i].numpy(),
                'table_dim': np.array(self.table_asset[i]['dim']),
                'scene_quat': scene_pose['quat'][i].numpy(),
                'scene_pos': scene_pose['pos'][i].numpy(),
                'scene_asset': self.scene_asset[i],
                'object_quat': object_pose['quat'][i].numpy(),
                'object_pos': object_pose['pos'][i].numpy(),
                'object_asset': self.object_asset[i]
            }
            self.motion_generator[i].update_scene(scene_info,
                                                  goal_obj_idx= \
                                                  int(self.task_obj_index[i][self.get_task_idx()].cpu().numpy()),
                                                  attach_goal_obj=attach_goal_obj)

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
            grasp_pose = self.obj_grasp_poses[i][goal_idx].cpu()
            random_batch = torch.randint(grasp_pose.shape[0], size=(max_pose_seed,))

            sample_quat, sample_pos = grasp_pose[random_batch][..., 3:7], grasp_pose[random_batch][..., :3]
            oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(max_pose_seed, 1),
                          ot[i:i+1, goal_idx].repeat(max_pose_seed, 1))
            gq, gt = tf_combine(oq_i, ot_i, sample_quat, sample_pos)
            sample_grasps.append(torch.concat([gt, gq], dim=-1))

        sample_grasps = torch.stack(sample_grasps, dim=0).numpy()
        return sample_grasps

    def sample_goal_obj_collision_free_grasp_pose(self):
        # Use IK solver to solve for candidate grasp pose
        annotated_grasp_pose = self._sample_goal_obj_annotated_grasp_pose()
        grasp_poses, grasp_success, pre_grasp_poses = [], [], []

        for i in range(self.num_envs):
            grasp_candidate = annotated_grasp_pose[i, :]
            Ts, success = self.motion_generator[i].check_collision_free_grasps(grasp_candidate)
            grasp_poses.append(Ts)
            grasp_success.append(success)

            offset_matrix = tr.translation_matrix([0, 0, -self.cfg["solution"]["pre_grasp_offset"]])
            pre_grasp_poses.append(Ts @ np.expand_dims(offset_matrix, axis=0))

        res = {
            'grasp_poses': grasp_poses,
            'grasp_success': grasp_success,
            'pre_grasp_poses': pre_grasp_poses,
        }

        if self.debug_viz and self.viewer is not None:
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                success = grasp_success[i].nonzero()[0]
                self.grasp_vis_debug(pose, grasp_poses[i][success], pre_grasp_poses[i][success], env_idx=i)

        return res

    """
    Motion Generation
    """

    def motion_gen_to_pose_goalset(self, target_poses):
        # motion generation to grasp the object
        self._refresh()

        trajs, poses, success = [], [], []
        for i in range(self.num_envs):
            q_start = self.states["q"][i, :-2].cpu().numpy()

            # get success mask
            if target_poses[i] is None:
                trajs.append(None)
                success.append(False)
                poses.append(None)
                continue

            result, traj, goal = self.motion_generator[i].plan_goalset(q_start, target_poses[i], ret_goal=True)
            trajs.append(traj)
            poses.append(goal)
            success.append(result)

        if self.debug_viz and self.viewer is not None:
            scene_pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                traj = trajs[i]

                if traj is None:
                    continue
                self.motion_vis_debug(scene_pose, self.motion_generator[i], traj, target_poses[i], env_idx=i)

        return self._get_batch_joint_trajs(trajs), success, poses

    def motion_gen_to_grasp_pose(self, poses, mask):
        target_poses = []
        for i in range(self.num_envs):
            m = mask[i].nonzero()[0]
            if len(m) == 0:
                target_poses.append(None)
            else:
                m = m.reshape(-1)
                target_poses.append(poses[i][m])

        return self.motion_gen_to_pose_goalset(target_poses)

    def motion_gen_by_z_offset(self, z, mask):
        eef = self._get_pose_in_robot_frame()['eef']
        eq = torch.cat([eef['quat'][:, -1:], eef['quat'][:, :-1]], dim=-1)

        target_poses = []
        for i, m in enumerate(mask):
            if m:
                trans = tr.translation_matrix(eef['pos'][i].numpy())
                rot = tr.quaternion_matrix(eq[i].numpy())
                offset = tr.translation_matrix([0, 0, z])
                pose = trans @ rot @ offset
                target_poses.append(np.expand_dims(pose, axis=0))
            else:
                target_poses.append(None)

        return self.motion_gen_to_pose_goalset(target_poses)

    def motion_gen_to_free_space(self, mask):

        target_pos = [[-0.2, -0.25, 0.66], [-0.2, 0.25, 0.66]]
        target_quat = [[0, 0.707, -0.707, 0], [0, 0.707, 0.707, 0]]

        end_poses = []
        for t in range(len(target_quat)):
            pos = target_pos[t]
            quat = target_quat[t]
            translation = tr.translation_matrix(pos)
            rotation = tr.quaternion_matrix(quat)
            end_poses.append(translation @ rotation)
        end_poses = np.stack(end_poses, axis=0)

        target_poses = []
        for i, m in enumerate(mask):
            if m:
                target_poses.append(end_poses)
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
            traj_command = {"joint_state": trajs[:, step].clone().to(self.device)}
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

            executed_pos = torch.stack(executed_pos, dim=1)
            executed_vel = torch.stack(executed_vel, dim=1)
            for i in range(self.num_envs):
                traj = trajs[i]

                # plot and save the joint error
                plot_trajs([
                    [traj.cpu().numpy(), torch.zeros_like(traj).cpu().numpy()],
                    [executed_pos[i].cpu().numpy(), executed_vel[i].cpu().numpy()]
                ], 0.015)

    """
    Get Control Error
    """

    def get_end_effect_error(self, target_poses):
        # assume the target poses are in robot frame
        scene_info = self._get_pose_in_robot_frame()['eef']
        err_pose = []

        for i in range(self.num_envs):
            eq, et = scene_info['quat'][i], scene_info['pos'][i]
            translation_matrix = tr.translation_matrix(et)
            eq = np.concatenate([eq[-1:], eq[:-1]], axis=-1)
            rotation_matrix = tr.quaternion_matrix(eq)
            eef_pose = translation_matrix @ rotation_matrix

            t_pose = target_poses[i]

            if t_pose is None:
                err_pose.append({'pos_err': 10.0, 'rot_err': 2 * np.pi})
                continue

            delta_pose = eef_pose @ tr.inverse_matrix(t_pose)
            err_pos = np.linalg.norm(delta_pose[:3, 3])
            err_rot = np.arccos((np.trace(delta_pose[:3, :3]) - 1) / 2.)
            err_pose.append({'pos_err': err_pos, 'rot_err': err_rot})

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
        self.update_pyompl_world_collider_pose(attach_goal_obj=False)
        grasp_result = self.sample_goal_obj_collision_free_grasp_pose()

        if self.cfg["solution"]["direct_grasp"]:

            start_time = time.time()
            traj, success, poses = (
                self.motion_gen_to_grasp_pose(grasp_result['grasp_poses'], mask=grasp_result['grasp_success']))
            print("Grasp Plan", success)
            log['grasp_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
            print("Grasp Phase End")
            log['grasp_execute_error'] = self.get_end_effect_error(poses)
        else:
            start_time = time.time()
            traj, success, poses = (
                self.motion_gen_to_grasp_pose(grasp_result['pre_grasp_poses'], mask=grasp_result['grasp_success']))
            print("Pre Grasp Plan", success)
            log['pre_grasp_plan_success'] = success
            computing_time += time.time() - start_time

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
        self.update_pyompl_world_collider_pose(attach_goal_obj=self.cfg["solution"]["attach_goal_obj"])

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

    def scene_vis_debug(self, poses, env_idx=0):
        scene = trimesh.Scene()
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        table_pose = poses['table']
        dq = np.concatenate([table_pose['quat'][..., -1:], table_pose['quat'][..., :-1]], axis=-1)
        dt = table_pose['pos']

        table_translation = tr.translation_matrix(dt[env_idx])
        table_rotation = tr.quaternion_matrix(dq[env_idx])

        table = trimesh.creation.box(extents=self.table_asset[env_idx]['dim'], transform=table_translation @ table_rotation)
        scene.add_geometry(table)

        scene_pose = poses['scene']

        sq = np.concatenate([scene_pose['quat'][..., -1:], scene_pose['quat'][..., :-1]], axis=-1)
        st = scene_pose['pos']

        # vis environment 0
        scene_translation = tr.translation_matrix(st[env_idx])
        scene_rotation = tr.quaternion_matrix(sq[env_idx])

        # vis scene
        for m in self.scene_asset[env_idx]['meshes']:
            mesh = m.copy().apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        object_poses = poses['object']
        oq = np.concatenate([object_poses['quat'][..., -1:], object_poses['quat'][..., :-1]], axis=-1)
        ot = object_poses['pos']

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
            trans = tr.translation_matrix(ot[env_idx][i])
            rot = tr.quaternion_matrix(oq[env_idx][i])
            mesh = o['mesh'].copy().apply_transform(trans @ rot)
            scene.add_geometry(mesh)

        return scene

    def grasp_vis_debug(self, poses, grasp_pose, pre_grasp_pose, env_idx=0):

        scene = self.scene_vis_debug(poses, env_idx)
        # grasp pose
        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(grasp_pose.shape[0]):
            grasp = grasp_pose[i] @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        # pre grasp pose
        for i in range(pre_grasp_pose.shape[0]):
            pre_grasp = pre_grasp_pose[i] @ vis_rot
            command_marker = create_gripper_marker([0, 0, 255]).apply_transform(pre_grasp)
            scene.add_geometry(command_marker)

        scene.show()

    def motion_vis_debug(self, poses, motion_gen, traj, target_poses, env_idx=0):
        scene = self.scene_vis_debug(poses, env_idx)

        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(len(target_poses)):
            grasp = target_poses[i] @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        for i in range(len(traj)):
            pose = motion_gen.trac_ik_interface.fk(np.array(traj[i]))
            grasp = pose @ vis_rot
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

