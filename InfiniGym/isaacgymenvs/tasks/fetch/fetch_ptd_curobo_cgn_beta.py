
import numpy as np
import os
import torch
import time
import trimesh
import trimesh.transformations as tr
import os

from curobo.types.math import Pose

from curobo.types.robot import JointState, RobotConfig

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd_curobo import FetchPtdCurobo
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video, create_gripper_marker, plot_trajs

from isaacgymenvs.tasks.fetch.utils.contact_graspnet_utils import ContactGraspNet, CGN_PATH


class FetchPtdCuroboCGNBeta(FetchPtdCurobo):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.grasp_net = ContactGraspNet(
            root_dir=CGN_PATH,
            ckpt_dir=f'{CGN_PATH}/checkpoints/'
                     f'{self.cfg["solution"]["cgn"].get("ckpt_name", "contact_graspnet")}',
            forward_passes=self.cfg["solution"]["cgn"]["num_forward_passes"]
        )
        self.cgn_log_dir = f'./logs/cgn_log/{self.cfg["experiment_name"]}'
        if not os.path.exists(self.cgn_log_dir):
            os.makedirs(self.cgn_log_dir)

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

        self.update_ptd_motion_gen_world()

        start_time = time.time()
        cgn_result, cgn_logs = self.sample_goal_obj_collision_free_grasp_pose()
        if self.cgn_log_dir is not None:
            np.save(f'{self.cgn_log_dir}/log_{self.get_task_idx()}.npy', cgn_logs)

        traj, success, poses, results = \
            self.motion_gen_to_grasp_pose_ordered(cgn_result['pre_grasp_poses'], lsts=cgn_result['grasp_ordered_lst'])
        print("Pre Grasp Plan", success)
        log['pre_grasp_plan_success'] = success
        computing_time += time.time() - start_time

        self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
        print("Pre Grasp Phase End")
        log['pre_grasp_execute_error'] = self.get_end_effect_error(poses)

        if self.cfg["solution"]["move_offset_method"] == 'motion_planning':
            if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                self.enable_motion_gen_collider(enable_goal_obj=False)

            start_time = time.time()
            traj, success, poses, results = self.motion_gen_by_z_offset(z=self.cfg["solution"]["pre_grasp_offset"],
                                                                        mask=success)
            computing_time += time.time() - start_time

            if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                self.enable_motion_gen_collider(enable_goal_obj=True)

            print("Grasp Plan", success)
            log['grasp_plan_success'] = success

            log['grasp_execute_error'] = self.get_end_effect_error(poses)

            self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
        elif self.cfg["solution"]["move_offset_method"] == 'cartesian_linear':
            offset = np.array([0, 0, self.cfg["solution"]["pre_grasp_offset"] *
                                   self.cfg["solution"]["grasp_overshoot_ratio"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=0)
        else:
            raise NotImplementedError

        print("Grasp Phase End")

        self.close_gripper()
        log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Gripper Close End")

        if self.cfg["solution"]["retract_offset"] > 0:
            offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
            log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

        if self.cfg["solution"]["update_motion_gen_collider_before_fetch"]:
            self.update_ptd_motion_gen_world()

        self.update_ptd_motion_gen_config(attach_goal_obj=True)

        start_time = time.time()
        traj, success, poses, results = self.motion_gen_to_free_space(mask=success)
        computing_time += time.time() - start_time

        print("Fetch Plan", success)
        self.update_ptd_motion_gen_config(attach_goal_obj=False)
        log['fetch_plan_success'] = success

        fetch_failure = []
        for r in results:
            if r is None:
                fetch_failure.append(r)
            else:
                fetch_failure.append(r.status)
        log['fetch_plan_failure'] = fetch_failure

        self.follow_motion_trajs(traj, gripper_state=-1)
        print("Fetch Phase End")

        log['fetch_execute_error'] = self.get_end_effect_error(poses)

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log

    def grasp_pose_vis_debug(self, motion_gen, grasp_poses, pre_grasp_poses, pt, cam_pose, grasp_success):
        scene = trimesh.Scene()

        model = motion_gen.world_model
        collider = motion_gen.world_coll_checker

        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        for i, m in enumerate(model.mesh):
            mesh = m.get_trimesh_mesh()
            pose_tensor = collider.get_mesh_pose_tensor()
            pose_pos, pose_quat = pose_tensor[:, i, :3], pose_tensor[:, i, 3:7]

            pose = Pose(pose_pos.clone(), pose_quat.clone()).inverse().get_numpy_matrix()[0]
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)

        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(grasp_poses.position.shape[0]):
            if not grasp_success[i].cpu().numpy():
                continue

            trans = tr.translation_matrix(grasp_poses.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(grasp_poses.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        for i in range(pre_grasp_poses.position.shape[0]):
            if not grasp_success[i].cpu().numpy():
                continue

            trans = tr.translation_matrix(pre_grasp_poses.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(pre_grasp_poses.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            command_marker = create_gripper_marker([0, 255, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)

        pt = np.concatenate([pt, np.ones_like(pt[:, :1])], axis=-1)
        pt = (cam_pose.reshape(4, 4) @ pt.T).T[:, :3]

        cloud = trimesh.points.PointCloud(pt, colors=np.array([[0, 0, 200, 100]]).repeat(pt.shape[0], 0))
        scene.add_geometry(cloud)

        scene.show()


