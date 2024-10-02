
import numpy as np
import time

from curobo.types.math import Pose

from curobo.types.robot import JointState, RobotConfig


from isaacgymenvs.tasks.fetch.fetch_mesh_curobo_cgn_beta import FetchMeshCuroboPtdCGNBeta
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video, create_gripper_marker, plot_trajs


class FetchMeshCuroboPtdCGNBetaRep(FetchMeshCuroboPtdCGNBeta):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        assert self.num_envs == 1

        self.num_max_trials = self.cfg["solution"]["num_max_trials"]

        state = self.motion_generators[0].compute_kinematics(
            JointState.from_position(self.robot_default_dof_pos.to(self.tensor_args.device).view(1, -1)[:, :-2])
        )

        self.retract_pose = Pose(position=state.ee_pos_seq, quaternion=state.ee_quat_seq).clone()

    """
    Motion Gen
    """

    def motion_plan_to_retract_pose(self):
        target_poses = []
        for i in range(self.num_envs):
            target_poses.append(self.retract_pose.clone().unsqueeze(dim=0))

        return self.motion_gen_to_pose_goalset(target_poses)

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

        for k in range(self.num_max_trials):
            self.update_cuRobo_world_collider_pose()

            st = time.time()
            cgn_result, cgn_logs = self.sample_goal_obj_collision_free_grasp_pose()
            computing_time += st - time.time()
            if self.cgn_log_dir is not None:
                np.save(f'{self.cgn_log_dir}/log_{self.get_task_idx()}.npy', cgn_logs)

            if self.cfg["solution"]["direct_grasp"]:
                self.update_cuRobo_world_collider_pose()
                start_time = time.time()
                traj, success, poses, results = self.motion_gen_to_grasp_pose_ordered(cgn_result['grasp_poses'],
                                                                                      cgn_result['grasp_ordered_lst'])
                print("Grasp Plan", success)
                log['grasp_plan_success'] = success
                computing_time += time.time() - start_time

                self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
                log['grasp_execute_error'] = self.get_end_effect_error(poses)

            else:
                start_time = time.time()
                traj, success, poses, results = self.motion_gen_to_grasp_pose_ordered(cgn_result["pre_grasp_poses"],
                                                                                      cgn_result['grasp_ordered_lst'])
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
            self.update_cuRobo_motion_gen_config(attach_goal_obj=True)

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

            log['num_repetitive_trials'] = [k]
            if self.eval()['success'][0] or (not self.eval()['task_repeat'][0]):
                break

            self.open_gripper()
            self.update_cuRobo_world_collider_pose()

            start_time = time.time()
            traj, success, poses, results = self.motion_plan_to_retract_pose()
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)
            print("Reset Phase End")

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log


