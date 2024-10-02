
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
import time
from isaacgymenvs.tasks.fetch.fetch_ptd_curobo_cgn_beta import FetchPtdCuroboCGNBeta
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video
from isaacgymenvs.tasks.fetch.imit.fetch_ptd_imit_two_stage import FetchPtdImitTwoStage


class FetchPtdImitCuroboCGN(FetchPtdImitTwoStage, FetchPtdCuroboCGNBeta):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        assert self.arm_control_type == 'joint'

    """
    Solve
    """

    def solve(self):
        log = {}

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

        self._task_cmd_status = torch.ones((self.num_envs, ), device=self.device, dtype=torch.bool)
        self.obs = []
        for i in range(self.cfg["solution"]["num_fetch_steps"] // self.cfg["solution"]["config"]["dataset"]["frame_skip"]):
            self._update_obs('fetch_traj')

            start_time = time.time()
            input = self.get_algo_input_batch()

            actions = self.algo.get_action(input, i)
            delay_time = time.time() - start_time
            computing_time += delay_time

            delay = self.step_action(actions, delay_time / self.num_envs, gripper_state=-1)

            if not self._task_cmd_status.any().cpu().numpy():
                break

        log['fetch_phase_delay_steps_per_cmd'] = delay

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log

