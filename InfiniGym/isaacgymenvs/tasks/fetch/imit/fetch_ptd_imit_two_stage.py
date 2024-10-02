
import numpy as np
import torch
import trimesh

from isaacgym import gymutil, gymtorch, gymapi
import time
from isaacgymenvs.utils.torch_jit_utils import to_torch,  tf_combine, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video
from isaacgymenvs.tasks.fetch.imit.fetch_ptd_imit_base import FetchPtdImitBase


class FetchPtdImitTwoStage(FetchPtdImitBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self._task_cmd_status = torch.ones((self.num_envs, ), device=self.device, dtype=torch.bool)

    """
    Action
    """

    def step_action(self, actions, computing_time, gripper_state):
        self.switch_arm_control_type(self._control_type, impedance=self.cfg["solution"]["impedance_control"])

        # simulate computing delay
        actions = actions.clone().to(self.device)
        delay_steps = int(computing_time / self.cfg["sim"]["dt"])
        if delay_steps > 0 and self.last_command is not None and self.cfg["env"]["sim_delay_steps"]:
            for i in range(delay_steps):
                self.pre_phy_step(self.last_command, robot_base=True)
                self.env_physics_step()
                self.post_phy_step()
                rgb, seg = self.get_camera_image(rgb=True, seg=False)
                self.log_video(rgb)
        else:
            delay_steps = 0

        command = {}
        if self._control_type == 'osc':
            task_status = self._task_cmd_status.float().reshape(-1, 1)
            robot_state = self._get_robot_state()
            eef_pos = robot_state["eef_pos"].clone() * (1. - task_status) + actions[:, :3] * task_status
            eef_quat = robot_state["eef_quat"].clone() * (1. - task_status) + actions[:, 3:-1] * task_status

            command = {
                'eef_pos': eef_pos,
                'eef_quat': eef_quat,
            }
        elif self._control_type == 'joint':

            curr_state = self.states["q"][:, :-2].clone()
            input_action = actions[:, :-1]

            task_status = self._task_cmd_status.float().reshape(-1, 1)
            joint_cmd = curr_state * (1. - task_status) + input_action * task_status

            joint_cmd = torch.clamp(joint_cmd, self.robot_dof_lower_limits[:-2].unsqueeze(0),
                                    self.robot_dof_upper_limits[:-2].unsqueeze(0))
            command = {"joint_state": joint_cmd}
        else:
            raise NotImplementedError

        if gripper_state == 0:
            command['gripper_state'] = None
        else:
            command['gripper_state'] = gripper_state * torch.ones((self.num_envs,), device=self.device)

        cmd_repeat = (self.cfg["solution"]["num_steps_repeat_per_cmd"] *
                      self.algo.global_config.dataset.frame_skip)

        for i in range(cmd_repeat):
            self.pre_phy_step(command, robot_base=True)
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

        self.switch_arm_control_type('joint')

        # update phase end
        command_task_state = actions[:, -1] < 0.
        self._task_cmd_status = command_task_state & self._task_cmd_status
        self.last_command = command
        return delay_steps

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

        self._task_cmd_status = torch.ones((self.num_envs, ), device=self.device, dtype=torch.bool)
        # grasp_phase
        for i in range(self.cfg["solution"]["num_grasp_steps"] // self.cfg["solution"]["config"]["dataset"]["frame_skip"]):
            self._update_obs('grasp_traj')

            start_time = time.time()
            input = self.get_algo_input_batch()

            actions = self.algo.get_action(input, i)
            delay_time = time.time() - start_time
            computing_time += delay_time

            delay = self.step_action(actions, delay_time / self.num_envs, 0)

            if not self._task_cmd_status.any().cpu().numpy():
                break

        self.close_gripper()
        log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
        log['grasp_phase_delay_steps_per_cmd'] = delay
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
