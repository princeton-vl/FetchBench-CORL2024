
import numpy as np
import torch
import trimesh

from isaacgym import gymutil, gymtorch, gymapi
import time
from isaacgymenvs.utils.torch_jit_utils import to_torch,  tf_combine, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video
from isaacgymenvs.tasks.fetch.imit.fetch_ptd_imit_base import FetchPtdImitBase


class FetchPtdImitE2E(FetchPtdImitBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self._task_status = None

        assert self.num_envs == 1

    """
    Action
    """
    def _get_gripper_state(self):
        dist = self.states['q'][:, -2:].sum(dim=-1).cpu().numpy()[0]
        if dist > 0.075:
            return 'open'
        elif dist < 0.002:
            return 'close'
        else:
            return 'grasp'

    def step_action(self, actions, computing_time):
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

        if self._control_type == 'osc':
            command = {
                'eef_pos': actions[:, :3],
                'eef_quat': actions[:, 3:-1],
            }
        elif self._control_type == 'joint':
            joint_cmd = torch.clamp(actions[:, :-1], self.robot_dof_lower_limits[:-2].unsqueeze(0),
                                    self.robot_dof_upper_limits[:-2].unsqueeze(0))
            command = {"joint_state": joint_cmd}
        else:
            raise NotImplementedError

        if 'grasp' in self._task_status and self._get_gripper_state() == 'open':
            command['gripper_state'] = None
        elif 'grasp' in self._task_status and not (self._get_gripper_state() == 'open'):
            command['gripper_state'] = torch.ones((self.num_envs,), device=self.device)
        elif 'fetch' in self._task_status:
            command['gripper_state'] = - torch.ones((self.num_envs,), device=self.device)
        else:
            raise NotImplementedError

        cmd_repeat = (self.cfg["solution"]["num_steps_repeat_per_cmd"] *
                      self.algo.global_config.dataset.frame_skip)

        for i in range(cmd_repeat):
            self.pre_phy_step(command, robot_base=True)
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

        self.switch_arm_control_type('joint')

        self.last_command = command

        # additional executions
        if actions[0, -1].cpu().numpy() > 0:
            self.obs = []
            self.last_command = None
            if 'grasp' in self._task_status:
                self._task_status = 'fetch_traj'
                return 'close'
            elif 'fetch' in self._task_status:
                self._task_status = 'end'
                return 'end'

        if 'fetch' in self._task_status and not (self._get_gripper_state() == 'grasp'):
            self.obs = []
            self.last_command = None
            self._task_status = 'grasp_traj'
            return 'open'

        return None

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

        self._task_status = 'grasp_traj'  # init with grasp traj

        step_counter = 0
        for i in range(self.cfg["solution"]["max_num_steps"]):
            self._update_obs(self._task_status)

            start_time = time.time()
            input = self.get_algo_input_batch()
            actions = self.algo.get_action(input, step_counter)
            delay_time = time.time() - start_time
            computing_time += delay_time

            gripper_action = self.step_action(actions, delay_time / self.num_envs)
            step_counter += 1

            if gripper_action == 'close':
                print("Close Gripper")
                self.close_gripper()
                step_counter = 0
            elif gripper_action == 'open':
                print("Re-open Gripper")
                self.open_gripper()
                step_counter = 0
            elif gripper_action == 'end':
                print("Policy End")
                break

        log['delay_steps_per_cmd'] = delay_time / self.num_envs

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log