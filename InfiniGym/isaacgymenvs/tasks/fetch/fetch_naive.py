
import numpy as np
import os
import torch
import imageio
import trimesh.transformations as tra

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_mul, quat_conjugate, quat_apply, quat_to_angle_axis, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_base import FetchBase


def image_to_video(obs_buf):
    video = []
    for s, images in enumerate(obs_buf):
        steps = []
        for e, imgs in enumerate(images):
            steps.append(np.concatenate(imgs, axis=0))
        video.append(np.concatenate(steps, axis=1))
    return video


class FetchNaive(FetchBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self._init_steps = self.cfg["solution"]["init_steps"]
        self._eval_steps = self.cfg["solution"]["eval_steps"]

        self._solution_video = []
        self._video_freq = self.cfg["solution"]["video_freq"]
        self._video_frame = 0

        assert self.arm_control_type == 'joint'

    """
    Your Solution
    """

    def repeat(self):
        # curr state
        q = self.states["q"].clone()
        self._arm_control = q[:, :-2]
        self._gripper_control = q[:, -2:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

        for _ in range(self._eval_steps):
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

    def solve(self):
        # set goal obj color
        self.set_target_color()
        self._solution_video = []

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()
            rgb, seg = self.get_camera_image(rgb=True, seg=False)
            self.log_video(rgb)

        log = {
            'traj_length': self._traj_length.cpu().numpy(),
            'computing_time':  [0 for _ in range(self.num_envs)]
        }

        self.repeat()

        self.set_default_color()

        return image_to_video(self._solution_video), log


    """
    Camera
    """

    def log_video(self, rgb):
        if self._video_frame % self._video_freq == 0:
            self._solution_video.append(rgb)
        self._video_frame += 1