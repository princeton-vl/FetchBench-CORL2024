
import numpy as np
import os
import torch
import imageio
import trimesh.transformations as tra

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.fetch.fetch_base import FetchBase
from isaacgymenvs.tasks.fetch.utils.load_utils import SCENE_PATH


class FetchCheck(FetchBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render)

        self.seg_checks = []

    def step(self, actions=None):
        # set goal obj color
        for i, env_ptr in enumerate(self.envs):
            self.gym.set_rigid_body_color(env_ptr,
                                          self.objects[i][self.task_obj_index[i][self._task_idx].cpu()],
                                          0, gymapi.MESH_VISUAL, self.task_obj_color)

        for _ in range(self.cfg["solution"]["init_steps"]):
            self.env_physics_step()
            self.post_phy_step()

        rgb, seg = self.get_camera_image(rgb=True, seg=True)
        self.seg_checks.append(self.check_image_seg(seg))
        self.log_camera_view_image(rgb)

        # set to default color
        for i, env_ptr in enumerate(self.envs):
            self.gym.set_rigid_body_color(env_ptr,
                                          self.objects[i][self.task_obj_index[i][self._task_idx].cpu()],
                                          0, gymapi.MESH_VISUAL, self.default_obj_color)

    def check_image_seg(self, render_seg_obs_buf):
        seg_bit = []
        for i, env_ptr in enumerate(self.envs):
            seg_obs_buf = render_seg_obs_buf[i]
            for j in range(len(seg_obs_buf)-1):
                seg_image = seg_obs_buf[j]
                obj_seg_idx = self.task_obj_index[i].cpu().numpy()[self._task_idx] + 4
                pixels = (seg_image == obj_seg_idx).astype(np.int32).sum()
                seg_bit.append(pixels > 500)

        return seg_bit

    def log_camera_view_image(self, render_rgb_obs_buf):
        for i, images in enumerate(render_rgb_obs_buf):
            if not os.path.exists(f'{SCENE_PATH}/{self.scene_config_path[i]}/video'):
                os.makedirs(f'{SCENE_PATH}/{self.scene_config_path[i]}/video')
            for j, img in enumerate(images):
                imageio.imwrite(f'{SCENE_PATH}/{self.scene_config_path[i]}/video/env_{i}_{self._task_idx}_{j}.png', img)

    def check_all_task_configs(self):
        assert self.num_envs == 1
        for idx in range(self.task_obj_index[0].shape[0]):
            self.reset_task(idx)
            self.step(None)

        return self.seg_checks



