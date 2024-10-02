
import numpy as np
import torch
import trimesh

from isaacgym import gymutil, gymtorch, gymapi
import time
from isaacgymenvs.utils.torch_jit_utils import to_torch,  tf_combine, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video
from isaacgymenvs.tasks.fetch.fetch_solution_base import FetchSolutionBase

import sys
sys.path.append('../third_party/Optimus')
import optimus.modules.functional as F


def vis_scene(ptc, env_idx=0):
    scene = trimesh.Scene()
    axis = trimesh.creation.axis()
    scene.add_geometry(axis)

    for i, k in enumerate(['scene', 'robot', 'goal']):
        pts = ptc[k][env_idx][0].cpu().numpy()
        pts = trimesh.points.PointCloud(pts, colors=np.array([[255, 0, 100 * i]]).repeat(pts.shape[0], axis=0))
        scene.add_geometry(pts)

    scene.show()


def regularize_pc_point_count(pc, npoints):
    if len(pc) > npoints:
        pc = pc.reshape(-1, *pc.shape[-2:]).transpose(1, 2)
        sample = F.furthest_point_sample(pc, npoints).transpose(1, 2)[0]
    else:
        sample = torch.zeros(size=(npoints, 3), device=pc.device)
        if len(pc) > 0:
            sample[:pc.shape[0]] = pc
    return sample


class FetchPtdImitBase(FetchPointCloudBase, FetchSolutionBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.algo = None

        self._control_type = None
        self._seq_length = 0

        self.obs = []
        self.last_command = None

    """
    Solver Utils
    """
    def _get_robot_state(self):
        self._refresh()
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        eq, et = tf_combine(rq, rt,  self.states["eef_quat"].clone(), self.states["eef_pos"].clone())
        rq = rq.unsqueeze(1).repeat(1, self._rigid_body_state.shape[1], 1)
        rt = rt.unsqueeze(1).repeat(1, self._rigid_body_state.shape[1], 1)
        rbq, rbt = tf_combine(rq, rt, self._rigid_body_state[..., 3:7].clone(), self._rigid_body_state[..., :3].clone())

        rev = eq[:, -1:] >= 0.
        eq = eq * (2 * rev.float() - 1.)

        rev_rb = rbq[..., -1:] >= 0.
        rbq = rbq * (2 * rev_rb.float() - 1.)

        pose = {
            'eef_pos': et.clone(),
            'eef_quat': eq.clone(),
            'q': self.states["q"].clone(),
            'rigid_pos': rbt[:, :11].clone(),
            'rigid_quat': rbq[:, :11].clone(),
        }

        return pose

    def _update_obs(self, phase=None):
        point_clouds = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=True, segmented_ptd=True,
                                            ptd_downscale=self.cfg["solution"]['imit_ptd_subsample'])['camera_pointcloud_seg']
        robot_state = self._get_robot_state()

        if phase is None:
            phase_index = np.array([[0., 0.]])
        elif 'fetch' in phase:
            phase_index = np.array([[0., 1.]])
        elif 'grasp' in phase:
            phase_index = np.array([[1., 0.]])
        else:
            raise NotImplementedError

        phase_index = torch.from_numpy(phase_index).float().repeat(self.num_envs, 1)

        ptc_samples = []
        for env_idx, ptc_full in enumerate(point_clouds):
            ptc = {}
            for k, v in ptc_full.items():
                pts = v[v.norm(dim=-1) <= 1.2]  # workspace range, hard coded
                ptc[k] = regularize_pc_point_count(pts, self.algo.algo_config['obs_shape'][f'num_{k}_points'])
            ptc_samples.append(ptc)

        self.obs.append({
            'point_cloud': ptc_samples,
            'robot_state': robot_state,
            'phase_index': phase_index
        })
        if len(self.obs) > self._seq_length:
            self.obs = self.obs[-self._seq_length:]

    def get_algo_input_batch(self):
        batch = {
            'q': [],
            'eef_pos': [],
            'eef_quat': [],
            'rigid_pos': [],
            'rigid_quat': [],
            'visual': {
                'goal': [],
                'scene': [],
                'robot': []
            }
        }

        if self.cfg["solution"]["config"]["algo"]["two_phase"]:
            batch['phase_index'] = []

        for step in range(-self._seq_length, 0):
            s = max(len(self.obs) + step, 0)

            batch['eef_pos'].append(self.obs[s]["robot_state"]["eef_pos"])
            batch['eef_quat'].append(self.obs[s]["robot_state"]["eef_quat"])
            batch['q'].append(self.obs[s]["robot_state"]["q"])
            batch['rigid_pos'].append(self.obs[s]['robot_state']['rigid_pos'])
            batch['rigid_quat'].append(self.obs[s]['robot_state']['rigid_quat'])

            for k in ['goal', 'scene', 'robot']:
                ptc_envs = []
                for env_idx in range(self.num_envs):
                    ptc_envs.append(self.obs[s]["point_cloud"][env_idx][k])
                ptc_envs = torch.stack(ptc_envs, dim=0)
                batch['visual'][k].append(ptc_envs)

            if self.cfg["solution"]["config"]["algo"]["two_phase"]:
                batch['phase_index'].append(self.obs[s]['phase_index'])

        batch['eef_pos'] = torch.stack(batch['eef_pos'], dim=1).detach()
        batch['eef_quat'] = torch.stack(batch['eef_quat'], dim=1).detach()
        batch['q'] = torch.stack(batch['q'], dim=1).detach()
        batch['rigid_pos'] = torch.stack(batch['rigid_pos'], dim=1).detach()
        batch['rigid_quat'] = torch.stack(batch['rigid_quat'], dim=1).detach()

        for k in ['goal', 'scene', 'robot']:
            batch['visual'][k] = torch.stack(batch['visual'][k], dim=1).detach()

        if self.cfg["solution"]["config"]["algo"]["two_phase"]:
            batch['phase_index'] = torch.stack(batch['phase_index'], dim=1).detach()

        return batch

    def update_algo(self, algo):
        self.algo = algo
        self.algo.set_eval()
        self._seq_length = self.algo.global_config.dataset.seq_length
        self._control_type = self.algo.global_config.dataset.action_type

        if self.cfg["solution"]["config"] is None:
            self.cfg["solution"]["config"] = self.algo.global_config

        assert self.cfg["solution"]["config"]["algo"]["two_phase"] == True

    def reset(self):
        super().reset()
        self.last_command = None
        self.obs = []

    """
    Solve
    """

    def solve(self):
        pass
