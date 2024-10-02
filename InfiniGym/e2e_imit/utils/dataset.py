import bisect

import numpy as np
import pandas as pd
import os

import trimesh.transformations as tra
import trimesh
import shutil

from multiprocessing import Pool
from collections import OrderedDict


ROOT_PATH = os.environ["ASSET_PATH"]


def truncate_traj(b, steps):
    if isinstance(b, np.ndarray):
        seg = b[steps].astype(np.float32)
    else:
        seg = {}
        for k, v in b.items():
            seg[k] = truncate_traj(v, steps)

    return seg


def regularize_pc_point_count(pc, npoints):

    if pc.shape[0] > npoints:
        center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            zeros = np.zeros((required, 3), dtype=pc.dtype)
            pc = np.concatenate((pc, zeros), axis=0)
    return pc


class SequenceDataset(object):
    def __init__(
            self,
            dir_path,
            metadata_file='metadata.csv',
            traj_keys=[],
            seq_length=8,
            frame_skip=1,

            filter_by_length=True,
            top_k_trajs=5,

            action_type='joint',
            action_horizon=8,
            action_params={},
            two_phase=True,
            obs_shape={},

            downscale_dataset_by_frame_skip=False,
            data_aug_params={},
            **kwargs
    ):

        if os.path.exists(f'/scratch/bh7032/{dir_path}/{metadata_file}'):
            self.dir_path = f'/scratch/bh7032/{dir_path}'
            print('=================Use scratch space.=================')
        else:
            # warning: slow training
            self.dir_path = f'{ROOT_PATH}/{dir_path}'
            print('=================Use local space.===================')

        self.obs_shape = obs_shape
        self.traj_keys = traj_keys

        self.frame_skip = frame_skip
        assert self.frame_skip >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.two_phase = two_phase

        self.action_type = action_type
        self.action_params = action_params
        self.action_horizon = action_horizon

        self.data_aug_params = data_aug_params

        self.traj_index_list = []
        self.load_demo_info(metadata_file, filter_by_length=filter_by_length, top_k_trajs=top_k_trajs)

        self.downscale_by_frame_skip = downscale_dataset_by_frame_skip

    def load_demo_info(self, file, filter_by_length=False, top_k_trajs=5):
        # filter demo trajectory by mask
        metadata = pd.read_csv(f'{self.dir_path}/{file}')

        if filter_by_length:
            new_metadata = []

            traj_mean = metadata['Traj_Len'].mean()
            traj_std = np.sqrt(metadata['Traj_Len'].var())
            n_trajs = metadata.loc[metadata['Traj_Len'] <= traj_mean + 2 * traj_std]

            for (group_value1, group_value2, group_value3), group_data in (
                    n_trajs.groupby(['Scene_Category', 'Scene_Idx', 'Task_Idx'])):

                if len(group_data) > top_k_trajs:
                    group_data = group_data.sort_values('Traj_Len', ascending=True).head(top_k_trajs)

                new_metadata.append(group_data)
            metadata = pd.concat(new_metadata)

        c = 0
        for i, traj in metadata.iterrows():
            self.traj_index_list.append(c)
            for k in self.traj_keys:
                c += traj[f'Traj_{k}_Steps']

        self.traj_metadata = metadata
        self.total_num_trajs = len(metadata)
        self.total_traj_steps = c
        print("Total Trajs Used: ", self.total_num_trajs)

    def __len__(self):
        if not self.downscale_by_frame_skip:
            return self.total_traj_steps
        else:
            return self.total_traj_steps // self.frame_skip

    def __getitem__(self, index):
        if not self.downscale_by_frame_skip:
            return self.get_traj_seg(index)
        else:
            return self.get_traj_seg(index * self.frame_skip)

    def get_obs_shape(self):
        obs_shape = OrderedDict({
            "q": (9,),
            "eef_pos": (3,),
            "eef_quat": (4,),
            "rigid_pos": (11, 3),
            "rigid_quat": (11, 4),
            "visual": {
                'scene': (self.obs_shape['num_scene_points'], 3),
                'goal': (self.obs_shape['num_goal_points'], 3),
                'robot': (self.obs_shape['num_robot_points'], 3)
            }
        })
        if self.two_phase:
            obs_shape['phase_index'] = (2,)

        return obs_shape

    def get_action_params(self):
        if self.action_type == 'joint':
            return {
                'shape': (8,),
                'type': 'joint',
                'scale': self.action_params['dof'] * self.frame_skip * self.action_params['clip_std_scale']
            }
        elif self.action_type == 'osc':
            return {
                'shape': (7,),
                'type': 'osc',
                'scale': {
                    'pos': self.action_params['eef_pos'] * self.frame_skip * self.action_params['clip_std_scale'],
                    'angle': self.action_params['eef_angle'] * self.frame_skip * self.action_params['clip_std_scale']
                }
            }
        else:
            raise NotImplementedError

    """
    Obs
    """

    def get_visual_input(self, traj, step):
        scene = traj[step]['scene_point_cloud']['scene']
        goal = traj[step]['scene_point_cloud']['goal']
        robot = traj[step]['scene_point_cloud']['robot']

        scene = regularize_pc_point_count(scene, self.obs_shape['num_scene_points'])
        goal = regularize_pc_point_count(goal, self.obs_shape['num_goal_points'])
        robot = regularize_pc_point_count(robot, self.obs_shape['num_robot_points'])

        if np.random.uniform() < self.data_aug_params['jitter_noise_prob']:
            scene = self._jitter_ptc(scene)
            goal = self._jitter_ptc(goal)
            robot = self._jitter_ptc(robot)

        return {'scene': scene, 'goal': goal, 'robot': robot}

    def get_obs_input(self, traj, seg_steps, phase='fetch'):

        batch_input = {
            'q': [],
            'eef_pos': [],
            'eef_quat': [],
            'rigid_pos': [],
            'rigid_quat': [],
            'visual': {
                'scene': [],
                'goal': [],
                'robot': []
            }
        }

        if self.two_phase:
            batch_input['phase_index'] = []

        for s in seg_steps:
            batch_input['q'].append(traj[s]['dof_state'][:, 0])
            eef_pos, eef_quat = self._convert_to_robot_base(traj[s]['eef_state'], traj[s]['root_state'])
            eef_quat = np.concatenate([eef_quat[1:], eef_quat[:1]], axis=0)  # (x, y, z, w)
            assert eef_quat[-1] >= 0, "Quaternion < 0"
            batch_input['eef_pos'].append(eef_pos)
            batch_input['eef_quat'].append(eef_quat)

            rb_poses, rb_quats = [], []
            for r in range(11):
                rb_pos, rb_quat = self._convert_to_robot_base(traj[s]['rigid_shape_state'][r], traj[s]['root_state'])
                rb_quat = np.concatenate([rb_quat[1:], rb_quat[:1]], axis=0)
                assert rb_quat[-1] >= 0, "Quaternion < 0"
                rb_poses.append(rb_pos)
                rb_quats.append(rb_quat)

            batch_input['rigid_pos'].append(np.array(rb_poses))
            batch_input['rigid_quat'].append(np.array(rb_quats))

            ptc = self.get_visual_input(traj, s)

            batch_input['visual']['scene'].append(ptc['scene'])
            batch_input['visual']['goal'].append(ptc['goal'])
            batch_input['visual']['robot'].append(ptc['robot'])

            if self.two_phase:
                if 'fetch' in phase:
                    batch_input['phase_index'].append(np.array([0., 1.]))
                elif 'grasp' in phase:
                    batch_input['phase_index'].append(np.array([1., 0.]))
                else:
                    raise NotImplementedError

        batch_input['q'] = np.stack(batch_input['q'], axis=0).astype(np.float32)
        batch_input['eef_pos'] = np.stack(batch_input['eef_pos'], axis=0).astype(np.float32)
        batch_input['eef_quat'] = np.stack(batch_input['eef_quat'], axis=0).astype(np.float32)
        batch_input['rigid_pos'] = np.stack(batch_input['rigid_pos'], axis=0).astype(np.float32)
        batch_input['rigid_quat'] = np.stack(batch_input['rigid_quat'], axis=0).astype(np.float32)

        if np.random.uniform() < self.data_aug_params['vec_noise_prob']:
            batch_input = self._aug_vec_input(batch_input)

        batch_input['visual']['scene'] = np.stack(batch_input['visual']['scene'], axis=0).astype(np.float32)
        batch_input['visual']['goal'] = np.stack(batch_input['visual']['goal'], axis=0).astype(np.float32)
        batch_input['visual']['robot'] = np.stack(batch_input['visual']['robot'], axis=0).astype(np.float32)

        if self.two_phase:
            batch_input['phase_index'] = np.stack(batch_input['phase_index'], axis=0).astype(np.float32)

        return batch_input

    def _aug_vec_input(self, batch):
        for k in ['q', 'eef_pos', 'eef_quat']:
            noise = np.random.randn(*batch[k].shape) * self.data_aug_params['vec_randn_std']
            noise = np.clip(noise, -self.data_aug_params['vec_randn_clip'], self.data_aug_params['vec_randn_clip'])
            batch[k] += noise

        return batch

    def _jitter_ptc(self, ptc):
        noise = np.random.normal(size=ptc.shape) * self.data_aug_params['jitter_noise_std']
        noise = np.clip(noise, -self.data_aug_params['jitter_noise_clip'], self.data_aug_params['jitter_noise_clip'])
        ptc += noise
        return ptc

    def _convert_to_robot_base(self, eef_state, root_states):
        base_pos, base_quat = root_states[0, :3], root_states[0, 3:7]
        eef_pos, eef_quat = eef_state[:3], eef_state[3:7]
        base_quat = np.concatenate([base_quat[-1:], base_quat[:-1]], axis=0)
        eef_quat = np.concatenate([eef_quat[-1:], eef_quat[:-1]], axis=0)

        base_T = tra.translation_matrix(base_pos) @ tra.quaternion_matrix(base_quat)
        eef_T = tra.translation_matrix(eef_pos) @ tra.quaternion_matrix(eef_quat)

        eef_in_base = tra.inverse_matrix(base_T) @ eef_T
        eef_in_base_pos = eef_in_base[:3, 3]
        eef_in_base_quat = tra.quaternion_from_matrix(eef_in_base[:3, :3])
        assert eef_in_base_quat[0] >= 0., "Quat Error"
        return eef_in_base_pos, eef_in_base_quat

    """
    Action
    """

    def get_action_seq_input(self, traj, step, phase):
        """
        This function returns the action of curr+h - curr .
        """
        target_step = min(len(traj) - 1, step + self.frame_skip)

        seq_actions = []
        if self.action_type == 'joint':
            ls = step
            for h in range(self.action_horizon):
                ns = min(len(traj)-1, target_step + h * self.frame_skip)
                arm_action = traj[ns]['dof_state'][:-2, 0] - traj[ls]['dof_state'][:-2, 0]
                arm_action = arm_action / (self.action_params['dof'] * self.frame_skip * self.action_params['clip_std_scale'])
                arm_action = np.clip(arm_action, -1.0, 1.0)

                if target_step + h * self.frame_skip >= len(traj) - 1 or (not self.two_phase and 'fetch' in phase):
                    gripper_action = 1.0
                else:
                    gripper_action = -1.0

                seq_actions.append(np.concatenate([arm_action, [gripper_action]], axis=0))
                ls = ns

        elif self.action_type == 'osc':
            ls = step
            for h in range(self.action_horizon):
                ns = min(len(traj) - 1, target_step + h * self.frame_skip)
                target_pos, target_quat = self._convert_to_robot_base(traj[ns]['eef_state'], traj[ns]['root_state'])
                curr_pos, curr_quat = self._convert_to_robot_base(traj[ls]['eef_state'], traj[ls]['root_state'])

                arm_delta_pos = target_pos - curr_pos
                delta_quat = tra.quaternion_multiply(target_quat, tra.quaternion_conjugate(curr_quat))

                axis_norm = np.linalg.norm(delta_quat[1:], axis=-1)
                angle = 2 * np.arctan2(axis_norm, np.abs(delta_quat[0]))

                if axis_norm < 5e-5:
                    axis_angle = np.array([0, 0, 0])
                else:
                    rev = delta_quat[0] > 0.
                    axis_angle = (delta_quat[1:] / np.sin(angle / 2.)) * angle
                    axis_angle = axis_angle * (rev.astype(np.float32) * 2. - 1.)

                arm_delta_angle = axis_angle
                arm_action = np.concatenate([arm_delta_pos, arm_delta_angle], axis=-1)
                arm_action[:3] = arm_action[:3] / (
                            self.action_params['eef_pos'] * self.frame_skip * self.action_params['clip_std_scale'])
                arm_action[3:] = arm_action[3:] / (
                            self.action_params['eef_angle'] * self.frame_skip * self.action_params['clip_std_scale'])

                if target_step + h * self.frame_skip >= len(traj) - 1 or (not self.two_phase and 'fetch' in phase):
                    gripper_action = 1.0
                else:
                    gripper_action = -1.0

                seq_actions.append(np.concatenate([arm_action, [gripper_action]], axis=0))
                ls = ns
        else:
            raise NotImplementedError

        return np.stack(seq_actions, axis=0).astype(np.float32)

    """
    Get Traj
    """

    def _locate_traj_id(self, index):
        traj_id = bisect.bisect_right(self.traj_index_list, index)-1
        end_step = index - (self.traj_index_list[traj_id] if traj_id > 0 else 0)

        return traj_id, end_step

    def get_traj_seg(self, index):
        traj_id, end_step = self._locate_traj_id(index)
        traj_data = self.traj_metadata.iloc[traj_id]

        traj = np.load(f'{self.dir_path}/{traj_data["Path"]}', allow_pickle=True).tolist()
        if isinstance(traj, list):
            traj = traj[traj_data['Traj_Idx']]

        seg = {}

        if len(self.traj_keys) == 2:
            if end_step >= traj_data[f'Traj_{self.traj_keys[0]}_Steps']:
                index_k = self.traj_keys[1]
                index_s = end_step - traj_data[f'Traj_{self.traj_keys[0]}_Steps']
            else:
                index_k = self.traj_keys[0]
                index_s = end_step

            seg_steps = [max(0, index_s - self.frame_skip * i) for i in range(self.seq_length-1, -1, -1)]
            seg = self.get_obs_input(traj[index_k], seg_steps, index_k)
            seg['actions'] = self.get_action_seq_input(traj[index_k], index_s, index_k)

        elif len(self.traj_keys) == 1:

            seg_steps = [max(0, end_step - self.frame_skip * i) for i in range(self.seq_length-1, -1, -1)]
            seg = self.get_obs_input(traj[self.traj_keys[0]], seg_steps, self.traj_keys[0])
            seg['actions'] = self.get_action_seq_input(traj[self.traj_keys[0]], end_step, self.traj_keys[0])

        return seg

    def get_full_traj(self, scene_cat, scene_idx, task_idx):
        trajs = (self.traj_metadata.loc[self.traj_metadata['Scene_Category'] == scene_cat].
        loc[self.traj_metadata['Scene_Idx'] == scene_idx].loc[self.traj_metadata['Task_Idx'] == task_idx])
        idx = np.random.randint(len(trajs))

        traj = np.load(f'{self.dir_path}/{trajs.iloc[idx]["Path"]}', allow_pickle=True).tolist()

        traj_actions, traj_obs = {}, {}
        for k in self.traj_keys:
            traj_seg = traj[k]
            num_steps = len(traj_seg[::self.frame_skip])
            actions, obs_steps = [], []
            for h in range(num_steps):
                actions.append(self.get_action_seq_input(traj_seg, h * self.frame_skip, k))
                obs_steps.append(h * self.frame_skip)

            traj_actions[k] = np.stack(actions, axis=0)
            traj_obs[k] = self.get_obs_input(traj[k], obs_steps, k)

        return traj_actions, traj_obs


if __name__ == '__main__':
    dataset = SequenceDataset('./Trajs', traj_keys=['grasp_traj', 'fetch_traj'],
                              frame_skip=2, action_type='joint', top_k_trajs=1,
                                         action_params={'clip_std_scale': 5, 'dof': 0.029,
                                                        'eef_pos': 0.012, 'eef_angle': 0.0215},
                                        obs_shape={
                                            'num_scene_points': 16384,
                                            'num_robot_points': 4096,
                                            'num_goal_points': 2048,
                                        },
                                         data_aug_params={'vec_noise_prob': 0.5,
                                                          'vec_randn_std': 0.005,
                                                          'vec_randn_clip': 0.015,
                                                          'jitter_noise_std': 0.0,
                                                          'jitter_noise_clip': 0.03})
    print(len(dataset))
    import time
    st = time.time()
    trajs = dataset[1]
    print(time.time() - st)

    # vis trajs
    scene = trimesh.Scene()
    for i, n in enumerate(['scene', 'goal', 'robot']):
        color = np.array([[0, 0, 0]])
        color[0][i] = 255
        color = color.repeat(len(trajs['visual'][n][0]), axis=0)
        pts = trimesh.points.PointCloud(trajs['visual'][n][0], colors=color)
        scene.add_geometry(pts)

    scene.show()



