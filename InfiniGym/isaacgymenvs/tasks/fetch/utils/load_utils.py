import trimesh
import json
import numpy as np
import os
import h5py
import pandas as pd


# Todo: Update this to be the asset path.
ASSET_PATH = os.environ["ASSET_PATH"]
SCENE_PATH = f"{ASSET_PATH}/Task"


def get_franka_panda_asset(type='franka_r3', mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': './assets',
            'urdf_file': f'urdf/franka_description/robots/{type}.urdf'
        }
    else:
        raise NotImplementedError

    return paths

"""
Scene Asset
"""


def load_scene_asset(path):
    file_metadata = np.load(f'{path}/metadata.npy')

    meshes, files = [], []
    for f in file_metadata:
        mesh_file = f'{path}/{f}.obj'
        m = trimesh.load_mesh(mesh_file)
        if isinstance(m, trimesh.Trimesh):
            meshes.append(m)
        else:
            meshes.extend(m.geometry.values())

        files.append(mesh_file)

    for m in meshes:
        m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, 1, 0, 0], [0, 0, 0, 1]]))

    with open(f'{path}/support.json', 'r') as f:
        support = json.load(f)

    with open(f'{path}/collider.json',  'r') as f:
        collider = json.load(f)

    robot_cam_config = np.load(f'{path}/robot_cam_config.npy', allow_pickle=True).tolist()

    return {
        'meshes': meshes,
        'files': files,
        'support': support,
        'collider': collider,
        'robot_cam_config': robot_cam_config
    }


def get_scene_asset(type, idx, mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/scenes/{type}/assets/scene_{idx}',
            'urdf_file': 'asset.urdf'
        }
    elif mode == 'benchmark':
        paths = {
            'asset_root': f'{ASSET_PATH}/benchmark_scenes/{type}/assets/scene_{idx}',
            'urdf_file': 'asset.urdf'
        }
    else:
        raise NotImplementedError

    return {**paths, **load_scene_asset(paths['asset_root'])}


def load_env_scene(config):
    if 'benchmark_scenes' in config['asset_root']:
        mode = 'benchmark'
    else:
        mode = 'ws'
    type = config['asset_root'].split('/')[-3]
    idx = config['asset_root'].split('/')[-1].split('_')[-1]
    asset = get_scene_asset(type, idx, mode)
    asset['name'] = config['name']
    return asset


def sample_random_scene(scene_type, scene_idx, mode='ws'):
    asset = get_scene_asset(scene_type, f"{scene_idx:03}", mode)
    asset['name'] = 'support_000'
    return asset


"""
Object Assets
"""


def load_object_asset(obj_path):
    mesh = trimesh.load_mesh(f'{obj_path}/mesh.obj')
    assert isinstance(mesh, trimesh.Trimesh)
    stable_poses = np.load(f'{obj_path}/stable_poses.npy')

    if os.path.exists(f'{obj_path}/grasp_poses.h5'):
        grasp_poses, grasp_success = load_h5_grasps(f'{obj_path}/grasp_poses.h5')
    elif os.path.exists(f'{obj_path}/grasp_poses.npy'):
        grasp_poses = np.load(f'{obj_path}/grasp_poses.npy', allow_pickle=True).tolist()
        grasp_poses = np.array(grasp_poses)
        grasp_poses = grasp_poses @ np.array([[0, -1, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], dtype=np.float32)
        grasp_success = np.ones((len(grasp_poses), ), dtype=np.float32)
    else:
        print("No Grasp File Found!")
        raise NotImplementedError

    metadata = np.load(f'{obj_path}/metadata.npy', allow_pickle=True).tolist()

    # load isaac labels
    default_label = np.load(f'{obj_path}/isaac_label_default.npy', allow_pickle=True).tolist()
    cvx_label = np.load(f'{obj_path}/isaac_label_cvx_4part.npy', allow_pickle=True).tolist()

    return {
        'mesh': mesh,
        'file': f'{obj_path}/mesh.obj',
        'stable_poses': stable_poses,
        'metadata': metadata,
        'grasp_poses': {
            'T': grasp_poses,
            'acronym_label': grasp_success,
            'isaac_label_default': default_label,
            'isaac_label_cvx': cvx_label
        }
    }


# Todo: Update combo asset loading
def load_object_combo_asset(path, placement_type):
    file_metadata = np.load(f'{path}/metadata.npy', allow_pickle=True).tolist()

    meshes, files = [], []
    for f in file_metadata['meshes']:
        mesh_file = f'{path}/{f}.obj'
        m = trimesh.load_mesh(mesh_file)
        if isinstance(m, trimesh.Trimesh):
            meshes.append(m)
        else:
            meshes.extend(m.geometry.values())

        files.append(mesh_file)

    for m in meshes:
        m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, 1, 0, 0], [0, 0, 0, 1]]))

    mesh = trimesh.util.concatenate(meshes)
    bb = mesh.bounds

    if placement_type == 'support':
        translation = np.array([0, 0, -bb[0][2]])
        stable_poses = [trimesh.transformations.translation_matrix(translation)]
    elif placement_type == 'hanging':
        translation = np.array([-bb[0][0], 0, -bb[0][2]])
        stable_poses = [trimesh.transformations.translation_matrix(translation)]
    else:
        raise NotImplementedError

    metadata = np.load(f'{path}/metadata.npy', allow_pickle=True).tolist()

    return {
        'meshes': meshes,
        'files': files,
        'stable_poses': stable_poses,
        'metadata': metadata
    }


def load_h5_grasps(filename):
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)

    # convert to the standard panda_hand frame
    T = T @ np.array([[0, -1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

    return T, success


def get_object_asset(type, idx, mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/objects/{type}/{idx}',
            'urdf_file': 'mesh.urdf',
        }
    elif mode == 'benchmark':
        paths = {
            'asset_root': f'{ASSET_PATH}/benchmark_objects/{type}/{idx}',
            'urdf_file': 'mesh.urdf',
        }
    else:
        raise NotImplementedError

    return {**paths, **load_object_asset(paths['asset_root'])}


# Todo: Update Combo Asset Load
def get_object_combo_asset(type, idx, mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/combos/{type}/{idx}',
            'urdf_file': ['organizer.urdf', 'object.urdf'],
            'combo_type': type
        }
    else:
        raise NotImplementedError

    if 'Hook' in type:
        paths['fixed_base'] = [True, False]
        paths['placement_type'] = 'hanging'
    else:
        paths['fixed_base'] = [False, False]
        paths['placement_type'] = 'support'

    return {**paths, **load_object_combo_asset(paths['asset_root'], placement_type=paths['placement_type'])}


def load_env_object(config):
    if 'benchmark_objects' in config['asset_root']:
        mode = 'benchmark'
    else:
        mode = 'ws'
    type = config['asset_root'].split('/')[-2]
    idx = config['asset_root'].split('/')[-1]
    asset = get_object_asset(type, idx, mode=mode)
    asset['name'] = config['name']

    return asset


# Todo: Update Combo Asset Load
def load_env_object_combo(config):
    type = config['asset_root'].split('/')[-2]
    idx = config['asset_root'].split('/')[-1]
    asset = get_object_combo_asset(type, idx)
    asset['name'] = config['name']

    return asset


def sample_random_objects(num_objs, eval_only=True,  mode='ws'):
    if mode == 'ws':
        metadata = pd.read_csv(f'{ASSET_PATH}/objects/metadata.csv')
        if eval_only:
            metadata = metadata.loc[metadata['Eval'] == True]
    elif mode == 'benchmark':
        metadata = pd.read_csv(f'{ASSET_PATH}/benchmark_objects/metadata.csv')
        metadata = metadata.loc[metadata['use_benchmark'] == True]
        if eval_only:
            metadata = metadata.loc[metadata['use_eval'] == True]
        else:
            metadata = metadata.loc[metadata['use_eval'] == False]
    else:
        raise NotImplementedError

    samples = metadata.sample(n=num_objs, replace=False)

    objects = []
    for i, (n, s) in enumerate(samples.iterrows()):
        o = get_object_asset(s['Category'], s['ID'], mode)
        o['name'] = f'obj_{i}'
        objects.append(o)

    return objects


# Todo: Update Combo Asset Sample
def sample_random_combos(num_objs, combo_type, mode='ws'):
    # Todo: Random sample of objects
    combo_indices = np.random.choice(os.listdir(f'{ASSET_PATH}/combos/{combo_type}'), size=(num_objs,))

    combos = []
    for i, idx in enumerate(combo_indices):
        o = get_object_combo_asset(combo_type, idx, mode)
        o['name'] = f'obj_combo_{i}'
        combos.append(o)

    return combos


def get_env_config(config):
    return f'{SCENE_PATH}/{config}'


class InfiniSceneLoader(object):
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        self.scene_asset_config = {}
        self.object_asset_config = []
        self.combo_asset_config = []
        self.robot_asset_config = {}
        self.camera_config = {}

        self.scene_pose = []
        self.robot_pose = []
        self.camera_poses = []
        self.object_poses = []
        self.object_labels = []

        self._num_compositions = 0

    def __len__(self):
        return len(self.scene_pose)

    def append_pose(self, pose, cat='scene'):
        if cat == 'scene':
            self.scene_pose.append(pose)
        elif cat == 'robot':
            self.robot_pose.append(pose)
        elif cat == 'camera':
            self.camera_poses.append(pose)
        elif cat == 'object':
            self.object_poses.append(pose)
        else:
            raise NotImplementedError

    def save_env_config(self, save_task_config=True):
        # vicinity check
        assert (len(self.scene_pose) == len(self.robot_pose) == len(self.camera_poses)
                == len(self.object_poses) == len(self.object_labels))

        self._num_compositions = len(self.scene_pose)
        # dump json
        with open(f'{self._path}/asset_config.json', 'w') as f:
            config = {
                'robot_config': self.robot_asset_config.copy(),
                'scene_config': self.scene_asset_config.copy(),
                'object_config': self.object_asset_config.copy(),
                'combo_config': self.combo_asset_config.copy(),
                'camera_config': self.camera_config.copy()
            }

            json.dump(config, f, indent=4)

        # dump ndarrays
        rearrange = {
            'robot_pose': self.robot_pose,
            'scene_pose': self.scene_pose,
            'object_poses': self.object_poses,
            'camera_poses': self.camera_poses,
            'object_labels': self.object_labels
        }
        np.savez(f'{self._path}/rearrange_config.npz', **rearrange)

        if save_task_config:
            tasks = self.create_env_tasks()
            np.savez(f'{self._path}/task_config.npz', **tasks)

    def create_env_tasks(self):
        init_root_states = self.get_scene_init_root_states()
        init_obj_labels = self.get_scene_init_obj_labels()
        init_camera_poses = self.get_camera_init_states()
        assert len(init_root_states) == len(init_obj_labels) == len(init_camera_poses)

        task_actor_states, task_obj_indices, task_obj_labels, task_camera_poses = [], [], [], []
        for k in range(len(init_root_states)):
            obj_indices, task_labels = self.get_obj_tasks(init_obj_labels[k])
            for idx, label in zip(obj_indices, task_labels):
                task_actor_states.append(init_root_states[k])
                task_obj_indices.append(idx)
                task_obj_labels.append(label)
                task_camera_poses.append(init_camera_poses[k])

        return {
            'task_init_state': task_actor_states,
            'task_obj_index': task_obj_indices,
            'task_obj_label': task_obj_labels,
            'task_camera_pose': task_camera_poses
        }

    def get_obj_tasks(self, obj_labels):
        task_obj_indices, task_labels = [], []
        for i, label in enumerate(obj_labels):
            if label.startswith('combo_org') or label.endswith('on_floor'):
                continue

            task_obj_indices.append(i)
            task_labels.append(label)

        return task_obj_indices, task_labels

    def load_env_config(self):
        with open(f'{self._path}/asset_config.json', 'r') as f:
            asset_config = json.load(f)

        self.scene_asset_config = asset_config['scene_config']
        self.robot_asset_config = asset_config['robot_config']
        self.object_asset_config = asset_config['object_config']
        self.combo_asset_config = asset_config['combo_config']
        self.camera_config = asset_config['camera_config']

        rearrange_config = np.load(f'{self._path}/rearrange_config.npz')

        self.scene_pose = rearrange_config['scene_pose']
        self.robot_pose = rearrange_config['robot_pose']
        self.object_poses = rearrange_config['object_poses']
        self.camera_poses = rearrange_config['camera_poses']
        self.object_labels = rearrange_config['object_labels']

        self._num_compositions = len(self.scene_pose)

    def load_task_config(self):
        task_config = np.load(f'{self._path}/task_config.npz')
        return {
            'task_init_state': task_config['task_init_state'],
            'task_obj_index': task_config['task_obj_index'],
            'task_obj_label': task_config['task_obj_label'],
            'task_camera_pose': task_config['task_camera_pose']
        }

    def get_camera_init_states(self):
        return self.camera_poses[:]

    def get_scene_init_root_states(self):
        # default seq: [robot, table, scene, *objects]
        if isinstance(self.robot_pose, list):
            self.robot_pose = np.stack(self.robot_pose, axis=0)
        if isinstance(self.scene_pose, list):
            self.scene_pose = np.stack(self.scene_pose, axis=0)
        if isinstance(self.object_poses, list):
            self.object_poses = np.stack(self.object_poses, axis=0)

        root_state = \
            np.concatenate(
                [self.robot_pose.reshape(-1, 2, 13), self.scene_pose.reshape(-1, 1, 13), self.object_poses], axis=1)
        return root_state[:]

    def get_scene_init_obj_labels(self):
        return self.object_labels[:]
