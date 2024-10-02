import numpy as np
import pandas as pd
import os
import trimesh.transformations as tra
import shutil


def get_actions(traj):
    traj_dofs, traj_eefs = [], []
    for t in range(len(traj) - 1):
        dof_action = traj[t+1]['dof_state'][:-2, 0] - traj[t]['dof_state'][:-2, 0]

        arm_delta_pos = traj[t+1]['eef_state'][:3] - traj[t]['eef_state'][:3]
        target_quat = np.concatenate([traj[t+1]['eef_state'][6:7], traj[t+1]['eef_state'][3:6]], axis=-1)
        curr_quat = np.concatenate([traj[t]['eef_state'][6:7], traj[t]['eef_state'][3:6]], axis=-1)
        arm_delta_quat = tra.quaternion_multiply(target_quat, tra.quaternion_conjugate(curr_quat))
        eef_action = np.concatenate([arm_delta_pos, arm_delta_quat], axis=-1)

        traj_dofs.append(dof_action)
        traj_eefs.append(eef_action)

    return traj_dofs, traj_eefs


def get_traj_stats(dof_action, eef_action):
    dof_mean = np.abs(dof_action).mean(axis=0).max()
    eef_pos = np.linalg.norm(eef_action[:, :3], axis=-1).mean()
    eef_quat = np.linalg.norm(eef_action[:, -3:], axis=-1).mean()

    return {
        'length': np.linalg.norm(dof_action, axis=-1).sum(),
        'dof_mean': dof_mean,
        'eef_mean': np.array([eef_pos, eef_quat]),
    }


def generate_metadata(args, len_keys):
    metadata, dof_actions, eef_actions = [], [], []

    for f in os.listdir(args.dataset_path):
        scene_name = f.split('_')[0]
        scene_idx = f.split('_')[1]
        path = args.dataset_path
        for t in os.listdir(f'{path}/{f}'):
            if (not t.endswith('.npy')) or t.endswith('scene.npy'):
                continue

            traj = np.load(f'{path}/{f}/{t}', allow_pickle=True).tolist()
            task_idx = t.split('_')[1]
            traj_idx = t.split('_')[-1][4:-4]

            entry = {
                'Scene_Category': scene_name,
                'Scene_Idx': int(scene_idx),
                'Task_Idx': int(task_idx),
                'Path': f'{f}/{t}',
                'Traj_Idx': int(traj_idx)
            }

            traj_dofs, traj_eefs, traj_steps = [], [], 0
            for k in len_keys:
                dof, eef = get_actions(traj[k])
                traj_dofs.extend(dof)
                traj_eefs.extend(eef)
                entry[f'Traj_{k}_Steps'] = len(dof)+1
                traj_steps += len(dof)+1

            traj_dofs = np.stack(traj_dofs, axis=0)
            traj_eefs = np.stack(traj_eefs, axis=0)

            stats = get_traj_stats(traj_dofs, traj_eefs)
            dof_actions.append(stats['dof_mean'])
            eef_actions.append(stats['eef_mean'])

            entry['Traj_Len'] = stats['length']
            entry['Traj_Steps'] = traj_steps

            metadata.append(entry)

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(f'{args.dataset_path}/metadata.csv')

def downscale_dataset(path):
    metadata = pd.read_csv(f'{path}/metadata.csv')
    new_metadata = []
    for (scene_cat, scene_idx), group in metadata.groupby(['Scene_Category', 'Scene_Idx']):
        task_indices = group['Task_Idx'].tolist()
        num_tasks = len(task_indices) // 3
        sample_indices = np.random.choice(task_indices, replace=False, size=(num_tasks,))

        sample_group = group.loc[group['Task_Idx'].isin(sample_indices)]
        new_metadata.append(sample_group)

    new_metadata = pd.concat(new_metadata)

    new_metadata.to_csv(f'{path}/metadata_10k.csv')


def copy_data_to_scratch_space(args, metadata):
    for i, row in metadata.iterrows():
        if os.path.exists(f'{args.scratch_path}/{row["Path"]}'):
            continue
        else:
            subdir = row["Path"].split('/')[0]
            if not os.path.exists(f'{args.scratch_path}/{subdir}'):
                os.makedirs(f'{args.scratch_path}/{subdir}')
            shutil.copyfile(f'{args.dataset_path}/{row["Path"]}', f'{args.scratch_path}/{row["Path"]}')
            print("Copy:", row["Path"])


def mp_copy_data_to_scratch_space(args):
    from multiprocessing import Process

    metadata = pd.read_csv(f'{args.dataset_path}/metadata_5k.csv')

    processes = []
    for i in range(10):
        p = Process(target=copy_data_to_scratch_space, args=(args, metadata[i::10]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=len(metadata[i::10]) * 5)

    shutil.copyfile(f'{args.dataset_path}/metadata_10k.csv', f'{args.scratch_path}/metadata_10k.csv')
    shutil.copyfile(f'{args.dataset_path}/metadata_5k.csv', f'{args.scratch_path}/metadata_5k.csv')
    shutil.copyfile(f'{args.dataset_path}/metadata_all.csv', f'{args.scratch_path}/metadata_all.csv')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--scratch-path', type=str)
    args = parser.parse_args()

    #generate_metadata(args, ['grasp_traj', 'fetch_traj'])
    #downscale_dataset('/home/beining/Desktop')
    mp_copy_data_to_scratch_space(args)






