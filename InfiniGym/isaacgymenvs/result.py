
import os
import numpy as np


def read_results(path, res_dict, **kwargs):
    res = np.load(f'{path}/result.npy', allow_pickle=True).tolist()

    for s in range(len(res['success'])):
        for k, v in res_dict.items():
            if k == 'computing_time' or k == 'traj_length':
                res_dict[k].extend(res['extra'][s][k])
            else:
                res_dict[k].extend(res[k][s])

    # sanity check
    res_len = len(res_dict['success'])
    for k, v in res_dict.items():
        if k == 'extra':
            continue
        else:
            assert len(v) == res_len

    return res_dict


def read_all_exp_results(dir_path, task_name, **kwargs):

    res_dict = {
        'success': [],
        'label': [],
        'z_threshold': [],
        'x_threshold': [],
        'e_threshold': [],
        'computing_time': [],
        'traj_length': []
    }

    config = 'benchmark_eval'

    not_founds = []
    for scene in os.listdir(f'config/scene/{config}'):
        file = f'{scene[:-5]}_{task_name}'
        found = False
        for f in os.listdir(dir_path):
            try:
                if f.startswith(file):
                    if found:
                        continue
                    res_dict = read_results(f'{dir_path}/{f}', res_dict, **kwargs)
                    found = True
            except:
                continue

        if not found:
            not_founds.append(scene[:-5])

    s = ''
    for f in not_founds:
        s += f + ' '

    print("Result Not Found: ", s)
    return res_dict


def summarize_experiments(res_dict, key='success'):
    rigid_task_labels = {
        'on_table': [],
        'on_shelf': [],
        'in_basket': [],
        'in_drawer': [],
        'total': []
    }

    for i in range(len(res_dict['label'])):
        for k, v in rigid_task_labels.items():
            if key in ['success', 'label', 'z_threshold', 'x_threshold', 'e_threshold', 'computing_time', 'traj_length']:
                if k in res_dict['label'][i]:
                    rigid_task_labels[k].append(res_dict[key][i])
                    rigid_task_labels['total'].append(res_dict[key][i])
            else:
                if k in res_dict['label'][i]:
                    rigid_task_labels[k].append(res_dict['extra'][key][i])
                    rigid_task_labels['total'].append(res_dict['extra'][key][i])

    for k, v in rigid_task_labels.items():
        mean = np.array(v).mean()
        count = len(v)
        print(k, count, mean)


def compare_baselines(result_dir, task_list, sum_key, **kwargs):
    for t in task_list:
        print(t)
        res_dict = read_all_exp_results(result_dir, t, **kwargs)
        summarize_experiments(res_dict, sum_key)


if __name__ == "__main__":
    task = [
            #'FetchMeshCurobo_Release',
            #'FetchPtdCuroboCGNBeta_Release',
            #'FetchPtdPyomplCGNBeta_Release',
            #'FetchPtdCabinetCGNBeta_Release'
            #'FetchPtdImitE2E_Release'
            ]
    compare_baselines('../runs', task, 'success')