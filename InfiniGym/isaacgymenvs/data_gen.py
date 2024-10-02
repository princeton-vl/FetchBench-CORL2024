
import os
from datetime import datetime
import pandas as pd

import imageio.v3 as iio
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig


def log_data(path, trajs, scene, task_id):
    for i, traj in enumerate(trajs):
        np.save(f'{path}/task_{task_id}_traj{i}.npy', traj)
    np.save(f'{path}/task_{task_id}_init_scene.npy', scene)


def datagen_hydra(task_config_path, num_tasks, cfg):

    import logging

    import isaacgym
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    scene_name = task_config_path.split('/')[-1]
    cfg.scene.name = scene_name
    cfg.task.task.scene_config_path = ListConfig([task_config_path])
    cfg.scene.num_tasks = num_tasks

    experiment_name = f'{scene_name}_{cfg.task.name}_{cfg.task.prefix}_{time_str}'
    experiment_dir = os.path.join(cfg.task.data_path, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    # Load Eval Env Set
    cfg.task.experiment_name = experiment_name

    # Create Vectorized Env
    vec_env = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg
    )

    for i in range(min(cfg.scene.num_tasks, cfg.task.max_tasks)):
        vec_env.reset_task(i)
        trajs, scene = vec_env.solve()

        trajs = vec_env.save_trajs(trajs, scene)
        log_data(f'{experiment_dir}', trajs, scene, task_id=i)

    exit()


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def multiprocess_label_grasp_poses(cfg: DictConfig):
    from multiprocessing import Process

    metadata = pd.read_csv(f'{cfg.task.path}/metadata.csv')
    metadata = metadata.loc[metadata['Category'] == cfg.task.category]

    bs = max(1, len(metadata) // cfg.task.num_batch)
    batch_init_idx = min(bs * cfg.task.batch_idx, len(metadata))
    batch_end_idx = min(bs * (cfg.task.batch_idx + 1), len(metadata))

    metadata = metadata.iloc[batch_init_idx:batch_end_idx]
    prefix = cfg.task.path.split('/')[-1]

    for i, b in metadata.iterrows():
        ntasks = min(b['NumTasks'], 90)
        p = Process(target=datagen_hydra, args=(f'{prefix}/{b["Path"]}', ntasks, cfg))
        p.start()
        p.join(timeout=ntasks * (cfg.task.solution.num_max_trajs * 900 + 600))


if __name__ == "__main__":
    multiprocess_label_grasp_poses()
