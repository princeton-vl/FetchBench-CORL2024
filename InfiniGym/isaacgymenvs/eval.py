
import hydra
from omegaconf import DictConfig, OmegaConf

import logging
import os
from datetime import datetime

import isaacgym
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import isaacgymenvs
import imageio.v3 as iio
import numpy as np


def log_videos(path, idx, videos, fps=10):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    iio.imwrite(f'{path}/log_{idx}.mp4', np.stack(videos, axis=0), fps=fps)


def log_results(path, results):
    count, success = 0, 0

    log = {
        'z_threshold': [],
        'x_threshold': [],
        'e_threshold': [],
        'success': [],
        'label': [],
        'task_repeat': [],
        'extra': []
    }

    for i, res in enumerate(results):
        count += np.product(*np.array(res['success']).shape)
        success += np.array(res['success']).astype(np.float32).sum()

        for k, v in res.items():
            log[k].append(v)

    print("Success Rate: ", success / count)

    np.save(f'{path}/result.npy', log)


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def launch_eval_hydra(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    experiment_name = f'{cfg.scene.name}_{cfg.task.name}_{cfg.task.prefix}_{time_str}'
    experiment_dir = os.path.join('runs', experiment_name)
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
    cfg.task.task.scene_config_path = cfg.scene.scene_list
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

    if cfg.task.name.startswith('FetchPtdOptimus'):
        from isaacgymenvs.tasks.fetch.utils.optimus_utils import load_optimus_algo
        algo = load_optimus_algo(cfg["task"])
        vec_env.update_algo(algo)
    if cfg.task.name.startswith('FetchPtdImit'):
        from isaacgymenvs.tasks.fetch.utils.imit_utils import load_imitation_algo
        algo = load_imitation_algo(cfg["task"]["solution"]["ckpt_path"])
        vec_env.update_algo(algo)

    # Eval Env
    results, logs = [], []
    for i in range(cfg.scene.num_tasks):
        vec_env.reset_task(i)
        rgb, log = vec_env.solve()
        res = vec_env.eval()

        res['extra'] = log
        results.append(res)

        log_videos(f'./videos/{experiment_name}', i, rgb, fps=24)

    # Log Results
    log_results(f'./runs/{experiment_name}', results)

    vec_env.exit()
    exit()


if __name__ == "__main__":
    launch_eval_hydra()
