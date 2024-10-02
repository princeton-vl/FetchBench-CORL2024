
import yaml
import hydra
import sys

from datetime import datetime

import isaacgym
import gym
import isaacgymenvs
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import numpy as np
import torch
import random
import wandb
import os
from torch.utils.data import DataLoader
import attrdict
from collections import OrderedDict
from datetime import timedelta

import e2e_imit.utils.train_utils as TrainUtils
from e2e_imit.utils.dataset import SequenceDataset

# Different Method
from e2e_imit.algo.bc_mlp import PTD_BC_MLPGaussian, PTD_BC_MLPGaussian_ACT
from e2e_imit.algo.bc_transformer import PTD_BC_TransformerGMM, PTD_BC_TransformerGMM_ACT
import imageio.v3 as iio

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def setup(rank, world_size, port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    timeout = timedelta(minutes=120)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup():
    dist.destroy_process_group()


def log_videos(path, idx, videos, fps=10):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    iio.imwrite(f'{path}/log_{idx}.mp4', np.stack(videos, axis=0), fps=fps)


def get_eval_env(cfg):
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # Load Eval Env Set
    cfg.task.task.scene_config_path = cfg.scene.scene_list

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

    return vec_env


def load_ckpt_config(path):
    config = {}
    config_file = f'{path}/config.yaml'
    with open(config_file, 'r') as f:
        config_file = yaml.safe_load(f)

    for k, v in config_file.items():
        config[k] = v['value']

    config["train"]["ckpt_path"] = f'{path}/model.pth'
    return config


def ddp_run(rank, world_size, cfg, train_config, log_dir, ckpt_dir, video_dir, port):
    assert train_config.train.use_ddp == True

    if rank == 0:  # only rank=0 log
        wandb.init(
            project=train_config.wandb_project_name,
            name=train_config.train.name,
            config=dict(train_config),
            dir=log_dir
        )

    print("\n============= Training Dataset =============")
    train_dataset = SequenceDataset(**train_config.dataset, obs_shape=train_config.algo.obs_shape,
                                    two_phase=train_config.algo.two_phase,
                                    action_horizon=train_config.algo.act.horizon)

    print("Length of the dataset: ", len(train_dataset))

    setup(rank, world_size, port)
    device_infos = {'rank': rank, 'world_size': world_size}

    if train_config.algo.model_type == 'MLP_Gaussian':
        algo = PTD_BC_MLPGaussian(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=device_infos,
            ckpt_path=ckpt_dir
        )
    elif train_config.algo.model_type == 'MLP_Gaussian_ACT':
        algo = PTD_BC_MLPGaussian_ACT(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=device_infos,
            ckpt_path=ckpt_dir
        )
    elif train_config.algo.model_type == 'Transformer_GMM':
        train_config["algo"]["transformer"]["context_length"] = train_config.dataset.seq_length
        algo = PTD_BC_TransformerGMM(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=device_infos,
            ckpt_path=ckpt_dir
        )
    elif train_config.algo.model_type == 'Transformer_GMM_ACT':
        train_config["algo"]["transformer"]["context_length"] = train_config.dataset.seq_length
        algo = PTD_BC_TransformerGMM_ACT(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=device_infos,
            ckpt_path=ckpt_dir
        )
    else:
        raise NotImplementedError

    if cfg["task"]["env"]["eval_policy"] and rank == 0:
        vec_env = get_eval_env(cfg)
        vec_env.update_algo(algo)

        def eval_algo_func(epoch):
            # Eval Env
            success = []
            for i in range(0, cfg.scene.num_tasks, cfg.task.env.eval_task_skip):
                vec_env.reset_task(i)
                rgb, log = vec_env.solve()
                res = vec_env.eval()
                success.append(res['success'])

                log_videos(f'{video_dir}/epoch_{epoch}', i, rgb, fps=24)

            np.save(f'{video_dir}/epoch_{epoch}/result.npy', np.stack(success, axis=0))
            return np.stack(success, axis=0)
        algo.fit_ddp(train_dataset, eval_algo_func)
    else:
        algo.fit_ddp(train_dataset, None)

    cleanup()


def dp_run(world_size, cfg, train_config, log_dir, ckpt_dir, video_dir):
    assert train_config.train.use_ddp == False

    wandb.init(
        project=train_config.wandb_project_name,
        name=train_config.train.name,
        config=dict(train_config),
        dir=log_dir
    )

    print("\n============= Training Dataset =============")
    train_dataset = SequenceDataset(**train_config.dataset, obs_shape=train_config.algo.obs_shape,
                                    two_phase=train_config.algo.two_phase,
                                    action_horizon=train_config.algo.act.horizon)
    print("Length of the dataset: ", len(train_dataset))

    if train_config.algo.model_type == 'MLP_Gaussian':
        algo = PTD_BC_MLPGaussian(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=range(world_size),
            ckpt_path=ckpt_dir
        )
    elif train_config.algo.model_type == 'MLP_Gaussian_ACT':
        algo = PTD_BC_MLPGaussian_ACT(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=range(world_size),
            ckpt_path=ckpt_dir
        )
    elif train_config.algo.model_type == 'Transformer_GMM':
        train_config["algo"]["transformer"]["context_length"] = train_config.dataset.seq_length
        algo = PTD_BC_TransformerGMM(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=range(world_size),
            ckpt_path=ckpt_dir
        )

    elif train_config.algo.model_type == 'Transformer_GMM_ACT':
        train_config["algo"]["transformer"]["context_length"] = train_config.dataset.seq_length
        algo = PTD_BC_TransformerGMM_ACT(
            algo_config=train_config.algo,
            global_config=train_config,
            obs_key_shapes=train_dataset.get_obs_shape(),
            ac_params=train_dataset.get_action_params(),
            device_infos=range(world_size),
            ckpt_path=ckpt_dir
        )
    else:
        raise NotImplementedError

    if cfg["task"]["env"]["eval_policy"]:
        vec_env = get_eval_env(cfg)
        vec_env.update_algo(algo)

        def eval_algo_func(epoch):
            # Eval Env
            success = []
            for i in range(0, cfg.scene.num_tasks, cfg.task.env.eval_task_skip):
                vec_env.reset_task(i)

                rgb, log = vec_env.solve()
                res = vec_env.eval()
                success.append(res['success'])

                log_videos(f'{video_dir}/epoch_{epoch}', i, rgb, fps=24)

            np.save(f'{video_dir}/epoch_{epoch}/result.npy', np.stack(success, axis=0))
            return np.stack(success, axis=0)

        algo.fit_dp(train_dataset, eval_algo_func)
    else:
        algo.fit_dp(train_dataset, None)


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def train(cfg):
    if not cfg["task"]["solution"]["ckpt_path"] is None:
        train_config = load_ckpt_config(cfg["task"]["solution"]["optimus"]["ckpt_path"])
    else:
        train_config = attrdict.AttrDict(cfg["task"]["solution"]["config"])

    port = str(int(np.random.randint(12000, 18000)))  # before set_seed
    set_seed(train_config.train.seed)

    log_dir, ckpt_dir, video_dir, time_str = TrainUtils.get_exp_dir(train_config, True)
    print("\n============= New Training Run with Config =============")
    print(train_config)
    print("")

    world_size = torch.cuda.device_count()
    if train_config.train.use_ddp:
        print("Port Num:", port)
        mp.spawn(
            ddp_run,
            args=(world_size, cfg, train_config, log_dir, ckpt_dir, video_dir, port),
            nprocs=world_size
        )
    else:
        dp_run(world_size, cfg, train_config, log_dir, ckpt_dir, video_dir)


if __name__ == "__main__":
    train()
