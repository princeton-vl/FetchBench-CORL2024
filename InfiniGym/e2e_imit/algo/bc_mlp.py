
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import trimesh

import torch
import torch.nn as nn

from tqdm import tqdm
import wandb
import time
import os
import yaml
from omegaconf import OmegaConf

import trimesh.transformations as tra
import e2e_imit.utils.tensor_utils as TensorUtils
from e2e_imit.models.mlp_policy import PTDMLPGaussianActorNetwork, PTDMLPGaussianACTActorNetwork

import sys
sys.path.append('../third_party/Optimus')

import optimus.modules.functional as F
from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_from_angle_axis

# DDP Setup
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def to_cuda(batch, device):
    if isinstance(batch, dict) or isinstance(batch, OrderedDict):
        for k, v in batch.items():
            batch[k] = to_cuda(v, device)
        return batch
    else:
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        if isinstance(batch, float):
            batch = torch.tensor([batch], dtype=torch.float32)

        return batch.to(f'cuda:{device}')


class PTD_BC_MLPGaussian(object):

    def __init__(self, algo_config, global_config, obs_key_shapes, ac_params, device_infos, ckpt_path):
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.global_config = global_config

        self.ac_dim = ac_params['shape'][0]
        self.ac_type = ac_params['type']
        self.ac_scale = ac_params['scale']

        self.obs_shapes = obs_key_shapes

        self.device_infos = device_infos
        self.ckpt_path = ckpt_path

        self.optimizer_type = self.algo_config.optim_params.optimizer_type
        self.lr_scheduler_type = self.algo_config.optim_params.lr_scheduler_type
        self.epoch_every_n_steps = self.algo_config.optim_params.policy.learning_rate.epoch_every_n_steps
        self.num_epochs = self.global_config.train.num_epochs

        self._create_networks()
        self._create_optimizers()
        self._refresh_log()

    """
    Create Net and Optim
    """
    def _create_networks(self):
        if self.algo_config["model_type"] == 'MLP_Gaussian':
            self.nets = PTDMLPGaussianActorNetwork(self.obs_shapes, self.ac_dim, self.algo_config)
        elif self.algo_config["model_type"] == 'MLP_Gaussian_ACT':
            self.nets = PTDMLPGaussianACTActorNetwork(self.obs_shapes, self.ac_dim, self.algo_config)
        else:
            raise NotImplementedError

        if self.global_config.train.ckpt_path is not None:
            state_dict = torch.load(self.global_config.train.ckpt_path)
            self.deserialize(state_dict)

        if self.global_config.train.use_ddp:
            self.nets.to(self.device_infos['rank'])
            self.nets = DDP(self.nets, device_ids=[self.device_infos['rank']],
                            output_device=self.device_infos['rank'],
                            find_unused_parameters=True)
        else:
            # use Data parallel only, warning: gpu low utilization
            self.nets = nn.DataParallel(self.nets, self.device_infos)
            self.nets.to(f'cuda:{self.device_infos[0]}')

    def _create_optimizers(self):

        if self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.nets.parameters(),
                lr=self.optim_params["policy"]["learning_rate"]["initial"],
                betas=self.optim_params["policy"]["learning_rate"]["betas"],
                weight_decay=self.optim_params["policy"]["learning_rate"]["decay_factor"],
            )
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.AdamW(
                self.nets.parameters(),
                lr=self.optim_params["policy"]["learning_rate"]["initial"],
                betas=self.optim_params["policy"]["learning_rate"]["betas"],
            )

        if self.lr_scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs * self.epoch_every_n_steps,
                eta_min=self.optim_params["policy"]["learning_rate"]["initial"] * 1 / 10,
            )
        elif self.lr_scheduler_type == "none":
            self.lr_scheduler = None
        else:
            raise NotImplementedError

    """
    Train Step
    """

    def _get_net_input(self, batch):
        batch['q'] = batch['q'][:, -1]
        batch['eef_quat'] = batch['eef_quat'][:, -1]
        batch['eef_pos'] = batch['eef_pos'][:, -1]
        batch['rigid_pos'] = batch['rigid_pos'][:, -1]
        batch['rigid_quat'] = batch['rigid_quat'][:, -1]
        for s in ['scene', 'goal', 'robot']:
            batch['visual'][s] = batch['visual'][s][:, -1]

        if 'actions' in batch:
            batch['actions'] = batch['actions'][:, 0]
        if 'phase_index' in batch:
            batch['phase_index'] = batch['phase_index'][:, -1]

        return batch

    def _forward_training(self, batch):
        batch = self._get_net_input(batch)
        predictions = self.nets(batch)

        arm_dists = self.nets.module.build_dist(predictions["mean"], predictions["scale"])
        gripper_ll = predictions["gripper"]

        predictions['arm_log_prob'] = arm_dists.log_prob(batch["actions"][..., :-1])
        predictions['arm_action'] = arm_dists.sample()
        predictions['gripper_log_prob'] = gripper_ll
        predictions['gripper_action'] = torch.argmax(gripper_ll, dim=-1)

        return predictions

    def _compute_losses(self, predictions, batch):
        g_Loss = nn.NLLLoss()
        loss_log = OrderedDict()

        arm_nll = - predictions['arm_log_prob'].mean()
        gripper_label = (batch['actions'][:, -1] > 0.).long()
        gripper_nll = g_Loss(predictions['gripper_log_prob'], gripper_label).mean()

        loss = arm_nll + self.algo_config.optim_params.gripper_ratio * gripper_nll
        loss_log[f'arm_nll'] = arm_nll
        loss_log[f'gripper_nll'] = gripper_nll
        loss_log[f'loss'] = loss

        return loss_log

    def _train_step(self, losses, max_grad_norm=None):
        # gradient step
        info = OrderedDict()

        optim = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss = losses["loss"]
        net = self.nets

        # backprop
        optim.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

        # compute grad norms
        grad_norms = 0.0
        for p in net.parameters():
            #only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                grad_norms += p.grad.data.norm(2).pow(2).item()
        info["policy_grad_norms"] = grad_norms

        optim.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        return info

    def train_on_batch(self, batch):
        # train iter
        self.set_train()

        predictions = self._forward_training(batch)
        losses = self._compute_losses(predictions, batch)

        info = self._train_step(losses, self.algo_config.optim_params.max_grad_norm)

        info["predictions"] = TensorUtils.detach(predictions)
        info["losses"] = TensorUtils.detach(losses)
        info["actions"] = batch['actions']

        self.set_eval()

        return info

    """
    Log
    """

    def _refresh_log(self):
        self.training_log = {}
        for n in range(self.ac_dim-1):
            self.training_log[f'arm_error_dim_{n}'] = []

        self.training_log['loss'] = []
        self.training_log['arm_nll'] = []
        self.training_log['gripper_nll'] = []
        self.training_log['grad_norms'] = []
        self.training_log['arm_actions'] = []
        self.training_log['gripper_error'] = []

    def _dump_log(self, epoch):
        log = OrderedDict()
        for k, v in self.training_log.items():
            if k.endswith('nll') or k.endswith('loss'):
                if len(v) > 0:
                    log[f"train/{k}"] = np.array(self.training_log[k]).mean()
            if k.startswith('eval'):
                log[k] = np.array(v).mean()

        log["train/grad_norm"] = np.array(self.training_log['grad_norms']).mean()
        log["train/epoch"] = epoch
        log["train/lr"] = self.optimizer.param_groups[0]['lr']

        m = 0.
        for n in range(self.ac_dim-1):
            err = np.array(self.training_log[f"arm_error_dim_{n}"]).mean()
            m += err
            log[f"arm_error_dim_{n}"] = err

        log["gripper_error"] = np.array(self.training_log[f"gripper_error"]).mean()

        log[f"train/arm_error_mean"] = m / self.ac_dim
        log["train/arm_actions"] = wandb.Histogram(self.training_log['arm_actions'], num_bins=20)
        for k, v in self.training_log.items():
            if k.startswith('time'):
                log[k] = np.array(v).mean()

        wandb.log(log)

        self._refresh_log()

    def _update_info(self, info, epoch):
        self.training_log["loss"].append(info["losses"]["loss"].item())
        self.training_log["grad_norms"].append(info["policy_grad_norms"])

        self.training_log["arm_nll"].append(info['losses']["arm_nll"].item())
        self.training_log["gripper_nll"].append(info['losses']["gripper_nll"].item())

        # action error
        arm_error = []
        arm_error.extend([info['predictions']['arm_action'] - info["actions"][..., :-1]])
        arm_error = torch.concat(arm_error, dim=0)
        arm_error = torch.abs(arm_error.reshape(-1, arm_error.shape[-1])).mean(dim=0)

        for n in range(self.ac_dim-1):
            self.training_log[f'arm_error_dim_{n}'].append(arm_error[n].item())

        # terminate error
        self.training_log[f'gripper_error'].extend([
                    1. - torch.eq(info['actions'][:, -1] > 0.,  info["predictions"]['gripper_action'] > 0.5).float().mean().cpu().numpy()
            ])

        self.training_log[f'arm_actions'].extend(info['actions'][:, :-1].reshape(-1).cpu().numpy().tolist())

    """
    Rollout
    """

    def _axis_angle_to_quat(self, curr_quat, cmd_angle):
        cmd_theta = cmd_angle.norm(dim=-1)
        cmd_quat = quat_from_angle_axis(cmd_theta, cmd_angle)

        target_quat = quat_mul(cmd_quat, curr_quat)
        return target_quat

    def get_action(self, batch, step):
        rank = self.device_infos['rank'] if self.global_config.train.use_ddp else self.device_infos[0]
        batch = to_cuda(batch, rank)
        batch = self._get_net_input(batch)
        with torch.no_grad():
            out = self.nets.module(batch)

        dists = self.nets.module.build_dist(out["mean"], out["scale"])
        action = dists.sample().detach()
        gripper = torch.argmax(out["gripper"], dim=-1)

        if self.ac_type == 'joint':
            action *= self.ac_scale
            action = batch['q'][:, :-2] + action
        elif self.ac_type == 'osc':
            eef_pos = batch['eef_pos'] + action[:, :3] * self.ac_scale['pos']
            eef_quat = self._axis_angle_to_quat(batch['eef_quat'], action[:, 3:] * self.ac_scale['angle'])
            action = torch.cat([eef_pos, eef_quat], dim=-1)
        else:
            raise NotImplementedError

        gripper = gripper * 2. - 1
        action = torch.concat([action, gripper.reshape(-1, 1)], dim=-1)

        return action

    """
    Train
    """

    def fit_dp(self, dataset, eval_func=None):
        dataloader = torch.utils.data.DataLoader(dataset, **self.global_config.data_loader)

        for epoch in range(self.num_epochs):
            tr_start = time.time()

            b_time = time.time()
            for i, batch in enumerate(tqdm(dataloader)):
                f_time = time.time()
                batch = to_cuda(batch, self.device_infos[0])

                info = self.train_on_batch(batch)
                self._update_info(info, epoch)
                t_time = time.time()

                print("Fetch Time: ", f_time - b_time, "Train Time: ", t_time - f_time)
                b_time = t_time

            tr_end = time.time()
            self.training_log['time/training_time'] = tr_end - tr_start

            if epoch % self.global_config.train.eval_every_num_epochs == 0:
                state_dict = self.serialize()
                os.makedirs(f'{self.ckpt_path}/{epoch}', exist_ok=False)
                torch.save(state_dict, f'{self.ckpt_path}/{epoch}/model.pth')  # save model pth

                config = OmegaConf.create(dict(self.global_config))

                with open(f'{self.ckpt_path}/{epoch}/config.yaml', 'w') as config_yaml:
                    yaml.dump(OmegaConf.to_yaml(config), config_yaml)

                if eval_func is not None:
                    self.set_eval()
                    res = eval_func(epoch)
                    eval_success = res.mean(axis=0)

                    for i in range(len(eval_success)):
                        self.training_log[f'eval/scene_{i}_success'] = eval_success[i]

                    self.training_log['eval/all_scene_success'] = np.mean(eval_success)
                    self.training_log['eval/epoch'] = epoch

            self._dump_log(epoch)

    def fit_ddp(self, dataset, eval_func=None):
        sampler = DistributedSampler(dataset, num_replicas=self.device_infos['world_size'],
                                     rank=self.device_infos['rank'], shuffle=True, drop_last=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.global_config.data_loader.batch_size,
                                                 num_workers=self.global_config.data_loader.num_workers,
                                                 pin_memory=False, drop_last=True, shuffle=False, sampler=sampler)

        for epoch in range(self.num_epochs):
            dataloader.sampler.set_epoch(epoch)
            tr_start = time.time()

            b_time = time.time()
            for i, batch in enumerate(tqdm(dataloader)):
                f_time = time.time()
                batch = to_cuda(batch, self.device_infos['rank'])

                info = self.train_on_batch(batch)
                self._update_info(info, epoch)
                t_time = time.time()

                print("Fetch Time: ", f_time - b_time, "Train Time: ", t_time - f_time)
                b_time = t_time

            tr_end = time.time()
            self.training_log['time/training_time'] = [tr_end - tr_start]

            if epoch % self.global_config.train.eval_every_num_epochs == 0 and self.device_infos['rank'] == 0:
                state_dict = self.serialize()
                os.makedirs(f'{self.ckpt_path}/{epoch}', exist_ok=False)
                torch.save(state_dict, f'{self.ckpt_path}/{epoch}/model.pth')  # save model pth

                config = OmegaConf.create(dict(self.global_config))

                with open(f'{self.ckpt_path}/{epoch}/config.yaml', 'w') as config_yaml:
                    yaml.dump(OmegaConf.to_yaml(config), config_yaml)

                if eval_func is not None:
                    self.set_eval()

                    res = eval_func(epoch)
                    eval_success = res.mean(axis=0)

                    for i in range(len(eval_success)):
                        self.training_log[f'eval/scene_{i}_success'] = eval_success[i]

                    self.training_log['eval/all_scene_success'] = np.mean(eval_success)
                    self.training_log['eval/epoch'] = epoch

            logs = [None for _ in range(self.device_infos['world_size'])]
            # the first argument is the collected lists, the second argument is the data unique in each process
            dist.all_gather_object(logs, self.training_log)

            if self.device_infos['rank'] == 0:
                self.merge_ddp_logs(logs)
                self._dump_log(epoch)

            self._refresh_log()

    """
    Utils
    """

    def set_eval(self):
        self.nets.eval()

    def set_train(self):
        self.nets.train()

    def serialize(self):
        return self.nets.module.state_dict()

    def deserialize(self, model_dict):
        self.nets.load_state_dict(model_dict)

    def merge_ddp_logs(self, logs):
        n_logs = {}
        for k, v in self.training_log.items():
            if k not in n_logs:
                n_logs[k] = []
            for log in logs:
                if k in log:
                    if isinstance(log[k], list):
                        n_logs[k].extend(log[k])
                    else:
                        n_logs[k].append(log[k])

        self.training_log = n_logs


class PTD_BC_MLPGaussian_ACT(PTD_BC_MLPGaussian):

    def __init__(self, algo_config, global_config, obs_key_shapes, ac_params, device_infos, ckpt_path):
        super().__init__(algo_config, global_config, obs_key_shapes, ac_params, device_infos, ckpt_path)

        self.action_chunks = []
        self.chunk_ratio = [
            self.algo_config["act"]["w_ratio"] ** i for i in range(self.algo_config["act"]["horizon"])
        ]

    def _get_net_input(self, batch):
        batch['q'] = batch['q'][:, -1]
        batch['eef_quat'] = batch['eef_quat'][:, -1]
        batch['eef_pos'] = batch['eef_pos'][:, -1]
        batch['rigid_pos'] = batch['rigid_pos'][:, -1]
        batch['rigid_quat'] = batch['rigid_quat'][:, -1]
        for s in ['scene', 'goal', 'robot']:
            batch['visual'][s] = batch['visual'][s][:, -1]

        if 'phase_index' in batch:
            batch['phase_index'] = batch['phase_index'][:, -1]

        return batch


    def _compute_losses(self, predictions, batch):
        g_Loss = nn.NLLLoss()
        loss_log = OrderedDict()

        arm_nll = - predictions['arm_log_prob'].mean()
        gripper_label = (batch['actions'][:, 0, -1] > 0.).long()    # stop signal is not chunked
        gripper_nll = g_Loss(predictions['gripper_log_prob'], gripper_label).mean()

        loss = arm_nll + self.algo_config.optim_params.gripper_ratio * gripper_nll
        loss_log[f'arm_nll'] = arm_nll
        loss_log[f'gripper_nll'] = gripper_nll
        loss_log[f'loss'] = loss

        return loss_log

    def _update_info(self, info, epoch):
        self.training_log["loss"].append(info["losses"]["loss"].item())
        self.training_log["grad_norms"].append(info["policy_grad_norms"])

        self.training_log["arm_nll"].append(info['losses']["arm_nll"].item())
        self.training_log["gripper_nll"].append(info['losses']["gripper_nll"].item())

        # action error
        arm_error = []
        arm_error.extend([info['predictions']['arm_action'] - info["actions"][..., :-1]])
        arm_error = torch.concat(arm_error, dim=0)
        arm_error = torch.abs(arm_error.reshape(-1, arm_error.shape[-1])).mean(dim=0)

        for n in range(self.ac_dim-1):
            self.training_log[f'arm_error_dim_{n}'].append(arm_error[n].item())

        # terminate error
        self.training_log[f'gripper_error'].extend([
                    1. - torch.eq(info['actions'][:, 0, -1] > 0.,  info["predictions"]['gripper_action'] > 0.5).float().mean().cpu().numpy()
            ])

        self.training_log[f'arm_actions'].extend(info['actions'][:, :-1].reshape(-1).cpu().numpy().tolist())

    def get_action(self, batch, step):
        if step == 0:
            self.action_chunks = []

        rank = self.device_infos['rank'] if self.global_config.train.use_ddp else self.device_infos[0]
        batch = to_cuda(batch, rank)
        batch = self._get_net_input(batch)
        with torch.no_grad():
            out = self.nets.module(batch)

        dists = self.nets.module.build_dist(out["mean"], out["scale"])
        action = dists.sample().detach()
        gripper = torch.argmax(out["gripper"], dim=-1)

        self.action_chunks.append(action.clone())
        if len(self.action_chunks) > self.algo_config["act"]["horizon"]:
            self.action_chunks = self.action_chunks[-self.algo_config["act"]["horizon"]:]

        chunk_ratio = [] #to_torch(action, dtype=torch.float32, device=rank)
        chunk_action = torch.zeros_like(action[:, 0])
        for s in range(len(self.action_chunks)):
            chunk_action += self.action_chunks[len(self.action_chunks)-s-1][:, s] * self.chunk_ratio[s]
            chunk_ratio.append(self.chunk_ratio[s])

        action = chunk_action / torch.tensor(chunk_ratio, dtype=chunk_action.dtype, device=chunk_action.device).sum()

        if self.ac_type == 'joint':
            action *= self.ac_scale
            action = batch['q'][:, :-2] + action
        elif self.ac_type == 'osc':
            eef_pos = batch['eef_pos'] + action[:, :3] * self.ac_scale['pos']
            eef_quat = self._axis_angle_to_quat(batch['eef_quat'], action[:, 3:] * self.ac_scale['angle'])
            action = torch.cat([eef_pos, eef_quat], dim=-1)
        else:
            raise NotImplementedError

        gripper = gripper * 2. - 1
        action = torch.concat([action, gripper.reshape(-1, 1)], dim=-1)

        return action
