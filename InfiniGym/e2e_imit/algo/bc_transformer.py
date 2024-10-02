
from collections import OrderedDict


import torch
import torch.nn as nn
from e2e_imit.algo.bc_mlp import PTD_BC_MLPGaussian, to_cuda
from e2e_imit.models.transformer_policy import PTDTransformerGMMActorNetwork, PTDTransformerGMMACTActorNetwork

# DDP Setup
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class PTD_BC_TransformerGMM(PTD_BC_MLPGaussian):

    def _create_networks(self):
        if self.algo_config["model_type"] == 'Transformer_GMM':
            self.nets = PTDTransformerGMMActorNetwork(self.obs_shapes, self.ac_dim, self.algo_config)
        elif self.algo_config["model_type"] == 'Transformer_GMM_ACT':
            self.nets = PTDTransformerGMMACTActorNetwork(self.obs_shapes, self.ac_dim, self.algo_config)
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

    def _get_net_input(self, batch):
        if 'actions' in batch:
            batch['actions'] = batch['actions'][:, 0]

        return batch

    def _forward_training(self, batch):
        batch = self._get_net_input(batch)
        predictions = self.nets(batch)

        arm_dists = self.nets.module.build_dist(predictions["mean"], predictions["scale"], predictions["logits"])
        gripper_ll = predictions["gripper"]

        predictions['arm_log_prob'] = arm_dists.log_prob(batch["actions"][..., :-1])
        predictions['arm_action'] = arm_dists.sample()
        predictions['gripper_log_prob'] = gripper_ll
        predictions['gripper_action'] = torch.argmax(gripper_ll, dim=-1)

        return predictions

    def get_action(self, batch, step):
        rank = self.device_infos['rank'] if self.global_config.train.use_ddp else self.device_infos[0]
        batch = to_cuda(batch, rank)
        batch = self._get_net_input(batch)
        with torch.no_grad():
            out = self.nets.module(batch)

        dists = self.nets.module.build_dist(out["mean"], out["scale"], out["logits"])
        action = dists.sample().detach()
        gripper = torch.argmax(out["gripper"], dim=-1)

        if self.ac_type == 'joint':
            action *= self.ac_scale
            action = batch['q'][:, -1, :-2] + action
        elif self.ac_type == 'osc':
            eef_pos = batch['eef_pos'][:, -1] + action[:, :3] * self.ac_scale['pos']
            eef_quat = self._axis_angle_to_quat(batch['eef_quat'][:, -1], action[:, 3:] * self.ac_scale['angle'])
            action = torch.cat([eef_pos, eef_quat], dim=-1)
        else:
            raise NotImplementedError

        gripper = gripper * 2. - 1
        action = torch.concat([action, gripper.reshape(-1, 1)], dim=-1)

        return action


class PTD_BC_TransformerGMM_ACT(PTD_BC_TransformerGMM):

    def __init__(self, algo_config, global_config, obs_key_shapes, ac_params, device_infos, ckpt_path):
        super().__init__(algo_config, global_config, obs_key_shapes, ac_params, device_infos, ckpt_path)

        self.action_chunks = []
        self.chunk_ratio = [
            self.algo_config["act"]["w_ratio"] ** i for i in range(self.algo_config["act"]["horizon"])
        ]

    def _get_net_input(self, batch):
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

        dists = self.nets.module.build_dist(out["mean"], out["scale"], out["logits"])
        action = dists.sample().detach()
        gripper = torch.argmax(out["gripper"], dim=-1)

        self.action_chunks.append(action.clone())
        if len(self.action_chunks) > self.algo_config["act"]["horizon"]:
            self.action_chunks = self.action_chunks[-self.algo_config["act"]["horizon"]:]

        chunk_ratio = 0. #to_torch(action, dtype=torch.float32, device=rank)
        chunk_action = torch.zeros_like(action[:, 0])
        for s in range(len(self.action_chunks)):
            chunk_action += self.action_chunks[len(self.action_chunks)-s-1][:, s] * self.chunk_ratio[s]
            chunk_ratio += self.chunk_ratio[s]

        action = chunk_action / chunk_ratio

        if self.ac_type == 'joint':
            action *= self.ac_scale
            action = batch['q'][:, -1, :-2] + action
        elif self.ac_type == 'osc':
            eef_pos = batch['eef_pos'][:, -1] + action[:, :3] * self.ac_scale['pos']
            eef_quat = self._axis_angle_to_quat(batch['eef_quat'][:, -1], action[:, 3:] * self.ac_scale['angle'])
            action = torch.cat([eef_pos, eef_quat], dim=-1)
        else:
            raise NotImplementedError

        gripper = gripper * 2. - 1
        action = torch.concat([action, gripper.reshape(-1, 1)], dim=-1)

        return action