
import textwrap
from collections import OrderedDict

import e2e_imit.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn

from robomimic.models.distributions import TanhWrappedDistribution
from torch.nn import functional as F

from e2e_imit.models.cabi_encs import ObsEncoder


class PTDMLPGaussianActorNetwork(nn.Module):
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        config,
        **kwargs
    ):
        super(PTDMLPGaussianActorNetwork, self).__init__()

        self.ac_dim = ac_dim

#        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes  # dummy variable

        self.obs_group_shapes = OrderedDict({
            'q': (9,),
            'eef_pos': (3,),
            'eef_quat': (4,),
            'point_cloud': {
                'scene': (obs_shapes['visual']['scene'], 3),
                'robot': (obs_shapes['visual']['robot'], 3),
                'goal': (obs_shapes['visual']['goal'], 3)
            }
        })

        # config
        self.config = config

        if config.two_phase:
            self.obs_group_shapes['phase_index'] = (2,)

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObsEncoder(obs_shapes=self.obs_group_shapes, config=config["encoder"])

        if config.two_phase:
            self.nets['phase_embedding'] = nn.Linear(2, config["encoder"]["phase_index_embedding"])

        self.obs_encoder_output_dim = self.nets["encoder"].get_output_size()
        self.layer_dims = config["mlp"]["layer_dims"]

        input_dim = self.obs_encoder_output_dim

        if config.two_phase:
            input_dim += config["encoder"]["phase_index_embedding"]

        mlp_list = []
        curr_dim = input_dim
        for l in self.layer_dims:
            mlp_list.append(nn.Linear(curr_dim, l))
            mlp_list.append(nn.BatchNorm1d(l))
            mlp_list.append(nn.GELU())
            curr_dim = l

        self.nets["mlp"] = nn.Sequential(*mlp_list)

        # decoder for output modalities
        self._create_output_networks(dim=curr_dim)

        # build distribution
        self.use_tanh = config["gaussian"]["use_tanh"]
        self.std_limits = [0.001, 10.0]
        self.std_activation = F.softplus

    def _create_output_networks(self, dim):
        self.nets["decoder"] = nn.ModuleDict({
            'mean': nn.Linear(dim, self.ac_dim-1),
            'scale': nn.Linear(dim, self.ac_dim-1),
            'gripper': nn.Sequential(*[nn.Linear(dim, 2), nn.LogSoftmax()])
        })

    def build_dist(self, mean, scale):
        if not self.use_tanh:
            mean = torch.tanh(mean)

        # Calculate scale
        if not self.training:
            scale = torch.ones_like(mean) * 1e-4
        else:
            scale = self.std_activation(scale)
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])

        dist = D.Normal(loc=mean, scale=scale)
        dist = D.Independent(dist, 1)

        if self.use_tanh:
            dist = TanhWrappedDistribution(base_dist=dist, scale=1., epsilon=1e-2)

        return dist

    def forward(self, inputs):
        # add pt segment
        for i, k in enumerate(['scene', 'goal', 'robot']):
            seg = torch.eye(3)[i].to(inputs['visual'][k].device).unsqueeze(0).unsqueeze(0).repeat(*inputs['visual'][k].shape[:2], 1)
            inputs['visual'][k] = torch.cat([inputs['visual'][k], seg], dim=-1)
            # erase seg for [0, 0, 0] pts, which is information-less
            valid = inputs['visual'][k][..., :3].norm(dim=-1) > 1e-3
            inputs['visual'][k] = inputs['visual'][k] * valid.float().unsqueeze(-1)

        scene_pc = torch.concat([inputs['visual']['scene'], inputs['visual']['goal'], inputs['visual']['robot']], dim=-2)
        object_pc = torch.concat([inputs['visual']['goal'], inputs['visual']['robot']], dim=-2)
        proprio_st = torch.concat([inputs['q'], inputs['eef_pos'], inputs['eef_quat']], dim=-1)

        mlp_inputs = self.nets["encoder"].forward(scene_pc, object_pc, proprio_st).squeeze(dim=1)

        if self.config["two_phase"]:
            phase_embed = self.nets["phase_embedding"](inputs['phase_index'])
            mlp_inputs = torch.concat([mlp_inputs, phase_embed], dim=-1)

        decoder_outputs = self.nets["mlp"].forward(mlp_inputs)
        policy_outputs = dict()
        for k, layer in self.nets["decoder"].items():
            policy_outputs[k] = layer(decoder_outputs)

        return policy_outputs


class PTDMLPGaussianACTActorNetwork(PTDMLPGaussianActorNetwork):

    def _create_output_networks(self, dim):
        self.nets["decoder"] = nn.ModuleDict({
            'mean': nn.Linear(dim, (self.ac_dim-1) * self.config["act"]["horizon"]),
            'scale': nn.Linear(dim, (self.ac_dim-1) * self.config["act"]["horizon"]),
            'gripper': nn.Sequential(*[nn.Linear(dim, 2), nn.LogSoftmax()])
        })


    def forward(self, inputs):
        # add pt segment
        policy_outputs = super().forward(inputs)
        for k in ["mean", "scale"]:
            policy_outputs[k] = policy_outputs[k].reshape(-1, self.config["act"]["horizon"], self.ac_dim-1)

        return policy_outputs