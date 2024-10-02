
import textwrap
from collections import OrderedDict

import e2e_imit.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
from attrdict import AttrDict

from robomimic.models.distributions import TanhWrappedDistribution
from torch.nn import functional as F

from e2e_imit.models.cabi_encs import ObsEncoder
from e2e_imit.models.gpt_transformer import GPT_Backbone


def transformer_args_from_config(transformer_config):
    return AttrDict(dict(
        transformer_num_layers=transformer_config.num_layers,
        transformer_context_length=transformer_config.context_length,
        transformer_embed_dim=transformer_config.embed_dim,
        transformer_num_heads=transformer_config.num_heads,
        transformer_embedding_dropout=transformer_config.embedding_dropout,
        transformer_block_attention_dropout=transformer_config.block_attention_dropout,
        transformer_block_output_dropout=transformer_config.block_output_dropout,
        transformer_block_drop_path=transformer_config.block_drop_path,
        layer_dims=transformer_config.layer_dims
    ))


class PTDTransformerGMMActorNetwork(nn.Module):
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        config,
        **kwargs
    ):
        super(PTDTransformerGMMActorNetwork, self).__init__()

        self.ac_dim = ac_dim
        self.obs_shapes = obs_shapes

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

        if config.two_phase:
            self.obs_group_shapes['phase_index'] = (2,)

        self.config = config

        self.num_modes = config.gmm.num_modes
        self.min_std = config.gmm.min_std
        self.low_noise_eval = config.gmm.low_noise_eval
        self.use_tanh = config.gmm.use_tanh

        # Define activations to use
        self.activations = {"softplus": F.softplus, "exp": torch.exp}
        self.std_activation = config.gmm.std_activation

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObsEncoder(obs_shapes=self.obs_group_shapes, config=config["encoder"])

        if config.two_phase:
            self.nets['phase_embedding'] = nn.Linear(2, config["encoder"]["phase_index_embedding"])

        input_dim = self.nets["encoder"].get_output_size()
        if config.two_phase:
            input_dim += config["encoder"]["phase_index_embedding"]

        self.params = nn.ParameterDict()
        tf_config = transformer_args_from_config(config.transformer)
        self.nets["embed_encoder"] = nn.Linear(input_dim, tf_config.transformer_embed_dim)
        self.params["embed_timestep"] = nn.Parameter(torch.zeros(1, tf_config.transformer_context_length,
                                                                 tf_config.transformer_embed_dim))

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(tf_config.transformer_embed_dim)
        self.nets["embed_drop"] = nn.Dropout(tf_config.transformer_embedding_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=tf_config.transformer_embed_dim,
            num_layers=tf_config.transformer_num_layers,
            num_heads=tf_config.transformer_num_heads,
            context_length=tf_config.transformer_context_length,
            block_attention_dropout=tf_config.transformer_block_attention_dropout,
            block_output_dropout=tf_config.transformer_block_output_dropout,
            block_drop_path=tf_config.transformer_block_drop_path,
        )

        self._create_output_networks(tf_config.transformer_embed_dim)

    def _create_output_networks(self, dim):
        self.nets["decoder"] = nn.ModuleDict({
            'mean': nn.Linear(dim, (self.ac_dim-1) * self.num_modes),
            'scale': nn.Linear(dim, (self.ac_dim-1) * self.num_modes),
            'logits': nn.Linear(dim, self.num_modes),
            'gripper': nn.Sequential(*[nn.Linear(dim, 2), nn.LogSoftmax(dim=-1)])
        })

    def input_embedding(self, inputs):
        v = self.nets["embed_encoder"](inputs)

        time_embeddings = self.params["embed_timestep"]
        v = v + time_embeddings
        v = self.nets["embed_ln"](v)
        embeddings = self.nets["embed_drop"](v)

        return embeddings

    def build_dist(self, means, scales, logits):
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            scales = torch.ones_like(means) * 1e-4
        else:
            scales = self.activations[self.std_activation](scales) + self.min_std

        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.0, epsilon=1e-2)

        return dists

    def _decoder_forward(self, outputs):
        policy_outputs = dict()
        for k, layer in self.nets["decoder"].items():
            out = layer(outputs)
            if k == 'mean' or k == 'scale':
                policy_outputs[k] = out.reshape(out.shape[0], self.num_modes, -1)
            elif k == 'logits':
                policy_outputs[k] = out.reshape(out.shape[0], self.num_modes)
            else:
                policy_outputs[k] = out

        return policy_outputs

    def forward(self, inputs):
        # add pt segment

        for i, k in enumerate(['scene', 'goal', 'robot']):
            seg = torch.eye(3)[i].to(inputs['visual'][k].device).reshape(1, 1, 1, 3).repeat(*inputs['visual'][k].shape[:3], 1)
            inputs['visual'][k] = torch.cat([inputs['visual'][k], seg], dim=-1)
            # erase seg for [0, 0, 0] pts, which is information-less
            valid = inputs['visual'][k][..., :3].norm(dim=-1) > 1e-3
            inputs['visual'][k] = inputs['visual'][k] * valid.float().unsqueeze(-1)

        scene_pc = torch.concat([inputs['visual']['scene'], inputs['visual']['goal'], inputs['visual']['robot']], dim=-2)
        object_pc = torch.concat([inputs['visual']['goal'], inputs['visual']['robot']], dim=-2)
        proprio_st = torch.concat([inputs['q'], inputs['eef_pos'], inputs['eef_quat']], dim=-1)

        bs, seq = inputs['q'].shape[:2]
        scene_pc = scene_pc.reshape(-1, *scene_pc.shape[-2:])
        object_pc = object_pc.reshape(-1, *object_pc.shape[-2:])
        proprio_st = proprio_st.reshape(bs * seq, -1)

        tf_inputs = self.nets["encoder"].forward(scene_pc, object_pc, proprio_st).squeeze(dim=1)

        if self.config["two_phase"]:
            phase_embed = self.nets["phase_embedding"](inputs['phase_index'].reshape(bs * seq, -1))
            tf_inputs = torch.concat([tf_inputs, phase_embed], dim=-1)

        tf_inputs = tf_inputs.reshape(bs, seq, -1)
        tf_embeddings = self.input_embedding(tf_inputs)
        tf_outputs = self.nets["transformer"].forward(tf_embeddings)
        tf_outputs = tf_outputs[:, -1]

        return self._decoder_forward(tf_outputs)


class PTDTransformerGMMACTActorNetwork(PTDTransformerGMMActorNetwork):

    def _create_output_networks(self, dim):
        self.nets["decoder"] = nn.ModuleDict({
            'mean': nn.Linear(dim, (self.ac_dim-1) * self.config["act"]["horizon"] * self.num_modes),
            'scale': nn.Linear(dim, (self.ac_dim-1) * self.config["act"]["horizon"] * self.num_modes),
            'logits': nn.Linear(dim, self.num_modes * self.config["act"]["horizon"]),
            'gripper': nn.Sequential(*[nn.Linear(dim, 2), nn.LogSoftmax()])
        })


    def _decoder_forward(self, outputs):
        policy_outputs = dict()
        for k, layer in self.nets["decoder"].items():
            out = layer(outputs)
            if k == 'mean' or k == 'scale':
                policy_outputs[k] = out.reshape(out.shape[0], self.config["act"]["horizon"], self.num_modes, -1)
            elif k == 'logits':
                policy_outputs[k] = out.reshape(out.shape[0], self.config["act"]["horizon"], self.num_modes)
            else:
                policy_outputs[k] = out

        return policy_outputs