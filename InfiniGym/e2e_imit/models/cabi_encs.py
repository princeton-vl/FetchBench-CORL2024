# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Third Party
import numpy as np
import torch
import torch.nn as nn
from torch import nn
from torch.cuda.amp import autocast

import sys
sys.path.append('../third_party/cabinet')

# NVIDIA
from cabi_net.errors import VoxelOverflowError
from pointnet2.pointnet2_modules import PointnetSAModule
from pointnet2.pytorch_utils import FC, Conv1d, Conv3d
from copy import deepcopy


def break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
    return xyz, features


class PointNet2Encoder(nn.Module):
    # config of pointnet encoder
    PN_NPOINTS = [256, 64, 16, None]
    PN_RADII = [0.1, 0.2, 0.4, None]
    PN_NSAMPLES = [64, 64, 32, None]
    PN_MLPS = [[6, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]]

    def __init__(self, activation="relu", bn=False, output_size=128):
        super().__init__()
        self.activation = activation
        self.obj_SA_modules = nn.ModuleList()
        for k in range(self.PN_NPOINTS.__len__()):
            self.obj_SA_modules.append(
                PointnetSAModule(
                    npoint=deepcopy(self.PN_NPOINTS[k]),
                    radius=deepcopy(self.PN_RADII[k]),
                    nsample=deepcopy(self.PN_NSAMPLES[k]),
                    mlp=deepcopy(self.PN_MLPS[k]),
                    use_xyz=True,
                    activation=self.activation,
                    bn=bn,
                    first=False,
                )
            )

        self.FCs = nn.Sequential(
            *[
                FC(self.PN_MLPS[-1][-1], 1024, activation=self.activation),
                nn.Linear(1024, output_size),
                nn.BatchNorm1d(output_size),
                #nn.ReLU()
            ]
        )

    def forward(self, pc):
        xyz, features = break_up_pc(pc)

        # Featurize obj, add xyz to features
        features = torch.concat([features, xyz.transpose(1, 2)], dim=-2)
        for i in range(len(self.obj_SA_modules)):
            xyz, features = self.obj_SA_modules[i](xyz, features)

        features = features.squeeze(axis=-1)
        outputs = self.FCs(features)

        return outputs

class PointNet2EncoderGamma(PointNet2Encoder):
    PN_NPOINTS = [256, 128, 64, 16, 4, None]
    PN_RADII = [0.05, 0.1, 0.2, 0.4, 0.8, None]
    PN_NSAMPLES = [64, 64, 32, 32, 32, None]
    PN_MLPS = [[6, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024], [1024, 1024, 2048], [2048, 2048, 4096]]

class VoxelEncoder(nn.Module):
    SCENE_PT_MLP = [9, 128, 256]
    SCENE_VOX_MLP = [256, 512, 1024, 2048, 2048]
    def __init__(
        self, bounds, vox_size, activation="relu", bn=False, output_size=128
    ):
        super().__init__()
        self.bounds = bounds
        self.vox_size = vox_size
        self.num_voxels = ((self.bounds[1] - self.bounds[0]) / self.vox_size).astype(np.int32)
        self.activation = activation

        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(self.SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                "pt_layer{}".format(i),
                Conv1d(self.SCENE_PT_MLP[i], self.SCENE_PT_MLP[i + 1], bn=bn, activation=self.activation, first=(i == 0)),
            )

        self.scene_vox_mlp = nn.ModuleList()

        n_voxels = self.num_voxels.copy()
        for i in range(len(self.SCENE_VOX_MLP) - 1):
            scene_conv = nn.Sequential()
            scene_conv.add_module(
                "3d_conv_layer{}".format(i),
                Conv3d(self.SCENE_VOX_MLP[i], self.SCENE_VOX_MLP[i + 1], kernel_size=3, padding=1,
                       bn=bn, activation=self.activation),
            )
            scene_conv.add_module("3d_max_layer{}".format(i), nn.MaxPool3d(2, stride=2))

            self.scene_vox_mlp.append(scene_conv)
            n_voxels = n_voxels // 2

        embedding_size = self.SCENE_VOX_MLP[-1] * np.product(n_voxels)

        self.last_layer = nn.Sequential(
            nn.Linear(embedding_size, output_size),
            nn.BatchNorm1d(output_size),
            #nn.ReLU()
        )

    def filter_pts(self, ptc):
        bounds = torch.from_numpy(self.bounds).to(ptc.device)
        valid = (ptc[..., :3] >= bounds[0] + 1e-3).int().prod(dim=-1) * (ptc[..., :3] < bounds[1] - 1e-3).int().prod(dim=-1)
        ptc = ptc * valid.unsqueeze(-1)
        return ptc

    def _inds_to_flat(self, inds, scale=1):
        self.scale = scale
        num_voxels = torch.from_numpy(self.num_voxels).to(device=inds.device)
        self._flat_tensor = torch.tensor(
            [num_voxels[1:].prod() // (self.scale**2), num_voxels[2] // self.scale, 1],
            device=inds.device, dtype=torch.int,
        )
        flat_inds = inds * self._flat_tensor
        return flat_inds.sum(axis=-1)

    def _inds_from_flat(self, flat_inds, scale=1):
        num_voxels = torch.from_numpy(self.num_voxels).to(device=flat_inds.device)
        ind0 = flat_inds // (num_voxels[1:].prod() // (scale**2))
        ind1 = (flat_inds % (num_voxels[1:].prod() // (scale**2))) // (num_voxels[2] // scale)
        ind2 = (flat_inds % (num_voxels[1:].prod() // (scale**2))) % (num_voxels[2] // scale)

        return torch.stack((ind0, ind1, ind2), dim=-1)

    def voxel_inds(self, xyz, scale=1):
        bounds = torch.from_numpy(self.bounds[0]).to(device=xyz.device)
        vox_size = torch.from_numpy(self.vox_size).to(device=xyz.device)
        num_voxels = torch.from_numpy(self.num_voxels).to(device=xyz.device)

        inds = ((xyz - bounds) // (scale * vox_size)).int()

        # check validity
        valid = (inds >= 0).prod(dim=-1) & (inds < num_voxels).prod(dim=-1)

        assert ~torch.any(~valid.bool()), "Error: out of voxel space."
        return self._inds_to_flat(inds, scale=scale)

    def forward(self, scene_pc):
        scene_xyz, scene_features = break_up_pc(scene_pc)

        bounds = torch.from_numpy(self.bounds).to(device=scene_xyz.device)
        vox_size = torch.from_numpy(self.vox_size).to(device=scene_xyz.device)
        num_voxels = torch.from_numpy(self.num_voxels).to(device=scene_xyz.device)

        scene_inds = self.voxel_inds(scene_xyz)

        # Featurize scene points and max pool over voxels
        scene_vox_centers = self._inds_from_flat(scene_inds) * vox_size + vox_size / 2 + bounds[0]
        scene_xyz_centered = (scene_pc[..., :3] - scene_vox_centers).transpose(2, 1)
        if scene_features is not None:
            scene_features = self.scene_pt_mlp(torch.cat((scene_xyz_centered, scene_vox_centers.transpose(2, 1), scene_features), dim=1))
        else:

            scene_features = self.scene_pt_mlp(scene_xyz_centered)

        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], num_voxels.prod()), device=scene_features.device
        )
        if scene_inds.max() >= num_voxels.prod():
            print(
                scene_xyz[range(len(scene_pc)), scene_inds.max(axis=-1)[1]],
                scene_inds.max(),
            )
        assert scene_inds.max() < num_voxels.prod()
        assert scene_inds.min() >= 0

        with autocast(enabled=False):
            # Third Party
            import torch_scatter

            max_vox_features[..., : scene_inds.max() + 1] = torch_scatter.scatter_max(
                scene_features.float(), scene_inds[:, None, :]
            )[0]

        max_vox_features = max_vox_features.reshape(*max_vox_features.shape[:2], *num_voxels.int())

        # 3D conv over voxels
        l_vox_features = [max_vox_features]
        for i in range(len(self.scene_vox_mlp)):
            li_vox_features = self.scene_vox_mlp[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)

        # Stack features from different levels
        output = self.last_layer(l_vox_features[-1].reshape(max_vox_features.shape[0], -1))
        return output

class VoxelEncoderGamma(VoxelEncoder):
    SCENE_PT_MLP = [9, 128, 256, 512]
    SCENE_VOX_MLP = [512, 1024, 2048, 4096]

class ProprioEncoder(nn.Module):
    Layer_Dims = [128, 256]
    def __init__(self, input_size, output_size):
        super().__init__()

        model = []

        in_ = input_size
        for l in self.Layer_Dims:
            model.extend([
                nn.Linear(in_, l),
                nn.BatchNorm1d(l),
                nn.ReLU()
            ])
            in_ = l

        model.extend([nn.Linear(in_, output_size), nn.BatchNorm1d(output_size),
                      #nn.ReLU()
                    ])
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ProprioEncoderGamma(ProprioEncoder):
    Layer_Dims = [256, 512, 1024]

class ObsEncoder(nn.Module):
    def __init__(self, config, obs_shapes):
        super().__init__()
        self.config = config

        self.nets = nn.ModuleDict()

        if self.config["use_voxel_encoder"]:
            bounds = np.asarray(config["voxel"]["bounds"]).astype(np.float32)
            vox_size = np.asarray(config["voxel"]["vox_size"]).astype(np.float32)

            if not self.config["use_gamma"]:
                self.nets["voxel_encoder"] = VoxelEncoder(
                    bounds=bounds, vox_size=vox_size, output_size=config["voxel"]["output_size"],
                    activation=config["voxel"]["activation"], bn=config["voxel"]["bn"]
                )
            else:
                self.nets["voxel_encoder"] = VoxelEncoderGamma(
                    bounds=bounds, vox_size=vox_size, output_size=config["voxel"]["output_size"],
                    activation=config["voxel"]["activation"], bn=config["voxel"]["bn"]
                )

        if self.config["use_points_encoder"]:
            if not self.config["use_gamma"]:
                self.nets["points_encoder"] = PointNet2Encoder(
                    activation=config["points"]["activation"], bn=config["points"]["bn"],
                    output_size=config["points"]["output_size"],
                )
            else:
                self.nets["points_encoder"] = PointNet2EncoderGamma(
                    activation=config["points"]["activation"], bn=config["points"]["bn"],
                    output_size=config["points"]["output_size"],
                )

        if self.config["use_proprio_encoder"]:
            proprio_dim = obs_shapes['eef_quat'][0] + obs_shapes['q'][0] + obs_shapes['eef_pos'][0]

            if not self.config["use_gamma"]:
                self.nets["proprio_encoder"] = ProprioEncoder(
                    input_size=proprio_dim,
                    output_size=config["proprio"]["output_size"]
                )
            else:
                self.nets["proprio_encoder"] = ProprioEncoderGamma(
                    input_size=proprio_dim,
                    output_size=config["proprio"]["output_size"]
                )

    def get_output_size(self):
        size = 0
        if self.config['use_voxel_encoder']:
            size += self.config['voxel']['output_size']
        if self.config['use_points_encoder']:
            size += self.config['points']['output_size']
        if self.config['use_proprio_encoder']:
            size += self.config['proprio']['output_size']

        return size

    def forward(self, scene_pc, obj_pc, proprio_st):

        features = []
        if self.config['use_voxel_encoder']:
            # filter by size
            scene_pc = self.nets['voxel_encoder'].filter_pts(scene_pc)
            features.append(self.nets['voxel_encoder'](scene_pc))
        if self.config['use_points_encoder']:
            features.append(self.nets['points_encoder'](obj_pc))
        if self.config['use_proprio_encoder']:
            features.append(self.nets['proprio_encoder'](proprio_st))

        features = torch.cat(features, dim=-1)

        return features