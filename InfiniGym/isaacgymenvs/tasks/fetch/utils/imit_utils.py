
from e2e_imit.algo.bc_mlp import PTD_BC_MLPGaussian, PTD_BC_MLPGaussian_ACT
from e2e_imit.algo.bc_transformer import PTD_BC_TransformerGMM, PTD_BC_TransformerGMM_ACT

import imageio.v3 as iio
from omegaconf import OmegaConf
from collections import OrderedDict


def load_ckpt_config(path):
    model_path = f'{path}/model.pth'
    config_path = f'{path}/config.yaml'
    config = OmegaConf.load(config_path)

    config["train"]["ckpt_path"] = model_path
    config["train"]["use_ddp"] = False

    print(config)
    return config


def get_obs_shape(config):
    obs_shape = OrderedDict({
        "q": (9,),
        "eef_pos": (3,),
        "eef_quat": (4,),
        "rigid_pos": (11, 3),
        "rigid_quat": (11, 4),
        "visual": {
            'scene': (config["algo"]["obs_shape"]['num_scene_points'], 3),
            'goal': (config["algo"]["obs_shape"]['num_goal_points'], 3),
            'robot': (config["algo"]["obs_shape"]['num_robot_points'], 3)
        }
    })
    if config["algo"]["two_phase"]:
        obs_shape['phase_index'] = (2,)

    return obs_shape


def get_action_params(config):
    action_params = config["dataset"]["action_params"]
    if config["dataset"]["action_type"] == 'joint':
        return {
            'shape': (8,),
            'type': 'joint',
            'scale': action_params['dof'] * config["dataset"]["frame_skip"] * action_params['clip_std_scale']
        }
    elif config["dataset"]["action_type"] == 'osc':
        return {
            'shape': (7,),
            'type': 'osc',
            'scale': {
                'pos': action_params['eef_pos'] * config["dataset"]["frame_skip"] * action_params['clip_std_scale'],
                'angle': action_params['eef_angle'] * config["dataset"]["frame_skip"] * action_params['clip_std_scale']
            }
        }
    else:
        raise NotImplementedError


def load_imitation_algo(ckpt_path):
    if not ckpt_path is None:
        config = load_ckpt_config(ckpt_path)
    else:
        raise NotImplementedError

    if config.algo.model_type == 'MLP_Gaussian':
        algo = PTD_BC_MLPGaussian(
            algo_config=config.algo,
            global_config=config,
            obs_key_shapes=get_obs_shape(config),
            ac_params=get_action_params(config),
            device_infos=[0],
            ckpt_path=None  # no ckpt saving
        )
    elif config.algo.model_type == 'MLP_Gaussian_ACT':
        algo = PTD_BC_MLPGaussian_ACT(
            algo_config=config.algo,
            global_config=config,
            obs_key_shapes=get_obs_shape(config),
            ac_params=get_action_params(config),
            device_infos=[0],
            ckpt_path=None
        )
    elif config.algo.model_type == 'Transformer_GMM':
        config["algo"]["transformer"]["context_length"] = config.dataset.seq_length
        algo = PTD_BC_TransformerGMM(
            algo_config=config.algo,
            global_config=config,
            obs_key_shapes=get_obs_shape(config),
            ac_params=get_action_params(config),
            device_infos=[0],
            ckpt_path=None
        )

    elif config.algo.model_type == 'Transformer_GMM_ACT':
        config["algo"]["transformer"]["context_length"] = config.dataset.seq_length
        algo = PTD_BC_TransformerGMM_ACT(
            algo_config=config.algo,
            global_config=config,
            obs_key_shapes=get_obs_shape(config),
            ac_params=get_action_params(config),
            device_infos=[0],
            ckpt_path=None
        )
    else:
        raise NotImplementedError

    return algo
