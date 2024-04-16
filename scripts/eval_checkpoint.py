import os
import pickle
import uuid
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
import random
from tqdm import tqdm
import json

def create_policy_nets(cfg):
    """
    Creates policy network architecture consisting of:
    1) Vision Encoder (ResNet)
    2) Prototype Predictor (Skill Alignment Transformer)
    3) Noise Predictor (Diffusion Model)


    Parameters
    ----------
    cfg : DictConfig
        Specifies model architecture


    Returns 
    -------
    nets : torch.nn.ModuleDict
        Dictionary with all three network components
    """
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    proto_dim = cfg.proto_dim

    if cfg.upsample_proto:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon +
            proto_horizon * cfg.upsample_proto_net.out_size,
        )
    else:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon + proto_horizon * proto_dim,
        )
    proto_pred_net = hydra.utils.instantiate(
        cfg.proto_pred_net,
        input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
    )

    # Complete policy architecture
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "proto_pred_net": proto_pred_net,
        "noise_pred_net": noise_pred_net,
    })

    if cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    return nets

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="eval_checkpoint",
)
def main(cfg: DictConfig):
    """
    Uses trained XSkill policy to evaluate on (19) held out episodes,
    with subtasks: ['microwave', 'kettle', 'light switch', 'slide cabinet'].


    Parameters
    ----------
    cfg : DictConfig
        Specifies configurations for policy networks as well as evaluation details

    Side Effects
    ------------
    - Saves gifs of policy rollouts and a json with subtask completion percentages to trained_model_path folder

    Returns 
    -------
    None
    """
    save_dir = os.path.join(cfg.trained_model_path, 'post_training')

    nets = create_policy_nets(cfg)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    device = torch.device("cuda")
    _ = nets.to(device)
    nets.eval()

    eval_callback = hydra.utils.instantiate(cfg.eval_callback)
    file = open(os.path.join(cfg.trained_model_path, 'stats.pickle'), 'rb')
    stats = pickle.load(file)

    with open(cfg.eval_cfg.eval_mask_path, "r") as f:
        eval_mask = json.load(f)
        eval_mask = np.array(eval_mask)
    
    eval_eps = np.arange(len(eval_mask))[eval_mask]

    result_dict = {
        'robot':{f'{ckpt_num}': {} for ckpt_num in cfg.checkpoint_list},
        'human':{f'{ckpt_num}': {} for ckpt_num in cfg.checkpoint_list}
    }

    speeds = {
        'robot': cfg.robot_speeds,
        'human': cfg.human_speeds
    }
    
    for i, ckpt_num in enumerate(cfg.checkpoint_list):
        nets.load_state_dict(torch.load(os.path.join(cfg.trained_model_path, f'ckpt_{ckpt_num}.pt')))
        for demo_type in ['robot', 'human']:
            cfg.eval_cfg.demo_type = demo_type
            for speed in speeds[demo_type]:
                eval_callback.task_progess_ratio = speed
                tasks_completed = 0
                for seed in eval_eps:
                    cfg.eval_cfg.demo_item = seed.item()
                    num_completed, _ = eval_callback.eval(
                        nets,
                        noise_scheduler,
                        stats,
                        cfg.eval_cfg,
                        save_dir,
                        seed,
                        epoch_num=None
                    )
                    tasks_completed += num_completed

                result_dict[demo_type][f'{ckpt_num}'][f'{speed}'] = tasks_completed / (4*len(eval_eps))
                print(result_dict)
    
    with open(os.path.join(save_dir, "policy_results.json"), "w") as outfile:
        json.dump(result_dict, outfile)

    averages = {"robot": {f'{speed}': 0 for speed in speeds['robot']}, "human": {f'{speed}': 0 for speed in speeds['human']}}
    counts = {"robot": {f'{speed}': 0 for speed in speeds['robot']}, "human": {f'{speed}': 0 for speed in speeds['human']}}

    for demo_type, values in result_dict.items():
        for ckpt_num, acc_dicts in values.items():
            for exec_speed, acc in acc_dicts.items():
                averages[demo_type][exec_speed] += acc
                counts[demo_type][exec_speed] += 1

    for demo_type, values in averages.items():
        for exec_speed, summed_acc in values.items():
            averages[demo_type][exec_speed] /= counts[demo_type][exec_speed]

    with open(os.path.join(save_dir, "policy_results_avg.json"), "w") as outfile:
        json.dump(averages, outfile)
            

if __name__ == '__main__':
    main()