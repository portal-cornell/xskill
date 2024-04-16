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

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="eval_checkpoint",
)
def main(cfg: DictConfig):
    unique_id = 'reproduce'
    save_dir = os.path.join(cfg.save_dir, unique_id)
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    
    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    # observation and action dimensions corrsponding to
    # the output of PushTEnv
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    proto_dim = cfg.proto_dim

    # create network object
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

    # the final arch has 3 parts
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "proto_pred_net": proto_pred_net,
        "noise_pred_net": noise_pred_net,
    })

    if cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    # device transfer
    device = torch.device("cuda")
    _ = nets.to(device)

    nets.load_state_dict(torch.load(cfg.model_path))

    eval_callback = hydra.utils.instantiate(cfg.eval_callback)
    file = open(os.path.join(cfg.trained_model_path, 'stats.pickle'), 'rb')
    stats = pickle.load(file)

    with open(cfg.eval_cfg.eval_mask_path, "r") as f:
        eval_mask = json.load(f)
        eval_mask = np.array(eval_mask)
    
    eval_eps = np.arange(len(eval_mask))[eval_mask]

    robot_res = 0
    human_res = 0
    robot_alltasks = 0
    human_alltasks = 0
    
    for demo_type in ['robot', 'human']:
        cfg.eval_cfg.demo_type = demo_type
        for seed in eval_eps:
            cfg.eval_cfg.demo_item = seed.item()
            task_list = ["slide cabinet", "light switch", "kettle", "microwave"]
            num_completed, _ = eval_callback.eval(
                nets,
                noise_scheduler,
                stats,
                cfg.eval_cfg,
                save_dir,
                seed,
                epoch_num=None,
                task_list=task_list
            )
            if demo_type == 'robot':
                robot_res += num_completed
                if num_completed == len(task_list):
                    robot_alltasks += 1
            else: 
                human_res += num_completed
                if num_completed == len(task_list):
                    human_alltasks += 1
    print('Robot')
    print(f"{robot_res} total tasks finished")
    print(f"{robot_res / (4*len(eval_eps)) * 100}% of tasks finished")
    print(f"{robot_alltasks / len(eval_eps) * 100}% of episodes completed all tasks")
    print('Human')
    print(f"{human_res} total tasks finished")
    print(f"{human_res / (4*len(eval_eps)) * 100}% of tasks finished")
    print(f"{human_alltasks / len(eval_eps) * 100}% of episodes completed all tasks")

if __name__ == '__main__':
    main()