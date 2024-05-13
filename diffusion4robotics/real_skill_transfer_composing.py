# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os, hydra, traceback, torch, tqdm, yaml
import numpy as np
from data4robotics import misc, transforms
from omegaconf import DictConfig, OmegaConf
import json
from hydra import compose
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
import random
import uuid


base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "experiments"),
    config_name="finetune.yaml",
    version_base="1.1",
)
def bc_finetune(cfg: DictConfig):
    model_cfg = compose(config_name="skill_transfer_composing")
    # create save dir
    use_wandb = model_cfg.use_wandb
    # unique_id = str(uuid.uuid4())
    unique_id = model_cfg.policy_name if model_cfg.policy_name != 'FILL' else 'real' + str(uuid.uuid4())
    save_dir = os.path.join(model_cfg.save_dir, unique_id)
    model_cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(model_cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")
    # Set up logger
    if use_wandb:
        wandb.init(project=model_cfg.project_name)
        wandb.config.update(OmegaConf.to_container(model_cfg))

    #set seed
    torch.manual_seed(model_cfg.seed)
    np.random.seed(model_cfg.seed)
    random.seed(model_cfg.seed)

    device = torch.device("cuda")
    print(device)

    # parameters
    pred_horizon = model_cfg.pred_horizon
    obs_horizon = model_cfg.obs_horizon
    proto_horizon = model_cfg.proto_horizon

    if model_cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18", weights="IMAGENET1K_V1")
    else:
        vision_encoder = ResnetConv(embedding_size=model_cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)

    if model_cfg.vision_feature_dim == 512:
        vision_encoder_wrist = get_resnet("resnet18")
    else:
        vision_encoder_wrist = ResnetConv(embedding_size=model_cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder_wrist = replace_bn_with_gn(vision_encoder_wrist)
    vision_feature_dim = model_cfg.vision_feature_dim * 2 # *2 for double cam

    # observation and action dimensions corrsponding to
    # the output of PushTEnv
    obs_dim = model_cfg.obs_dim
    action_dim = model_cfg.action_dim
    proto_dim = model_cfg.proto_dim

    # create network object
    if model_cfg.upsample_proto:
        noise_pred_net = hydra.utils.instantiate(
            model_cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon +
            proto_horizon * model_cfg.upsample_proto_net.out_size,
        )
    else:
        noise_pred_net = hydra.utils.instantiate(
            model_cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon + proto_horizon * proto_dim,
        )
    proto_pred_net = hydra.utils.instantiate(
        model_cfg.proto_pred_net,
        input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
    )

    # the final arch has 3 parts
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "vision_encoder_wrist": vision_encoder_wrist,
        "proto_pred_net": proto_pred_net,
        "noise_pred_net": noise_pred_net,
    })

    if model_cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(model_cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    noise_scheduler = hydra.utils.instantiate(model_cfg.noise_scheduler)
    # device transfer
    
    _ = nets.to(device)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(params=nets.parameters(),
                                  lr=model_cfg.lr,
                                  weight_decay=model_cfg.weight_decay)

    ob_data = open('../../../ob_norm.json')
    ac_data = open('../../../ac_norm.json')
    with open('../../../obs_config.yaml', 'r') as file:
        obs_config = yaml.safe_load(file)
    with open('obs_config.yaml', 'w') as file:
        yaml.dump(obs_config, file)

    # returns JSON object as
    # a dictionary
    ob_dict = json.load(ob_data)
    ac_dict = json.load(ac_data)
    ac_max = np.array(ac_dict['maximum'])
    ac_min = np.array(ac_dict['minimum'])
    with open('ac_norm.json', 'w') as f:
        json.dump(ac_dict, f)
    with open('ob_norm.json', 'w') as f:
        json.dump(ob_dict, f)
    print(cfg.task)

    task = hydra.utils.instantiate(
        cfg.task, batch_size=model_cfg.batch_size, num_workers=cfg.num_workers
    )
    num_trans = len(task.train_loader.dataset.wrapped)
    max_iterations = int(cfg.num_epochs*num_trans/model_cfg.batch_size) # iterations per epoch
    # create a gpu train transform (if used)
    gpu_transform = (
        transforms.get_gpu_transform_by_name(cfg.train_transform)
        if "gpu" in cfg.train_transform
        else None
    )

    train_iterator = iter(task.train_loader)
    min_eval = np.inf
    min_l2 = np.inf
    print(task.train_loader)
    num_batches = int(num_trans/model_cfg.batch_size)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_batches * model_cfg.num_epochs,
    )

    # create eval callbacl
    eval_callback = hydra.utils.instantiate(model_cfg.eval_callback)
    batch_sampler = iter(task.train_loader)

    import copy
    net_copy = copy.deepcopy(nets)

    try:
    # gracefully handle and log errors
        for epoch_idx in tqdm.tqdm(range(model_cfg.num_epochs)):
            epoch_loss = list()
            epoch_action_loss = list()
            epoch_proto_prediction_loss = list()
            
            for batch_idx in range(num_batches):
                batch = next(batch_sampler)
                
                if gpu_transform is not None:
                    (imgs, obs), actions, mask = batch
                    imgs = {k: v.to(device) for k, v in imgs.items()}
                    imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                    batch = ((imgs, obs.to(device)), actions.to(device), mask)
                
                # batch[0][1] := observation (robot state)
                nobs = batch[0][1] # (B, 2*J)
                B = nobs.shape[0]
                nobs = torch.reshape(nobs, (B, cfg.img_chunk, -1)) # (B, 2, J)


                # batch[0][0] := dictionary with both camera's frames (cam0, cam1)
                nimage = batch[0][0]['cam1'] # (B, obs_horizon, 3, 224, 224)
                nimage_wrist = batch[0][0]['cam0'] # (B, obs_horizon, 3, 224, 224)
            
                # batch[1] := tensor (B, action pred horizon, action dim) (actions)
                naction = batch[1] # (B, action pred horizon, action_dim)

                # TODO
                nproto = torch.zeros((B, 1, 256)).to(device)
                proto_snap = torch.zeros((B, 100, 256)).to(device)

                # encoder vision features
                image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                image_features = image_features.reshape(
                    *nimage.shape[:2], -1)  # (B,obs_horizon,visual_feature)

                image_features_wrist = nets["vision_encoder_wrist"](nimage_wrist.flatten(end_dim=1))
                image_features_wrist = image_features.reshape(
                    *nimage_wrist.shape[:2], -1)  # (B,obs_horizon,visual_feature)

                obs_feature = torch.cat(
                    [image_features, image_features_wrist, nobs],
                    dim=-1)  # (B,obs_horizon,low_dim_feature+visual_feature)
                
                # feed in: (B,obs_feature*obs_horizon),(B,snap_frame,D)
                proto_cond = torch.clone(nproto).detach()
                if model_cfg.unconditioned_policy:
                    proto_cond.fill_(0)
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        proto_cond.flatten(start_dim=1)
                    ],
                    dim=1,
                )

                predict_proto = proto_pred_net(obs_feature.flatten(start_dim=1),
                                           proto_snap)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (B, ),
                    device=device).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(naction, noise,
                                                        timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(noisy_actions,
                                            timesteps,
                                            global_cond=obs_cond)
                
                # L2 loss
                action_loss = nn.functional.mse_loss(noise_pred, noise)
                if model_cfg.unconditioned_policy:
                    proto_prediction_loss = 0
                else:
                    proto_prediction_loss = nn.functional.mse_loss(
                        predict_proto, nproto.squeeze(1))
                
                loss = action_loss + proto_prediction_loss
                
                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                epoch_action_loss.append(action_loss.item())
                if not model_cfg.unconditioned_policy:
                    epoch_proto_prediction_loss.append(proto_prediction_loss.item())
                
                with torch.no_grad():
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim),
                        device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(60)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = nets["noise_pred_net"](sample=naction,
                                                            timestep=k,
                                                            global_cond=obs_cond)
                        breakpoint()

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(model_output=noise_pred,
                                                    timestep=k,
                                                    sample=naction).prev_sample
                    
                    # unnormalize action
                    naction = naction.detach().to("cpu").numpy()
                    # (B, pred_horizon, action_dim)
                    naction = naction[0]
                    ac = naction[0]
            if use_wandb:
                wandb.log({
                    "epoch loss":
                    np.mean(epoch_loss),
                    "epoch action loss":
                    np.mean(epoch_action_loss),
                    "epoch proto prediction loss":
                    np.mean(epoch_proto_prediction_loss),
                })

            if epoch_idx % model_cfg.ckpt_frequency == 0:
                ema.copy_to(net_copy.parameters())
                torch.save(
                    net_copy.state_dict(),
                    os.path.join(save_dir, f"ckpt_{epoch_idx}.pt"),
                )

    except:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    bc_finetune()
