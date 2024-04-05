from PIL import Image
import numpy as np
import cv2
import torch
import os
import json
from omegaconf import DictConfig
import hydra
import omegaconf
from xskill.utility.diffusion_bc_callback import load_images, convert_images_to_tensors



def load_model(cfg):
    """
    Loads pretrained model from XSkill's vision encoder.


    Parameters
    ----------
    cfg : DictConfig
        Determines model path and checkpoint to be loaded.

    Returns 
    -------
    model : xskill.model.core.Model
        Vision encoder from XSkill
    """
    exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)
    model.eval()
    print("model loaded")  
    return model

def gif_of_clip(cfg, demo_type, ep_num, frame_num, slide, output_dir, cycle=False):
    """
    Computes latent representations of a video from embodiment type {demo_type}
    and episode number {ep_num} based on a provided model.


    Parameters
    ----------
    cfg : DictConfig
        Configures path to data and image details
    demo_type : str
        'robot' or 'human'
    ep_num : int
        Episode number
    frame_num : int 
        Start frame for video clip
    slide : int 
        Duration of clip
    output_dir : str
        Directory to save gif
    cycle : bool (optional)
        True => retrieved from cycle-back operation

    Side Effects
    ------------
    - Saves gif to specified output_dir

    Returns 
    -------
    pil_imgs : Image list
        Images from the specified clip
    """
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    sub_imgs = images_arr[frame_num:frame_num + slide + 1]

    pil_imgs = [Image.fromarray(img) for img in sub_imgs]
    pil_imgs[0].save(os.path.join(output_dir, f'{demo_type}_{ep_num}_index{frame_num}{"_cycle" if cycle else ""}.gif'), save_all=True, append_images=pil_imgs[1:], duration=200, loop=0)
    
    return pil_imgs

def traj_representations(cfg, model, pipeline, demo_type, ep_num, frame_list=None):
    """
    Computes latent representations of a video from embodiment type {demo_type}
    and episode number {ep_num} based on a provided model.


    Parameters
    ----------
    cfg : DictConfig
        Configures path to data and image details
    model : xskill.model.core.Model
        Pretrained vision encoder model
    pipeline : torch.nn.modules.container.Sequential
        Image preprocessing pipeline
    demo_type : str
        'robot' or 'human'
    ep_num : int
        Episode number
    frame_list : int list (optional)
        Subset of frames to extract clips from
    

    Returns 
    -------
    traj_rep : torch.Tensor
        Tensor of shape (T, D=256), representing a sequence of latents 
        in the skill representation space
    z : torch.Tensor
        Tensor of shape (T, K), representing a distribution across prototypes
    """
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda() # (T, C, H, W)

    eps_len = images_tensor.shape[0]
    im_q = torch.stack(
        [
            images_tensor[j : j + model.slide + 1]
            for j in range(eps_len - model.slide)
        ]
    )  # (B,slide+1,C,H,W)

    clips = im_q
    if frame_list is not None:
        clips = clips[frame_list]

    z = model.encoder_q(im_q, None) # (T, K)
    state_rep = model.encoder_q.get_state_representation(clips, None) # (T, 2, D)
    traj_rep = model.encoder_q.get_traj_representation(state_rep) # (T, D)
    return traj_rep, z