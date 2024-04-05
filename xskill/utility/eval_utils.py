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
    TODO
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
    TODO
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
    TODO
    """
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

    eps_len = images_tensor.shape[0]
    im_q = torch.stack(
        [
            images_tensor[j : j + model.slide + 1]
            for j in range(eps_len - model.slide)
        ]
    )  # (b,slide+1,c,h,w)

    clips = im_q
    if frame_list is not None:
        clips = clips[frame_list]

    z = model.encoder_q(im_q, None)
    state_rep = model.encoder_q.get_state_representation(clips, None)
    traj_rep = model.encoder_q.get_traj_representation(state_rep)

    return traj_rep, z