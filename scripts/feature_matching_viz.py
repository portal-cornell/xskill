import numpy as np
from PIL import Image
import os
from skimage.transform import resize
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import json
import torch
import torchvision.transforms as Tr
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch import nn, einsum
import omegaconf
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_state_and_to_tensor(vid):
    state_path = os.path.join(vid, "states.json")
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data = np.array(state_data, dtype=np.float32)
    return state_data


def load_images(folder_path, resize_shape=None):
    images = []  # initialize an empty list to store the images

    # get a sorted list of filenames in the folder
    filenames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # loop through all PNG files in the sorted list
    for filename in filenames:
        # open the image file using PIL library
        img = Image.open(os.path.join(folder_path, filename))
        # convert the image to a NumPy array
        img_arr = np.array(img)
        if resize_shape is not None:
            img_arr = cv2.resize(img_arr, resize_shape)
        images.append(img_arr)  # add the image array to the list

    # convert the list of image arrays to a NumPy array
    images_arr = np.array(images)
    return images_arr


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor

def gif_of_clip(cfg, demo_type, ep_num, frame_num, slide, output_dir, cycle=False):
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    sub_imgs = images_arr[frame_num:frame_num + slide + 1]

    pil_imgs = [Image.fromarray(img) for img in sub_imgs]
    pil_imgs[0].save(os.path.join(output_dir, f'{demo_type}_{ep_num}_index{frame_num}{"_cycle" if cycle else ""}.gif'), save_all=True, append_images=pil_imgs[1:], duration=200, loop=0)

    return pil_imgs


def traj_representations(cfg, model, pipeline, demo_type, ep_num, frame_list=None):
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


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="compute_nn",
)
def label_dataset(cfg: DictConfig):
    model = load_model(cfg)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

    T = 50
    threshold = 20
    for clip_num in range(0, 251, 25):
        robot_clip_traj, _ = traj_representations(cfg, model, pipeline, 'robot', clip_num)
        human_clip_traj, _ = traj_representations(cfg, model, pipeline, 'human', clip_num)
        dists_bw_robot_human = torch.cdist(robot_clip_traj, human_clip_traj, p=2)[T]
        dists_bw_robot_robot = torch.cdist(robot_clip_traj, robot_clip_traj, p=2)[T]

        # ignore frames within a threshold
        dists_bw_robot_human[[np.arange(T-threshold, T+threshold)]] = float('inf')
        dists_bw_robot_robot[[np.arange(T-threshold, T+threshold)]] = float('inf')

        closest_human_frame = torch.argmin(dists_bw_robot_human).item()
        closest_robot_frame = torch.argmin(dists_bw_robot_robot).item()

        if cfg.save_clips:
            output_dir = f'gif_outputs2/{clip_num}_{T}'
            os.makedirs(output_dir, exist_ok=True)
            robot_gif = gif_of_clip(cfg, 'robot', clip_num, T, 8, output_dir)
            human_gif = gif_of_clip(cfg, 'human', clip_num, closest_human_frame, 8, output_dir)
            returned_robot_gif = gif_of_clip(cfg, 'robot', clip_num, closest_robot_frame, 8, output_dir)


if __name__ == "__main__":
    label_dataset()