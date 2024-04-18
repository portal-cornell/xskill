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
from xskill.utility.eval_utils import gif_of_clip, traj_representations, load_model


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="compute_nn",
)
def main(cfg: DictConfig):
    """
    Performs qualitative visual feature analysis by consider a frame at timestep T in 
    a robot video and finding its nearest neighbor (outside of a threshold around T)
    in both the robot and human videos to see if they are visually similar.
    

    Parameters
    ----------
    cfg : DictConfig
        Specifies configuration details associated with the visual encoder being tested.
        - Note that exp_path must be the path to a trained model

    Side Effects
    ------------
    - save_clips == True will save gifs from the videos

    Returns 
    -------
    None
    """
    model = load_model(cfg)

    # Image preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)


    T = 50 # Any arbitrary frame index from a clip
    threshold = 20 # +/- threshold on T to ignore when finding nearest neighbors
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
            output_dir = os.path.join(cfg.clip_path, f'features/{clip_num}_{T}')
            os.makedirs(output_dir, exist_ok=True)

            robot_gif = gif_of_clip(cfg, 'robot', clip_num, T, 8, output_dir)
            human_gif = gif_of_clip(cfg, 'human', clip_num, closest_human_frame, 8, output_dir)
            returned_robot_gif = gif_of_clip(cfg, 'robot', clip_num, closest_robot_frame, 8, output_dir)


if __name__ == "__main__":
    main()