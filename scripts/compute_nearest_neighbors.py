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
    Computes temporal cyclic consistency metrics between human and robot videos.


    Parameters
    ----------
    cfg : DictConfig
        Specifies configuration details associated with the visual encoder being tested.
        - Note that exp_path must be the path to a trained model
        - correct_thresholds specifies the range of frames to consider for TCC accuracy

    Side Effects
    ------------
    - Prints out TCC accuracy over the correct_thresholds
    - save_clips == True will save gifs from the videos and a plot of TCC accuracy

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

    tasks_list = json.load(open(cfg.task_json_path))
    clip_count = 0
    thresh_accs = [0 for _ in range(cfg.correct_thresholds)]

    # Specify some clips to be evaluated on
    for clip_num in range(0, 251, 25):
        clip_count += 1

        # Obtain human and robot sequences of z's
        robot_clip_traj, _ = traj_representations(cfg, model, pipeline, 'robot', clip_num) # (T, D=256)
        human_clip_traj, _ = traj_representations(cfg, model, pipeline, 'human', clip_num) # (T, D=256)
        dists = torch.cdist(robot_clip_traj, human_clip_traj, p=2) # (T, T)

        closest_to_robot = torch.argmin(dists, dim=1)
        closest_to_human = torch.argmin(dists, dim=0) 

        cycled_back = closest_to_human[closest_to_robot]

        # Compute cycle-back distances
        diffs = torch.abs(cycled_back - torch.arange(cycled_back.shape[0], device = cfg.device))

        for thresh in range(cfg.correct_thresholds):
            correct_class = diffs <= thresh
            acc = correct_class.sum() / correct_class.shape[0]
            # print(f'Threshold {thresh}: {acc.item()}')
            thresh_accs[thresh] += acc.item()
        
        if cfg.save_clips:
            # Save video clip with worst cycle-back error
            frame_num = torch.argmax(diffs[20:-20]).item() # ignores frames at the very beginning and end of the episode
            human_nn_frame = closest_to_robot[frame_num].item()
            tcc_frame = closest_to_human[human_nn_frame].item()

            output_dir = os.path.join(cfg.clip_path, f'tcc/{clip_num}_{frame_num}')
            os.makedirs(output_dir, exist_ok=True)

            robot_gif = gif_of_clip(cfg, 'robot', clip_num, frame_num, 8, output_dir) # Start frame (Robot)
            human_gif = gif_of_clip(cfg, 'human', clip_num, human_nn_frame, 8, output_dir) # Nearest Neighbor (Human)
            returned_robot_gif = gif_of_clip(cfg, 'robot', clip_num, tcc_frame, 8, output_dir, cycle=True) # Nearest Neighbor (Robot)

            dictionary = {
                'robot_clip_num': clip_num,
                'robot_clip_tasks': tasks_list[clip_num],
                'robot_frame_num': frame_num,
                'human_clip_num': clip_num,
                'human_clip_tasks': tasks_list[clip_num],
                'human_frame_num': human_nn_frame,
                'tcc_robot_frame': tcc_frame
            }
            with open(os.path.join(output_dir, "output.json"), "w") as outfile:
                json.dump(dictionary, outfile)

    thresh_accs = np.array(thresh_accs)
    thresh_accs = thresh_accs / clip_count
    print(thresh_accs)

    if cfg.save_clips:
        with open(os.path.join(cfg.clip_path, "tcc_results.json"), "w") as outfile:
            json.dump({'correct_thresholds': list(range(cfg.correct_thresholds)), 'threshold_accs': thresh_accs.tolist()}, outfile)

        plt.figure(figsize=(8, 6))
        plt.plot(list(range(cfg.correct_thresholds)), thresh_accs)
        plt.title('Cycle Back Accuracy')
        plt.xlabel('Frame Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(cfg.clip_path, "tcc_results.png"))




if __name__ == "__main__":
    main()