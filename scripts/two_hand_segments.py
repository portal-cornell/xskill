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
from xskill.utility.utils import read_json, write_json


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}


def detect_moving_objects_array(arr, obs_indices, threshold=0.005):
    """Returns a numpy array where each row corresponds to a time step and each column corresponds to an object.
    The value at each cell in the array is a boolean indicating whether the corresponding object has moved or not.

    Parameters:
    arr (np.ndarray): The input array of shape (T, 30).
    obs_indices (dict): A dictionary containing indices of objects in the array.
    threshold (float): The threshold for detecting a change in the object's value.

    Returns:
    np.ndarray: A boolean array of shape (T, len(obs_indices)) indicating which objects have moved for each time step.
    """
    moving_objects_array = np.zeros((arr.shape[0], len(obs_indices)), dtype=bool)

    for t in range(arr.shape[0]):
        for i, (obj, indices) in enumerate(obs_indices.items()):
            # Get the value of the object at the current time step
            obj_value = arr[t, indices]

            # Get the value of the object at the previous time step
            if t > 0:
                prev_obj_value = arr[t - 1, indices]
            else:
                prev_obj_value = obj_value
            # print(np.abs(obj_value - prev_obj_value)> threshold)
            # Check if the difference between the current and previous values is greater than threshold
            if (np.abs(obj_value - prev_obj_value) > threshold).any():
                moving_objects_array[t, i] = True

    return moving_objects_array


def load_state_and_to_tensor(vid):
    state_path = os.path.join(vid, "states.json")
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data = np.array(state_data, dtype=np.float32)
    return state_data

import shutil

def find_lowest_unused_number(episode_save_path):
    os.makedirs(episode_save_path, exist_ok=True)
    existing_folders = os.listdir(episode_save_path)
    existing_numbers = [int(folder_name) for folder_name in existing_folders if folder_name.isdigit()]
    if not existing_numbers:
        return 0
    return max(existing_numbers) + 1

def save_episode_frames(data_folder, start_idx, end_idx, episode_save_path):
    episode_num = find_lowest_unused_number(episode_save_path)
    episode_folder = os.path.join(episode_save_path, str(episode_num))
    os.makedirs(episode_folder, exist_ok=True)
    for i, frame_idx in enumerate(range(start_idx, end_idx + 1)):
        frame_path = os.path.join(data_folder, f"{frame_idx}.png")  # Assuming frames are in PNG format
        shutil.copy(frame_path, os.path.join(episode_folder, f"{i}.png"))  # Copy frame to episode folder with new index

def get_segments(cfg, moved_obj, demo_type, data_folder):
    segments = []
    # Find start and end indices of each episode
    episode_start = None
    prev_segment_end = 0
    current_episode_moved_obj = None
    for idx, episode in enumerate(moved_obj):
        if current_episode_moved_obj is None and np.any(episode):
            current_episode_moved_obj = episode
        if np.array_equal(episode, current_episode_moved_obj):
            if episode_start is None:
                episode_start = idx
        elif episode_start is not None:
            # End of episode
            episode_end = idx - 1
            episode_name = "_".join([obj for i, obj in enumerate(OBS_ELEMENT_INDICES.keys()) if moved_obj[episode_start][i]])
            episode_save_path = os.path.join(cfg.data_path, demo_type + '_segments_paired_twohands')
            if (episode_end - episode_start) >= 10:
                # save_episode_frames(data_folder, prev_segment_end if cfg.include_transition else episode_start, episode_end, episode_save_path)
                segments.append({
                                'data_folder': data_folder,
                                'start_idx': prev_segment_end if cfg.include_transition else episode_start,
                                'end_idx': episode_end,
                                'episode_save_path': episode_save_path
                            })
            episode_start = None
            prev_segment_end = episode_end + 1
            current_episode_moved_obj = None

    # Handle the case where the last episode extends until the end
    if episode_start is not None:
        episode_end = len(moved_obj) - 1
        episode_name = "_".join([obj for i, obj in enumerate(OBS_ELEMENT_INDICES.keys()) if moved_obj[episode_start][i]])
        episode_save_path = os.path.join(cfg.data_path, demo_type + '_segments_paired_twohands')
        if (episode_end - episode_start) >= 10:
            # save_episode_frames(data_folder, prev_segment_end if cfg.include_transition else episode_start, episode_end, episode_save_path)
            segments.append({
                                'data_folder': data_folder,
                                'start_idx': prev_segment_end if cfg.include_transition else episode_start,
                                'end_idx': episode_end,
                                'episode_save_path': episode_save_path
                            })
            
    return segments


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
)
def label_dataset(cfg: DictConfig):
    with open(cfg.vid_mask, 'r') as f:
        vid_mask = json.load(f)
    segment_vid_mask = []
    for demo_type in ["robot"]:
    # for demo_type in ["twohands", "human"]:
        paired_type = "twohands"
        data_path = os.path.join(cfg.data_path, demo_type)
        paired_data_path = os.path.join(cfg.data_path, paired_type)
        all_folders = os.listdir(data_path)
        all_folders = sorted(all_folders, key=lambda x: int(x))
        for folder_path in tqdm(all_folders, disable=not cfg.verbose):
            data_folder = os.path.join(data_path, folder_path)
            
            state_arr = load_state_and_to_tensor(data_folder)
            moved_obj = detect_moving_objects_array(state_arr, OBS_ELEMENT_INDICES)
            moved_obj = np.array(moved_obj, dtype=np.int32)
            robot_list = get_segments(cfg, moved_obj, demo_type, data_folder)

            paired_data_folder = os.path.join(paired_data_path, folder_path)
            
            state_arr = load_state_and_to_tensor(paired_data_folder)
            moved_obj = detect_moving_objects_array(state_arr, OBS_ELEMENT_INDICES)
            moved_obj = np.array(moved_obj, dtype=np.int32)

            human_list = get_segments(cfg, moved_obj, paired_type, paired_data_folder)
            human_list = []
            human_list.append({
                'data_folder': paired_data_folder,
                'start_idx': 0,
                'end_idx': 59,
                'episode_save_path': os.path.join(cfg.data_path, paired_type + '_segments_paired_twohands')
            })
            human_list.append({
                'data_folder': paired_data_folder,
                'start_idx': 60,
                'end_idx': 119,
                'episode_save_path': os.path.join(cfg.data_path, paired_type + '_segments_paired_twohands')
            })

            if len(robot_list) < 4:
                robot_list = [{
                    'data_folder': data_folder,
                    'start_idx': robot_list[0]['start_idx'],
                    'end_idx': robot_list[1]['end_idx'],
                    'episode_save_path': os.path.join(cfg.data_path, demo_type + '_segments_paired_twohands')
                }]
                human_list = [human_list[0]]
            else:
                robot_list = [{
                    'data_folder': data_folder,
                    'start_idx': robot_list[0]['start_idx'],
                    'end_idx': robot_list[1]['end_idx'],
                    'episode_save_path': os.path.join(cfg.data_path, demo_type + '_segments_paired_twohands')
                },
                {
                    'data_folder': data_folder,
                    'start_idx': robot_list[2]['start_idx'],
                    'end_idx': robot_list[3]['end_idx'],
                    'episode_save_path': os.path.join(cfg.data_path, demo_type + '_segments_paired_twohands')
                }]
            for robot_dict, human_dict in zip(robot_list, human_list):
                if vid_mask[int(folder_path)]:
                    segment_vid_mask.append(True)
                else:
                    segment_vid_mask.append(False)
                # save_episode_frames(**robot_dict)
                # save_episode_frames(**human_dict)

    write_json(
        f"{cfg.base_dev_dir}/xskill/datasets/kitchen_dataset/robot_segments_paired_twohands/train_mask.json",
        segment_vid_mask,
    )
            
            




if __name__ == "__main__":
    label_dataset()
