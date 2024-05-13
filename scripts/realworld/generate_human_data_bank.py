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
from xskill.utility.eval_utils import traj_representations, load_model


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

def load_state_and_to_tensor(vid):
    state_path = os.path.join(vid, "states.json")
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data = np.array(state_data, dtype=np.float32)
    return state_data

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


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
)
def main(cfg: DictConfig):
    """
    Generates l2 distance matrix between each robot video and all human z's from the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Specifies vision encoder to use.

    Side Effects
    ------------
    - Saves l2 distance matrices to the folders corresponding to each robot episode.

    Returns 
    -------
    None
    """
    model = load_model(cfg)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((200, 200)), normalize)

    data_path = os.path.join(cfg.data_path, cfg.human_type)
    all_folders = os.listdir(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    # np.random.shuffle(all_folders)
    # all_folders = all_folders[:cfg.batch_size]
    num_zs = []

    all_human_zs = []
    index_to_object = []

    vid_to_idx_range = {}
    lower = 0
    for folder_path in tqdm(all_folders, disable=not cfg.verbose):
        data_folder = os.path.join(data_path, folder_path)

        # state_arr = load_state_and_to_tensor(data_folder)
        # moved_obj = detect_moving_objects_array(state_arr, OBS_ELEMENT_INDICES)
        # moved_obj = np.array(moved_obj, dtype=np.int32)
        # moved_obj = moved_obj.tolist()
        # index_to_object.extend(moved_obj)

        traj_representation, _ = traj_representations(cfg, model, pipeline, cfg.human_type, int(folder_path))
        traj_representation = traj_representation.detach().cpu().numpy()
        traj_representation = np.array(traj_representation).tolist()

        all_human_zs.extend(traj_representation)
        num_zs.append(len(traj_representation))
        
        upper = lower + len(traj_representation)
        vid_to_idx_range[int(folder_path)] = [lower, upper]
        lower = upper
    
    save_folder = os.path.join(
        cfg.exp_path, f"{cfg.human_type}_l2", f"ckpt_{cfg.ckpt}"
    )
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(cfg.exp_path, f"{cfg.human_type}_l2", f"ckpt_{cfg.ckpt}", f"{cfg.human_type}_z_moved_obj.json"), "w") as f:
        json.dump(index_to_object, f)

    with open(os.path.join(cfg.exp_path, f"{cfg.human_type}_l2", f"ckpt_{cfg.ckpt}", "vid_to_idx_range.json"), "w") as f:
        json.dump(vid_to_idx_range, f)
    
    all_human_zs = torch.Tensor(all_human_zs).cuda()

    data_path = os.path.join(cfg.data_path, 'robot')
    all_folders = os.listdir(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    for folder_path in tqdm(all_folders, disable=not cfg.verbose):
        save_folder = os.path.join(
            cfg.exp_path, f"{cfg.human_type}_l2", f"ckpt_{cfg.ckpt}", folder_path
        )
        os.makedirs(save_folder, exist_ok=True)

        traj_representation, _ = traj_representations(cfg, model, pipeline, 'robot', int(folder_path))

        l2_dists = torch.cdist(traj_representation, all_human_zs, p=2)
        l2_dists = l2_dists.detach().cpu().numpy()
        l2_dists = np.array(l2_dists).tolist()

        with open(os.path.join(save_folder, "l2_dists.json"), "w") as f:
            json.dump(l2_dists, f)


if __name__ == "__main__":
    main()
