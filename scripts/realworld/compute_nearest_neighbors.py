import numpy as np
from PIL import Image
import random
import os
import os.path as osp
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

def find_episode_and_frame(human_z_idx, idx_dict):
    """
    Given a list of human z indices from the data bank and a dictionary that maps episode numbers to human z indices,
    extracts the true episode number and the frame from within that episode.


    Parameters
    ----------
    human_z_idx : numpy.ndarray
        List of indices (ranging between 0 and the total number of z's in the bank)
    idx_dict : dictionary
        Key: episode num, Value: 2-element list with lower (inclusive) and upper (exclusive) index bound

    Ex) human_z_idx = [5, 260]
        idx_dict = {0: [0, 250], 250: [250, 500]}
        (human z indices 0-249 come from episode 0, indices 250-499 come from episode 250)
    

    Returns 
    -------
    episode_numbers : numpy.ndarray
        episode_number[i] = episode number for which human_z_idx[i] comes from
    frames : numpy.ndarray
        frame[i] = frame number within episode_number[i]
    """
    episode_numbers = []
    frames = []
    for idx in human_z_idx:
        for ep_num, (lower, upper) in idx_dict.items():
            if lower <= idx < upper:
                episode_numbers.append(ep_num)
                frames.append(idx-lower)
                break

    return np.array(episode_numbers).astype(np.int32), np.array(frames).astype(np.int32)

def add_representation_suffix(path):
    return os.path.join(path, "traj_representation.json")

@hydra.main(
    version_base=None,
    config_path="../../config/realworld",
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
    pipeline = nn.Sequential(Tr.CenterCrop((200, 200)), normalize)

    tasks_list = json.load(open(cfg.task_json_path))
    clip_count = 0
    thresh_accs = [0 for _ in range(cfg.correct_thresholds)]

    # Specify some clips to be evaluated on
    for clip_num in range(5):
        clip_count += 1

        # Obtain human and robot sequences of z's
        robot_clip_traj, _ = traj_representations(cfg, model, pipeline, 'robot', clip_num) # (T, D=256)
        human_clip_traj, _ = traj_representations(cfg, model, pipeline, cfg.human_type, clip_num) # (T, D=256)

        # USE THIS FOR COMPARING ZR AND ZH ON SIMULT EPS
        # snap_idx = random.sample(list(range(len(robot_clip_traj))), k=100)
        # snap_idx.sort()

        # zr = robot_clip_traj[snap_idx]
        # zh = human_clip_traj[snap_idx]

        # dists = torch.cdist(zr, zh, p=2) # (T, T)
        # closest_to_robot = torch.argmin(dists, dim=1)
        # closest_zh = zh[closest_to_robot]
        # zr_to_zh_diffs = torch.norm(zr - closest_zh, dim=1)


        if cfg.is_lookup: # does nearest neighbor replacement on the robot sequence of z's
            l2_dist_path = osp.join(cfg.nearest_neighbor_data_dirs, f'{clip_num}')
            l2_dist_path = os.path.join(l2_dist_path, 'l2_dists.json')
            with open(l2_dist_path, "r") as f:
                l2_dist_data = json.load(f)
            l2_dist_data = np.array(l2_dist_data, dtype=np.float32)
            eps_len = len(l2_dist_data)
            snap_idx = random.sample(list(range(eps_len)), k=100)
            snap_idx.sort()
            
            l2_dist_data = l2_dist_data[snap_idx]
            human_z_idx = np.argmin(l2_dist_data, axis=1)

            with open(os.path.join(cfg.nearest_neighbor_data_dirs, 'episode_list.json'), 'r') as f:
                episode_list = json.load(f)

            with open(os.path.join(cfg.nearest_neighbor_data_dirs, 'vid_to_idx_range.json'), 'r') as f:
                idx_dict = json.load(f)

            z_tilde = []
            
            episode_nums, frame_nums = find_episode_and_frame(human_z_idx, idx_dict)
            
            gif_cfg = DictConfig({'data_path': cfg.data_path, 'resize_shape': [124,124]})
            reconstructed_video = []
            orig_video = []
            human_vid = []
            robot_vid_num = clip_num
            for k, (ep_num, frame_num) in enumerate(zip(episode_nums, frame_nums)):
                human_proto_path = osp.join(cfg.paired_proto_dirs, str(ep_num))
                human_proto_path = add_representation_suffix(human_proto_path)
                with open(human_proto_path, "r") as f:
                    human_proto_data = json.load(f)

                # NN Replacement
                z_tilde.append(human_proto_data[int(frame_num)])
                
                if cfg.save_lookups and k % 9 == 0:
                    human_imgs = gif_of_clip(gif_cfg, cfg.human_type, ep_num, frame_num, 8, None, save=False)
                    robot_imgs = gif_of_clip(gif_cfg, 'robot', robot_vid_num, snap_idx[k], 8, None, save=False)
                    reconstructed_video.extend(human_imgs)
                    orig_video.extend(robot_imgs)
                    gt_human_imgs = gif_of_clip(gif_cfg, cfg.human_type, robot_vid_num, snap_idx[k], 8, None, save=False)
                    human_vid.extend(gt_human_imgs)


            if cfg.save_lookups:
                os.makedirs(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num)), exist_ok=True)
                reconstructed_video[0].save(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), f'constructed_human.gif'), save_all=True, append_images=reconstructed_video[1:], duration=200, loop=0)
                orig_video[0].save(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), f'orig_robot.gif'), save_all=True, append_images=orig_video[1:], duration=200, loop=0)
                human_vid[0].save(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), f'gt_human.gif'), save_all=True, append_images=human_vid[1:], duration=200, loop=0)

            def cos_sim_fn(a, b):
                return np.sum(a * b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))

            znn = np.array(z_tilde)
            zr = robot_clip_traj[snap_idx].cpu().detach().numpy()
            zh = human_clip_traj[snap_idx].cpu().detach().numpy()
            znn_to_zh_diffs = np.linalg.norm(znn - zh, axis=1)
            znn_to_zr_diffs = np.linalg.norm(znn - zr, axis=1)
            zr_to_zh_diffs = np.linalg.norm(zr - zh, axis=1)
            znn_minus_zr = znn - zr
            zh_minus_zr = zh - zr
            cos_sim = cos_sim_fn(znn_minus_zr, zh_minus_zr)
            cos_sim2 = cos_sim_fn(znn, zh)
            import matplotlib.pyplot as plt
            plt.plot(np.arange(100), znn_to_zh_diffs)
            plt.xlabel('Frame Idx')
            plt.ylabel('L2 Dist (Znn & Zh)')
            plt.savefig(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), 'errors.png'))
            plt.clf()

            plt.plot(np.arange(100), znn_to_zr_diffs)
            plt.xlabel('Frame Idx')
            plt.ylabel('L2 Dist (Znn & Zr)')
            plt.savefig(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), 'errors_zr.png'))
            plt.clf()

            plt.plot(np.arange(100), zr_to_zh_diffs)
            plt.xlabel('Frame Idx')
            plt.ylabel('L2 Dist (Zr & Zh)')
            plt.savefig(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), 'errors_zh.png'))
            plt.clf()

            plt.plot(np.arange(100), 1-cos_sim)
            plt.xlabel('Frame Idx')
            plt.ylabel('Cosine Dist (Zr->Znn & Zr->Zh)')
            plt.savefig(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), 'cos_dists.png'))
            plt.clf()

            plt.plot(np.arange(100), 1-cos_sim2)
            plt.xlabel('Frame Idx')
            plt.ylabel('Cosine Dist (Znn & Zh)')
            plt.savefig(os.path.join(f'{cfg.exp_path}/nn', str(robot_vid_num), 'cos_dists_znn.png'))
            plt.clf()

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
            human_gif = gif_of_clip(cfg, cfg.human_type, clip_num, human_nn_frame, 8, output_dir) # Nearest Neighbor (Human)
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
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(list(range(cfg.correct_thresholds)), thresh_accs)
        plt.title('Cycle Back Accuracy')
        plt.xlabel('Frame Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(cfg.clip_path, "tcc_results.png"))




if __name__ == "__main__":
    main()