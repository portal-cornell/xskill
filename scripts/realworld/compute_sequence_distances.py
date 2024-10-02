import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import omegaconf
from tqdm import tqdm
import os
import zarr
import imagecodecs
import torch
import numcodecs 
from imagecodecs.numcodecs import Jpeg2k 
numcodecs.register_codec(Jpeg2k) 
from xskill.utility.eval_utils import traj_representations, load_model, compute_tcc_loss, compute_optimal_transport_loss
import imageio

@hydra.main(
    version_base=None,
    config_path="../../config/realworld",
    config_name="sequence_metrics",
)
def main(cfg: DictConfig):
    human_prototypes = zarr.open(cfg.prototypes_path + '/human_protos_images/prototype_dict.zarr', mode='r')
    robot_prototypes = zarr.open(cfg.prototypes_path + '/robot_protos/prototype_dict.zarr', mode='r')
    data_dir = cfg.data_dir
    # reconstructed_dataset = zarr.open(video_path + '/robot/ABC/replay_buffer.zarr', mode='r')
    distances = {}
    for robot_ep in robot_prototypes.keys():
        print(robot_ep)
        distances[robot_ep] = {}
        human_video = []
        for segment in robot_prototypes[robot_ep].keys():
            distances[robot_ep][segment] = {}
            robot_protos = robot_prototypes[robot_ep][segment]['protos']
            for human_ep in human_prototypes.keys():
                human_protos = human_prototypes[human_ep]['protos']
                distance = compute_optimal_transport_loss(torch.tensor(robot_protos).unsqueeze(0), torch.tensor(human_protos).unsqueeze(0))
                distances[robot_ep][segment][human_ep] = distance
            # find min human_ep for robot_ep and segment
            min_human_ep = min(distances[robot_ep][segment], key=distances[robot_ep][segment].get)
            print(f"Robot episode {robot_ep}, segment {segment} is closest to human episode {min_human_ep} with distance {distances[robot_ep][segment][min_human_ep]}")
            human_video.extend(list(human_prototypes[min_human_ep]['images']))
        human_video = np.array(human_video)
        # convert human_video into a list of dictionaries with single key 'obs'
        human_video = [{'obs': {'human_image':frame}} for frame in human_video]
        # breakpoint()
        task = robot_ep.split('_')[0]
        ep_num = robot_ep.split('_')[1]
        # represent ep_num as a 4 digit string with leading zeros
        ep_num = ep_num.zfill(5)
        robot_ep_file = os.path.join(data_dir, f'robot/{task}/demos/demo{ep_num}.npz')
        robot_ep_data = np.load(robot_ep_file, allow_pickle=True)

        # Convert NpzFile to a dictionary
        robot_ep_data_dict = {key: robot_ep_data[key] for key in robot_ep_data.keys()}
        
        # Make modifications to robot_ep_data_dict as needed
        # For example, adding a new key-value pair
        robot_ep_data_dict['human_video'] = human_video
        
        robot_save_file = os.path.join(data_dir, f'robot/{task}/demos_new/demo{ep_num}.npz')
        # make sure robot_save_file directory exists
        os.makedirs(os.path.dirname(robot_save_file), exist_ok=True)
        np.savez(robot_save_file, **robot_ep_data_dict)
        # Save the modified data back to a new .npz file
        # np.savez(robot_ep_file, **robot_ep_data_dict)

        # breakpoint()
        # break
if __name__ == "__main__":
    main()
