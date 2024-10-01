import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import omegaconf
from tqdm import tqdm
import os
import zarr
import imagecodecs
import numcodecs 
from imagecodecs.numcodecs import Jpeg2k 
numcodecs.register_codec(Jpeg2k) 
from xskill.utility.eval_utils import traj_representations, load_model, compute_tcc_loss, compute_optimal_transport_loss

def get_prototype_dict_human(proto_path, video_path):
    prototype_dict = {}
    proto_data = zarr.open(proto_path, mode='r')
    for task in ['ABC', 'ABD', 'BCD']:
        human_protos = proto_data[f'human/{task}/raw_rep']
        video_data = zarr.open(video_path+f'/human/{task}/replay_buffer.zarr', mode='r')
        task_segment_info_files = os.listdir(video_path + '/human/' + task + '/demos')
        task_segment_id_dict = {}
        for task_segment_info_file in task_segment_info_files:
            if task_segment_info_file == 'stats.npz':
                continue
            demo_num = int(task_segment_info_file[4:-4])
            print("Demo num: ", demo_num)
            task_segment_info = np.load(video_path + '/human/' + task + '/demos/' + task_segment_info_file, allow_pickle=True)
            episode_info = task_segment_info['episode']
            for timestep in range(len(episode_info)):
                if demo_num not in task_segment_id_dict:
                    task_segment_id_dict[demo_num] = []
                if timestep == len(episode_info)-1 or episode_info[timestep]['segment']+1 == episode_info[timestep+1]['segment']:
                    task_segment_id_dict[demo_num].append(timestep)
        episode_ends = video_data['meta/episode_ends']
        prototype_dict[task] = {}
        for i in range(len(episode_ends)):
            print("here I am at demo num = ", i)
            if i == 0:
                ep_start = 0
                ep_end = episode_ends[i]
            else:
                ep_start = episode_ends[i-1]
                ep_end = episode_ends[i]
            prototype_dict[task][i] = {
            }
            for time_idx in range(len(task_segment_id_dict[i])):
                print("here I am at time_idx = ", time_idx)
                segment_start = task_segment_id_dict[i][time_idx-1]+ep_start if time_idx > 0 else ep_start
                segmend_end = task_segment_id_dict[i][time_idx]+ep_start
                prototype_dict[task][i][time_idx] = {
                    'protos': human_protos[segment_start:segmend_end],
                    'images': video_data['data/third_person_cam'][segment_start:segmend_end]
                    }
    return prototype_dict

def get_prototype_dict_robot(proto_path, video_path, num_segments=3):
    prototype_dict = {}
    proto_data = zarr.open(proto_path, mode='r')
    for task in ['ABC', 'ABD', 'BCD']:
        human_protos = proto_data[f'robot/{task}/raw_rep']
        video_data = zarr.open(video_path+f'/robot/{task}/replay_buffer.zarr', mode='r')
        episode_ends = video_data['meta/episode_ends']
        prototype_dict[task] = {}
        for i in range(len(episode_ends)):
            if i == 0:
                ep_start = 0
                ep_end = episode_ends[i]
            else:
                ep_start = episode_ends[i-1]
                ep_end = episode_ends[i]
            prototype_dict[task][i] = {
            }
            # divide time indices between ep_start and ep_end into num_segments
            segment_length = (ep_end-ep_start)/num_segments
            for segment_idx in range(num_segments):
                segment_start = int(segment_idx*segment_length)+ep_start
                segment_end = int((segment_idx+1)*segment_length)+ep_start
                prototype_dict[task][i][segment_idx] = {
                    'protos': human_protos[segment_start:segment_end],
                    'images': video_data['data/third_person_cam'][segment_start:segment_end]
                }

    return prototype_dict

@hydra.main(
    version_base=None,
    config_path="../../config/realworld",
    config_name="segment_wise_dists",
)
def main(cfg: DictConfig):
    human_proto_dict = get_prototype_dict_human(cfg.proto_path, cfg.video_path)
    robot_proto_dict = get_prototype_dict_robot(cfg.proto_path, cfg.video_path)
    # save human_proto_dict and robot_proto_dict
    np.save(cfg.save_path + '/human_proto_dict.npy', human_proto_dict)
    np.save(cfg.save_path + '/robot_proto_dict.npy', robot_proto_dict)
    

        
    


if __name__ == "__main__":
    main()
