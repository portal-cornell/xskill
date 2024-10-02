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

def get_prototype_dict_human(proto_path, video_path, save_path):
    # make prototype dict a zarr file
    # prototype_dict = {}
    prototype_dict = zarr.open(save_path + '/human_protos_images/prototype_dict.zarr', mode='w')
    proto_data = zarr.open(proto_path, mode='r')
    for task in ['ABC', 'ABD', 'BCD']:
        print("I am on task ===========", task)
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
        # prototype_dict[task] = {}

        for i in range(len(episode_ends)):
            if i == 0:
                ep_start = 0
                ep_end = episode_ends[i]
            else:
                ep_start = episode_ends[i-1]
                ep_end = episode_ends[i]
            for time_idx in range(len(task_segment_id_dict[i])):
                segment_start = task_segment_id_dict[i][time_idx-1]+ep_start if time_idx > 0 else ep_start
                segmend_end = task_segment_id_dict[i][time_idx]+ep_start
                task_segment_id = f"{task}_{i}_{time_idx}"
                prototype_dict.create_group(task_segment_id)
                prototype_dict[task_segment_id].create_dataset('protos', data=human_protos[segment_start:segmend_end])
                prototype_dict[task_segment_id].create_dataset('images', data=video_data['data/third_person_cam'][segment_start:segmend_end])

    # return prototype_dict

def get_prototype_dict_robot(proto_path, video_path, save_path, num_segments=3):
    prototype_dict = zarr.open(save_path + '/robot_protos/prototype_dict.zarr', mode='w')
    proto_data = zarr.open(proto_path, mode='r')
    for task in ['ABC', 'ABD', 'BCD']:
        print("I am on robot task ===========", task)
        human_protos = proto_data[f'robot/{task}/raw_rep']
        video_data = zarr.open(video_path+f'/robot/{task}/replay_buffer.zarr', mode='r')
        episode_ends = video_data['meta/episode_ends']
        for i in range(len(episode_ends)):
            if i == 0:
                ep_start = 0
                ep_end = episode_ends[i]
            else:
                ep_start = episode_ends[i-1]
                ep_end = episode_ends[i]
            # divide time indices between ep_start and ep_end into num_segments
            segment_length = (ep_end-ep_start)/num_segments
            task_segment_id = f"{task}_{i}"
            prototype_dict.create_group(task_segment_id)
            for segment_idx in range(num_segments):
                segment_start = int(segment_idx*segment_length)+ep_start
                segment_end = int((segment_idx+1)*segment_length)+ep_start
                prototype_dict[task_segment_id].create_dataset(f'{segment_idx}/protos', data=human_protos[segment_start:segment_end])
                # prototype_dict[task_segment_id].create_dataset(f'{segment_idx}/protos', data=human_protos[segment_start:segment_end])
                # prototype_dict[task_segment_id].create_dataset('images', data=video_data['data/third_person_cam'][segment_start:segment_end])
    return prototype_dict

@hydra.main(
    version_base=None,
    config_path="../../config/realworld",
    config_name="chop_into_segments",
)
def main(cfg: DictConfig):
    get_prototype_dict_human(cfg.proto_path, cfg.video_path, cfg.save_path)
    get_prototype_dict_robot(cfg.proto_path, cfg.video_path, cfg.save_path)
    # # save human_proto_dict and robot_proto_dict
    # np.save(cfg.save_path + '/human_proto_dict.npy', human_proto_dict)
    # np.save(cfg.save_path + '/robot_proto_dict.npy', robot_proto_dict)
    

        
    


if __name__ == "__main__":
    main()
