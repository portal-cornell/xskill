import omegaconf
from omegaconf import DictConfig
import hydra
import numpy as np
import os
import zarr
import imagecodecs
import numcodecs 
from imagecodecs.numcodecs import Jpeg2k 
numcodecs.register_codec(Jpeg2k) 
import matplotlib.pyplot as plt

def plot(x, y, divider, colors_used, cfg):
    color_mapper = {
        'red': '#FFA500',  # Using hex color codes for precision
        'lightred': '#FFD580',
        'blue': '#0000FF',
        'lightblue': '#ADD8E6',
        'green': '#008000',
        'lightgreen': '#90EE90',
        'orange': '#FFA500',
        'lightorange': '#FFD580',
    }

    # Map colors
    colors = [color_mapper[color] for color in y]

    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.yticks([])
    plt.xticks([])

    # Segment data
    robot_x = x[:divider]
    human_x = x[divider:]

    robot_colors = colors[:divider]
    human_colors = colors[divider:]

    # Draw robot and human points
    plt.scatter(robot_x[:, 0], robot_x[:, 1], c=robot_colors, s=200, marker='o', edgecolors='black', linewidths=1)
    plt.scatter(human_x[:, 0], human_x[:, 1], c=human_colors, s=200, marker='P', edgecolors='black', linewidths=1)

    # Additional aesthetics (if necessary, e.g., drawing cluster lines)
    # plt.plot(...) for cluster lines if needed

    plt.savefig(cfg.save_path)
    plt.show()

def get_task_prototype_dict_human(task, proto_path, video_path):
    proto_data = zarr.open(proto_path, mode='r')
    human_protos = proto_data[f'human/{task}/raw_rep']
    video_data = zarr.open(video_path+f'/human/{task}/replay_buffer.zarr', mode='r')
    task_segment_info_files = os.listdir(video_path + '/human/' + task + '/demos')
    # sort task segment info files by number in demo000x.npz 
    # task_segment_info_files.sort(key=lambda x: int(x[4:-4]))
    task_segment_id_dict = {}
    for task_segment_info_file in task_segment_info_files:
        if task_segment_info_file == 'stats.npz':
            continue
        demo_num = int(task_segment_info_file[4:-4])
        print("Demo num: ", demo_num)
        if demo_num == 2:
            break
        task_segment_info = np.load(video_path + '/human/' + task + '/demos/' + task_segment_info_file, allow_pickle=True)
        episode_info = task_segment_info['episode']
        for timestep in range(len(episode_info)):
            if demo_num not in task_segment_id_dict:
                task_segment_id_dict[demo_num] = {}
            task_segment_id_dict[demo_num][timestep] = episode_info[timestep]['segment']
    episode_ends = video_data['meta/episode_ends']
    
    task_prototype_dict = {}
    # for i in range(len(episode_ends)):
    for i in range(2):
        if i == 0:
            ep_start = 0
            ep_end = episode_ends[i]
        else:
            ep_start = episode_ends[i-1]
            ep_end = episode_ends[i]
        
        for timestep in range(ep_start+30, ep_end):
            task = task_segment_id_dict[i][timestep-ep_start]
            if task not in task_prototype_dict:
                task_prototype_dict[task] = []
            task_prototype_dict[task].append(human_protos[timestep])
        break
    return task_prototype_dict

def get_task_prototype_dict_robot(task, proto_path, video_path):
    proto_data = zarr.open(proto_path, mode='r')
    robot_protos = proto_data[f'robot/{task}/raw_rep']
    video_data = zarr.open(video_path+f'/robot/{task}/replay_buffer.zarr', mode='r')
    task_segment_info_files = os.listdir(video_path + '/robot/' + task + '/demos')
    # sort task segment info files by number in demo000x.npz 
    # task_segment_info_files.sort(key=lambda x: int(x[4:-4]))
    task_segment_id_dict = {}
    for task_segment_info_file in task_segment_info_files:
        if task_segment_info_file == 'stats.npz':
            continue
        demo_num = int(task_segment_info_file[4:-4])
        print("Demo num: ", demo_num)
        if demo_num == 2:
            break
        task_segment_info = np.load(video_path + '/robot/' + task + '/demos/' + task_segment_info_file, allow_pickle=True)
        episode_info = task_segment_info['episode']
        for timestep in range(len(episode_info)):
            if demo_num not in task_segment_id_dict:
                task_segment_id_dict[demo_num] = {}
            task_segment_id_dict[demo_num][timestep] = int(timestep/len(episode_info)*3)
    episode_ends = video_data['meta/episode_ends']
    
    task_prototype_dict = {}
    # for i in range(len(episode_ends)):
    for i in range(2):
        if i == 0:
            ep_start = 0
            ep_end = episode_ends[i]
        else:
            ep_start = episode_ends[i-1]
            ep_end = episode_ends[i]
        
        for timestep in range(ep_start+30, ep_end):
            task = task_segment_id_dict[i][timestep-ep_start]
            if task not in task_prototype_dict:
                task_prototype_dict[task] = []
            task_prototype_dict[task].append(robot_protos[timestep])
        break
    return task_prototype_dict

@hydra.main(version_base=None,
            config_path="../../config/realworld",
            config_name="plot_tsne")
def plot_tsne(cfg: DictConfig):
    task = 'ABC'
    robot_task_prototype_dict = get_task_prototype_dict_robot(task, cfg.proto_path, cfg.video_path)
    human_task_prototype_dict = get_task_prototype_dict_human(task, cfg.proto_path, cfg.video_path)
    # task_prototype_dict = human_task_prototype_dict
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    x, y = [], []
    # colors = 
    # choose two sets of same 3 colors with one set being lighter than the other
    color_mapper = {
        'red': '#FFA500',  # Using hex color codes for precision
        'lightred': '#FFD580',
        'blue': '#0000FF',
        'lightblue': '#ADD8E6',
        'green': '#008000',
        'lightgreen': '#90EE90',
        'orange': '#FFA500',
        'lightorange': '#FFD580',
    }

    task_color_dict = {
        0: 'red',
        1: 'blue',
        2: 'green',
    }


    for task in robot_task_prototype_dict:
        print(task)
        x.extend(robot_task_prototype_dict[task])
        y.extend(color_mapper[task_color_dict[task]] for _ in range(len(robot_task_prototype_dict[task])))
        x.extend(human_task_prototype_dict[task])
        y.extend(color_mapper[f'light{task_color_dict[task]}'] for _ in range(len(human_task_prototype_dict[task])))

    x = np.array(x)
    y = np.array(y)
    x_embedded = TSNE(n_components=2).fit_transform(x)
    # plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y)
    # # add legend
    # # plt.legend({'Robot': 'red', 'Human': '
    # plt.show()
    # # save the plot
    # plt.savefig('check.png')
    sampled_idxs = np.random.choice(len(x), 200)
    x_embedded, y = x_embedded[sampled_idxs], y[sampled_idxs]
    plt.figure(figsize=(8, 8))
    plt.yticks([])
    plt.xticks([])
    for x1, y1, in zip(x_embedded, y):
        # print(x1, y1)
        lightcolors = [color_mapper['lightred'], color_mapper['lightblue'], color_mapper['lightgreen']]
        marker = 'P' if y1 in lightcolors else 'o'
        plt.scatter(x1[0], x1[1], c=y1, s=200, marker=marker, edgecolors='black', linewidths=1)
    plt.show()
    plt.savefig("check.png")
    # plot(x_embedded, y, len(robot_task_prototype_dict), color_mapper, cfg

    # breakpoint()

if __name__ == "__main__":
    plot_tsne()
