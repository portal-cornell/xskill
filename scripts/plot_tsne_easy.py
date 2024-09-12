import pandas as pd
import matplotlib.pyplot as plt
import os
from omegaconf import DictConfig
import hydra
import numpy as np
import seaborn as sns
import json


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

def plot_proto_task_relation(demo_type="human", cfg=None):
    """
    Heatmap showing the concentration of prototype activations for each subtask.


    Parameters
    ----------
    demo_type : str
        'robot' or 'human'
    cfg : DictConfig
        Specifies pretrained vision encoder model path

    Side Effects
    ------------
    - Saves heatmap to exp_path folder
    - Saves csvs mapping subtasks to prototypes to exp_path folder

    Returns 
    -------
    None
    """

    if demo_type == "human":
        encode_path = os.path.join(
            cfg.exp_path, "human_encode_protos", f"ckpt_{cfg.ckpt}"
        )
    else:
        encode_path = os.path.join(cfg.exp_path, "encode_protos", f"ckpt_{cfg.ckpt}")

    all_folders = os.listdir(encode_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    if cfg.plot_top_k is not None:
        all_folders = all_folders[: cfg.plot_top_k]
    softmax_protos = []
    labels = []
    for f in all_folders:
        with open(
            os.path.join(encode_path, f, "softmax_encode_protos.json"), "r"
        ) as file:
            softmax_protos.append(np.array(json.load(file)))
        with open(os.path.join(encode_path, f, "moved_obj.json"), "r") as file:
            labels.append(np.array(json.load(file)))


    softmax_protos = np.concatenate(softmax_protos)
    labels = np.concatenate(labels)

    max_proto = np.argmax(softmax_protos, axis=1)
    viz_pd = pd.DataFrame()
    viz_pd["max_proto"] = max_proto
    viz_pd["task"] = [
        list(OBS_ELEMENT_INDICES.keys())[np.argmax(labels[i])]
        for i in range(len(labels))
    ]
    non_zero_index = (labels != 0).any(axis=1)
    viz_pd = viz_pd[non_zero_index]
    sns.histplot(viz_pd, x="max_proto", y="task", bins=100)

    import matplotlib.pyplot as plt

    plt.savefig(
        os.path.join(cfg.exp_path, f"{demo_type}_proto_task_relation_{cfg.ckpt}.png")
    )
    viz_pd.to_csv(
        os.path.join(cfg.exp_path, f"{demo_type}_proto_task_relation_{cfg.ckpt}.csv"),
        index=False,
    )


def proto_scatter_plot(exp_path, demo_type):
    """
    Scatter plot showing top 5 most activate prototypes for each of
    the following subtasks: ['kettle', 'microwave', 'hinge cabinet', 'slide cabinet'].


    Parameters
    ----------
    exp_path : str
        Path to pretrained vision encoder model
    demo_type : str
        'robot' or 'human'

    Side Effects
    ------------
    - Saves scatter plot to exp_path folder

    Returns 
    -------
    None
    """
    # Create DataFrame from CSV data
    df = pd.read_csv(os.path.join(exp_path, f'{demo_type}_proto_task_relation_79.csv'))

    tasks_of_interest = ['kettle', 'microwave', 'hinge cabinet', 'slide cabinet']
    filtered_df = df[df['task'].isin(tasks_of_interest)]

    grouped_df = filtered_df.groupby(['task', 'max_proto']).size().reset_index(name='count')

    colors = {
        'kettle': 'blue', 
        'microwave': 'black', 
        'hinge cabinet': 'red', 
        'slide cabinet': 'orange'
    }

    # Plot counts for each task
    plt.figure(figsize=(10, 6))
    for task in tasks_of_interest:
        task_data = grouped_df[grouped_df['task'] == task]
        task_data_sorted = task_data.sort_values(by='count', ascending=False).head(5)  # Select top 5 most frequent max_proto
        task_data_sorted = task_data_sorted.sort_values(by='max_proto', ascending=True)
        plt.scatter(task_data_sorted['max_proto'], task_data_sorted['count'], label=task, color=colors[task], s=100) 

    plt.title(f'Count of max_proto Numbers for Each Task (Top 5) - {demo_type}')
    plt.xlabel('max_proto')
    plt.ylabel('Count')
    plt.legend(title='Task')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'{demo_type}_scatter.png'))
    plt.show()

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

# def plot(x, y, divider, colors_used, cfg):
#     color_mapper = {
#         'red': 'r',
#         'lightred': 'lightcoral',
#         'blue': 'b',
#         'lightblue': 'lightblue',
#         'green': 'g',
#         'lightgreen': 'lightgreen',
#     }
#     for i in range(len(y)):
#         y[i] = color_mapper[y[i]]
#     plt.figure(figsize=(8, 8))
#     plt.yticks([])
#     plt.xticks([])
#     # in the next line, the scatter plot, mark with circle
#     plt.scatter(x[: divider, 0], x[:divider, 1], c=y[:divider], s=50, marker='o')
#     plt.scatter(x[divider:, 0], x[divider:, 1], c=y[divider:], s=50, marker='P')
#     plt.savefig(cfg.save_path)
#     plt.show()

def plot_tsne(cfg):
    """
    t-SNE plot showing the concentration of prototype activations for each subtask.


    Parameters
    ----------
    exp_path : str
        Path to pretrained vision encoder model
    demo_type : str
        'robot' or 'human'

    Side Effects
    ------------
    - Saves t-SNE plot to exp_path folder

    Returns 
    -------
    None
    """
    robot_tasks_of_interest = [
            "kettle", 
            "bottom burner", 
            # "slide cabinet", 
            "light switch",
            # "hinge cabinet"
            ]
    human_tasks_interested = [
        ["kettle", "light switch"],
        ["kettle", "bottom burner"],
        # ["light switch", "slide cabinet"],
        # ["top burner", "hinge cabinet"],
        # ["slide cabinet", "hinge cabinet"]
    ]
    colors = {
        'kettle': 'blue', 
        'bottom burner': 'orange', 
        'light switch': 'green', 
        'demo_kettle': 'lightblue',
        'demo_bottom burner': 'lightorange',
        'demo_light switch': 'lightgreen',
        # 'kettle_light switch': 'yellow',
        # 'kettle_bottom burner': 'pink',
        # 'light switch_slide cabinet': 'brown',
        # 'slide cabinet_hinge cabinet': 'grey',

        # 'kettle_light switch': 'purple',
        # 'microwave_kettle': 'brown',
        # 'microwave_light switch': 'pink',
        # 'light switch_hinge cabinet': 'yellow'
        # 'light switch_hinge cabinet': 'purple',
        # 'top burner_kettle': 'brown',
        # 'kettle_light switch': 'pink',
        # 'top burner_hinge cabinet': 'yellow'


    }

    task_completions_list = json.load(open(os.path.join(cfg.data_path, "task_completions.json"), 'r'))
    tasks_list = ["bottom burner", "top burner", "light switch", "slide cabinet", "hinge cabinet", "microwave", "kettle"]
    task_to_idx = {task: idx for idx, task in enumerate(tasks_list)}

    robot_task_frame_dict = {}
    
    for task in robot_tasks_of_interest: robot_task_frame_dict[task] = []

    human_task_frame_dict = {}
    
    for task in robot_tasks_of_interest:
        human_task_frame_dict[f'demo_{task}'] = []
    # task to idx dict
    
    
    # breakpoint()
    

    

    
    import random
    # for eps_num in random.sample(range(1, 600), 100):
    # for eps_num in [342, 177, 343, 325, 195]: 
    for eps_num in [17, 56, 45, 31, 30, 19, 48, 38, 59, 10, 5, 58, 20, 57, 6, 42]:
        # print(eps_num)
        # if len(task_completions_list[eps_num]) != 4:
        #     continue
        # t1, t2, t3, t4 = task_completions_list[eps_num]
        # # print(t1, t2, t3, t4)
        # if [t1, t2] not in human_tasks_interested and [t2, t1] not in human_tasks_interested:
        #     # print(f'{t1}_{t2}')
        #     # breakpoint()
        #     continue
        # if [t3, t4] not in human_tasks_interested and [t4, t3] not in human_tasks_interested:
        #     # print(f'{t1}_{t2}')
        #     # print(f'{t3}_{t4}')
        #     continue
        # print(eps_num)
        # print(t1, t2, t3, t4)
        # breakpoint()
        robot_data = json.load(open(os.path.join(cfg.exp_path, "encode_protos", f"ckpt_{cfg.ckpt}", f'{eps_num}/traj_representation.json')))
        human_data = json.load(open(os.path.join(cfg.exp_path, f"{cfg.human_type}_encode_protos", f"ckpt_{cfg.ckpt}", f'{eps_num}/traj_representation.json')))

        robot_states = json.load(open(os.path.join(cfg.data_path, f"robot/{eps_num}/states.json"), 'r'))
        robot_states = np.array(robot_states)
        robot_moved_obj = detect_moving_objects_array(robot_states, OBS_ELEMENT_INDICES)

        human_states = json.load(open(os.path.join(cfg.data_path, f"{cfg.human_type}/{eps_num}/states.json"), 'r'))
        human_states = np.array(human_states)
        human_moved_obj = detect_moving_objects_array(human_states, OBS_ELEMENT_INDICES)

        for task in robot_tasks_of_interest:
            for i in range(len(robot_moved_obj)):
                start = min(i, len(robot_moved_obj)-1)
                end = max(i, 0) 
                if robot_moved_obj[start, task_to_idx[task]] and robot_moved_obj[end, task_to_idx[task]]:
                    robot_task_frame_dict[task].append(robot_data[i])

        for task in robot_tasks_of_interest:
            for i in range(len(human_moved_obj)):
                start = min(i, len(human_moved_obj)-1)
                end = max(i, 0) 
                # find if both task 1 and task 2 are moved
                if human_moved_obj[start, task_to_idx[task]] and human_moved_obj[end, task_to_idx[task]]:
                    human_task_frame_dict[f'demo_{task}'].append(human_data[i])
                


        # breakpoint()
        
        # # create a y1 array on length of human data length and divide into 2 parts with 0 and 1
        # num_human_splits = 2
        # len1 = human_data.shape[0] // num_human_splits
        # num_robot_splits = 4
        # len2 = robot_data.shape[0] // num_robot_splits
        # y = np.zeros(human_data.shape[0] + robot_data.shape[0])
        # for i in range(num_human_splits):
        #     y[i*len1: (i+1)*len1] = i
        # for i in range(num_robot_splits):
        #     y[human_data.shape[0] + i*len2: human_data.shape[0] + (i+1)*len2] = i + num_human_splits

    # breakpoint()
    # plot t-sne of human data array
    x, y  = [[]], []
    from sklearn.manifold import TSNE
    x, y  = [], []
    colors_used = {}
    # for task in robot_tasks_of_interest:
    #     # breakpoint()
    #     if len(robot_task_frame_dict[task]) == 0:
    #         continue
    #     sample = random.sample(robot_task_frame_dict[task], min(50, len(robot_task_frame_dict[task])))
    #     x.extend(sample)
    #     y.extend([colors[task]]*len(sample))
    #     # x.extend(robot_task_frame_dict[task])
    #     # y.extend([colors[task]]*len(robot_task_frame_dict[task]))
    #     colors_used[task] = colors[task]

    robot_len = len(x)
    for task in robot_tasks_of_interest:
        if len(human_task_frame_dict[f'demo_{task}']) == 0:
            continue
        print("im here")
        
        sample = random.sample(human_task_frame_dict[f'demo_{task}'], min(50, len(human_task_frame_dict[f'demo_{task}'])))
        x.extend(sample)
        y.extend([colors[f'demo_{task}']]*len(sample))
        # x.extend(human_task_frame_dict[f'{task[0]}_{task[1]}'])
        # y.extend([colors[f'{task[0]}_{task[1]}']]*len(human_task_frame_dict[f'{task[0]}_{task[1]}']))
        colors_used[f'demo_{task}'] = colors[f'demo_{task}']
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=10000)
    tsne_obj = tsne.fit_transform(np.array(x))
    plot(tsne_obj, y, robot_len, colors_used, cfg)
    # create a legend based on colors
    # plt.figure(figsize=(12, 8))
    # # in the next line, the scatter plot, mark with circle
    # plt.scatter(tsne_obj[: robot_len, 0], tsne_obj[:robot_len, 1], c=y[:robot_len], s=50, marker='o')
    # plt.scatter(tsne_obj[robot_len:, 0], tsne_obj[robot_len:, 1], c=y[robot_len:], s=50, marker='x')
    # # mark legend with colorr dict
    # for task, color in colors_used.items():
    #     plt.scatter([], [], c=color, label=task)
    # # put legend top left
    # plt.legend(loc='upper left')
    # # plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1])
    # # plt.savefig(os.path.join(cfg.exp_path, f"tsne_{cfg.ckpt}.png"))
    # plt.savefig(cfg.save_path)
    # plt.show()
    # breakpoint()

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset_easy",
)
def main(cfg: DictConfig):
    """
    Generates figures showing prototype activations based on subtask.


    Parameters
    ----------
    cfg : DictConfig
        Specifies configurations for vision encoder model that generates prototypes

    Side Effects
    ------------
    - Saves plots for prototype activations to exp_path folder

    Returns 
    -------
    None
    """
    # for demo_type in ['robot', 'human']:
        # plot_proto_task_relation(demo_type=demo_type, cfg=cfg)
    plot_tsne(cfg)
    
if __name__ == '__main__':
    main()