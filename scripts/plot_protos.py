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

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
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
    for demo_type in ['robot', 'human']:
        plot_proto_task_relation(demo_type=demo_type, cfg=cfg)
        proto_scatter_plot(cfg.exp_path, demo_type)
    
if __name__ == '__main__':
    main()