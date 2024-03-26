# import d4rl
import gym
import numpy as np
from PIL import Image
import os
from skimage.transform import resize
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import json
from xskill.dataset.kitchen_mjl_lowdim_dataset import KitchenMjlLowdimDataset
from xskill.env.kitchen.v0 import KitchenAllV0

ACTION_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

ACTION_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

KETTLE_INIT = np.array([-0.269, 0.35, 1.6192839, 1.0, 0.0, 0.0, 0.0])


def set_goal(positions, action_item, start_time, time_count):
    goal = ACTION_GOALS[action_item]
    action_index = ACTION_INDICES[action_item]
    for i in range(len(action_index)):
        position_index = action_index[i]
        goal_position = goal[i]
        change = np.linspace(
            positions[start_time][position_index], goal_position, num=time_count
        )
        positions[start_time : start_time + time_count, position_index] = change
        positions[start_time + time_count :, position_index] = goal_position
    return positions


def create_pos(
    actions=["bottom burner", "top burner"],
    durations=[40, 20],
):
    assert len(actions) == len(durations)
    eps_len = sum(durations)
    res = np.array([[0.0] * 30 for i in range(eps_len)], dtype="f")
    res[:, 23] = np.array([KETTLE_INIT[0]] * eps_len, dtype="f")
    res[:, 24] = np.array([KETTLE_INIT[1]] * eps_len, dtype="f")
    res[:, 25] = np.array([KETTLE_INIT[2]] * eps_len, dtype="f")
    res[:, 26] = np.array([KETTLE_INIT[3]] * eps_len, dtype="f")
    start_time = 0
    for task_index in range(len(actions)):
        res = set_goal(res, actions[task_index], start_time, durations[task_index])
        start_time = start_time + durations[task_index]
    return res


@hydra.main(
    version_base=None, config_path="../config/simulation", config_name="replay_kitchen"
)
def create_dataset(cfg: DictConfig):
    if cfg.embodiment == "robot":
        env = KitchenAllV0(use_abs_action=True, use_sphere_agent=False)
    elif cfg.embodiment == "human":
        env = KitchenAllV0(use_abs_action=True, use_sphere_agent=True)
    elif cfg.embodiment == "none":
        env = KitchenAllV0(use_abs_action=True, use_sphere_agent=False, use_none=True)
    else:
        raise NotImplementedError
    store_video, video_path = cfg.store_video, cfg.video_path
    if store_video:
        import imageio

    env.reset()
    frames = []

    reset_pos = create_pos()
    for i in range(len(reset_pos)):
        env.robot.reset(env, reset_pos[i], env.init_qvel[:].copy())
        image_observations = env.render(width=cfg.res, height=cfg.res)
        image_observations = Image.fromarray(image_observations)
        frames.append(image_observations)

    if store_video:
        video_filename = f"rollout_test_new.mp4"
        video_filepath = os.path.join(video_path, video_filename)
        # Save the frames as a video using imageio
        imageio.mimsave(video_filepath, frames, fps=30)

    env.close()


if __name__ == "__main__":
    create_dataset()
