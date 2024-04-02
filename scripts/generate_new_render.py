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

OPENING_TASKS = [
    "slide cabinet",
    "full slide cabinet",
    "hinge cabinet",
    "full hinge cabinet",
    "microwave",
    "full microwave",
]

ACTION_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "full slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "full hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "full microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
    "lift kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

ACTION_GOALS = {
    "bottom burner": [np.array([-0.88, 0])],
    "top burner": [np.array([-0.92, 0])],
    "light switch": [np.array([-0.69, -0.05])],
    "slide cabinet": [np.array([0.37])],
    "full slide cabinet": [np.array([0.5])],
    "hinge cabinet": [np.array([0.0, 1.45])],
    "full hinge cabinet": [np.array([0.0, 3])],
    "microwave": [np.array([-0.75])],
    "full microwave": [np.array([-1.5])],
    "kettle": [np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06])],
    "lift kettle": [
        np.array([-0.26, 0.3, 1.9, 0.99, 0.0, 0.0, -0.06]),
        np.array([-0.26, 0.65, 1.8, 0.99, 0.0, 0.0, -0.06]),
        np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    ],
}

KETTLE_INIT = np.array([-0.269, 0.35, 1.62, 0.99, 0.0, 0.0, 0.0])


def ease_in_out_sine(x):
    return -(np.cos(np.pi * x) - 1) / 2


def ease_linear(x):
    return x


def ease_out_quad(x):
    return 1 - (1 - x) * (1 - x)


def interpolate(start, end, ease_function, duration):
    steps = (np.array(range(duration))) / (duration - 1)
    ease = np.vectorize(ease_function)
    easedValues = ease(steps)
    res = start + (end - start) * easedValues
    return res


def set_goal(
    positions,
    action_item,
    start_time,
    time_count,
    pauses,
    completion=1,
    easeFunction=ease_linear,
):
    goal = ACTION_GOALS[action_item]
    action_index = ACTION_INDICES[action_item]
    for i in range(len(action_index)):
        position_index = action_index[i]
        start = start_time
        for j in range(len(goal)):
            duration = time_count[j]
            pause = pauses[j]
            goal_position = goal[j][i] * float(completion)
            change = interpolate(
                positions[start][position_index], goal_position, easeFunction, duration
            )
            end_of_action = start + duration
            positions[start:end_of_action, position_index] = change
            positions[end_of_action:, position_index] = goal_position
            start = end_of_action + pause
    return positions


def create_pos(
    actions=[
        "bottom burner",
        "top burner",
        "hinge cabinet",
        "light switch",
        "microwave",
        "kettle",
        "slide cabinet",
    ],
    durations=[[40], [20], [20], [30], [40], [45], [20]],
    pause=[[25], [25], [25], [25], [25], [25], [25]],
    completions=[1, 1, 1],
    ease=ease_linear,
):
    assert len(actions) == len(durations)
    eps_len = np.sum(durations) + np.sum(pause)
    res = np.array([[0.0] * 30 for i in range(eps_len)], dtype="f")
    res[:, 23] = np.array([KETTLE_INIT[0]] * eps_len, dtype="f")
    res[:, 24] = np.array([KETTLE_INIT[1]] * eps_len, dtype="f")
    res[:, 25] = np.array([KETTLE_INIT[2]] * eps_len, dtype="f")
    res[:, 26] = np.array([KETTLE_INIT[3]] * eps_len, dtype="f")
    start_time = 0
    opening_ind = 0
    for task_index in range(len(actions)):
        if actions[task_index] in OPENING_TASKS:
            open_completion = completions[opening_ind]
            opening_ind = opening_ind + 1
        else:
            open_completion = 1
        res = set_goal(
            res,
            actions[task_index],
            start_time,
            durations[task_index],
            pause[task_index],
            open_completion,
            ease,
        )
        start_time = (
            start_time + np.sum(durations[task_index]) + np.sum(pause[task_index])
        )
    return res


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="generate_kitchen",
)
def create_dataset(cfg: DictConfig):
    env = KitchenAllV0(use_abs_action=True, use_sphere_agent=False, use_none=True)
    store_video, video_path = cfg.store_video, cfg.video_path
    if store_video:
        import imageio

    env.reset()
    frames = []

    test = create_pos(
        ["top burner", "full microwave"],
        [[15], [20]],
        [[10], [10]],
        [0.25],
        ease_out_quad,
    )
    reset_pos_full_microwave = create_pos(
        ["full microwave", "full slide cabinet", "full hinge cabinet"],
        [[20], [20], [20]],
        [[10], [10], [10]],
        [1, 1, 1],
        ease_out_quad,
    )
    reset_pos_full_microwave_halved = create_pos(
        ["full microwave", "full slide cabinet", "full hinge cabinet"],
        [[20], [20], [20]],
        [[10], [10], [10]],
        [0.5, 0.5, 0.5],
        ease_out_quad,
    )
    reset_pos_microwave = create_pos(
        ["microwave", "slide cabinet", "hinge cabinet"],
        [[20], [20], [20]],
        [[10], [10], [10]],
        [1, 1, 1],
        ease_out_quad,
    )

    all_sine = create_pos(ease=ease_in_out_sine)
    all_quad = create_pos(ease=ease_out_quad)
    all_linear = create_pos(ease=ease_linear)
    scenes = np.array(
        [
            # test,
            # reset_pos_full_microwave,
            # reset_pos_full_microwave_halved,
            # reset_pos_microwave,
            # all_linear,
            # all_quad,
            # all_sine,
        ],
        dtype=object,
    )

    episode_idx = 0
    for reset_pos in scenes:
        for i in range(len(reset_pos)):
            env.robot.reset(env, reset_pos[i], env.init_qvel[:].copy())
            image_observations = env.render(width=cfg.res, height=cfg.res)
            image_observations = Image.fromarray(image_observations)
            frames.append(image_observations)

        if store_video:
            video_filename = f"test_ease_elastic{episode_idx}.mp4"
            video_filepath = os.path.join(video_path, video_filename)
            # Save the frames as a video using imageio
            imageio.mimsave(video_filepath, frames, fps=30)
            frames = []
        episode_idx = episode_idx + 1
    env.close()


if __name__ == "__main__":
    create_dataset()
