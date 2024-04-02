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
import numpy as np


def ease_in_out_sine(x):
    return -(np.cos(np.pi * x) - 1) / 2


def ease_linear(x):
    return x


def ease_out_quad(x):
    return 1 - (1 - x) * (1 - x)


def create_action(action, duration=[20], pause=[10], completion=1, ease=ease_out_quad):
    return {
        "action": action,
        "durations": duration,
        "pause": pause,
        "completion": completion,
        "ease": ease,
    }


lift_kettle_action = create_action(
    "lift kettle", [40, 20, 20], [10, 0, 0], ease=ease_in_out_sine
)

top_burner_action = create_action("top burner")
bottom_burner_action = create_action("bottom burner")
microwave_action = create_action("microwave")
kettle_action = create_action("kettle")
light_action = create_action("light switch")
slide_action = create_action("slide cabinet")
hinge_action = create_action("hinge cabinet")

full_hinge_action = create_action("full hinge cabinet")
full_slide_action = create_action("full slide cabinet")
full_microwave_action = create_action("full microwave")
half_microwave_action = create_action("full microwave", completion=0.5)
half_hinge_action = create_action("full hinge cabinet", completion=0.5)
quarter_slide_action = create_action("full slide cabinet", completion=0.25)

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
    "full hinge cabinet": [np.array([0.0, 1.7])],
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
    completion,
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
        bottom_burner_action,
        top_burner_action,
        hinge_action,
        light_action,
        microwave_action,
        kettle_action,
        slide_action,
    ],
):
    durations = 0
    pause = 0
    for i in range(len(actions)):
        durations += np.sum(actions[i]["durations"])
        pause += np.sum(actions[i]["pause"])
    eps_len = durations + pause
    res = np.array([[0.0] * 30 for i in range(eps_len)], dtype="f")
    res[:, 23] = np.array([KETTLE_INIT[0]] * eps_len, dtype="f")
    res[:, 24] = np.array([KETTLE_INIT[1]] * eps_len, dtype="f")
    res[:, 25] = np.array([KETTLE_INIT[2]] * eps_len, dtype="f")
    res[:, 26] = np.array([KETTLE_INIT[3]] * eps_len, dtype="f")
    start_time = 0
    for task_index in range(len(actions)):
        action = actions[task_index]["action"]
        duration = actions[task_index]["durations"]
        pause = actions[task_index]["pause"]
        open_completion = actions[task_index]["completion"]
        ease = actions[task_index]["ease"]
        res = set_goal(
            res,
            action,
            start_time,
            duration,
            pause,
            open_completion,
            ease,
        )
        start_time = (
            start_time + np.sum(actions[task_index]["durations"]) + np.sum(pause)
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

    burner_microwave = create_pos([top_burner_action, half_microwave_action])
    reset_pos_full_microwave = create_pos(
        [full_microwave_action, full_hinge_action, full_slide_action]
    )
    reset_pos_full_microwave_halved = create_pos(
        [half_hinge_action, half_microwave_action, quarter_slide_action]
    )
    reset_pos_microwave = create_pos(
        [microwave_action, slide_action, hinge_action, lift_kettle_action]
    )

    scenes = np.array(
        [
            burner_microwave,
            reset_pos_full_microwave,
            reset_pos_full_microwave_halved,
            reset_pos_microwave,
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
            video_filename = f"test_configs{episode_idx}.mp4"
            video_filepath = os.path.join(video_path, video_filename)
            # Save the frames as a video using imageio
            imageio.mimsave(video_filepath, frames, fps=30)
            frames = []
        episode_idx = episode_idx + 1
    env.close()


if __name__ == "__main__":
    create_dataset()
