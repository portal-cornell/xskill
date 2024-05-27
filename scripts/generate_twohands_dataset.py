import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from xskill.env.kitchen.v0 import KitchenAllV0
import numpy as np
from collections import defaultdict
from actions import *
# base_dev_dir = "/share/portal/pd337"
import json
from xskill.utility.utils import read_json

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

HAND_GOALS = {
    "bottom burner": [
        np.array([-0.30, 0.40, 2.30]),
        np.array([-0.30, 0.40, 2.30])],
    "top burner": [
        np.array([-0.30, 0.40, 2.40]),
        np.array([-0.30, 0.40, 2.40])],
    "light switch": [
        np.array([-0.40, 0.50, 2.28]),
        np.array([-0.40, 0.50, 2.28])],
    "slide cabinet": [
        np.array([-0.1, 0.55, 2.6]),
        np.array([0.3, 0.55, 2.6])],
    "hinge cabinet": [
        np.array([-0.5, 0.55, 2.6]),
        np.array([-0.10, 0.30, 2.6])],
    "microwave": [
        np.array([-0.6, 0.3, 1.95]),
        np.array([-0.75, 0.1, 1.85])],
    "lift kettle": [
        np.array([-0.26, 0.3, 1.9]),
        np.array([-0.26, 0.3, 2.2]),
        np.array([-0.26, 0.65, 2.1]),
        np.array([-0.23, 0.75, 1.9]),
    ],
}

HAND_GOALS = {
    "bottom burner": [
        np.array([-0.30, 0.40, 2.30]),
        np.array([-0.30, 0.40, 2.30])],
    "top burner": [
        np.array([-0.30, 0.40, 2.40]),
        np.array([-0.30, 0.40, 2.40])],
    "light switch": [
        np.array([-0.40, 0.50, 2.28]),
        np.array([-0.40, 0.50, 2.28])],
    "slide cabinet": [
        np.array([-0.1, 0.55, 2.6]),
        np.array([0.3, 0.55, 2.6])],
    "hinge cabinet": [
        np.array([-0.5, 0.55, 2.6]),
        np.array([-0.10, 0.30, 2.6])],
    "microwave": [
        np.array([-0.6, 0.3, 1.95]),
        np.array([-0.75, 0.1, 1.85])],
    "lift kettle": [
        np.array([-0.26, 0.3, 1.9]),
        np.array([-0.26, 0.3, 2.2]),
        np.array([-0.26, 0.65, 2.1]),
        np.array([-0.23, 0.75, 1.9]),
    ],
}
def randomize_handgoals():
    global HAND_GOALS
    for key in HAND_GOALS.keys():
        if key == "bottom burner":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.30+xnoise, 0.40, 2.30+znoise]),
                np.array([-0.30+xnoise, 0.40, 2.30+znoise])]
        elif key == "top burner":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.30+xnoise, 0.40, 2.40+znoise]),
                np.array([-0.30+xnoise, 0.40, 2.40+znoise])]
        elif key == "light switch":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.40+xnoise, 0.50, 2.28+znoise]),
                np.array([-0.40+xnoise, 0.50, 2.28+znoise])]
        elif key == "slide cabinet":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.1+xnoise, 0.55, 2.6+znoise]),
                np.array([0.3+xnoise, 0.55, 2.6+znoise])]
        elif key == "hinge cabinet":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.5+xnoise, 0.55, 2.6+znoise]),
                np.array([-0.10+xnoise, 0.30, 2.6+znoise])]
        elif key == "microwave":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.6+xnoise, 0.3, 1.95+znoise]),
                np.array([-0.75+xnoise, 0.1, 1.85+znoise])]
        elif key == "lift kettle":
            xnoise = np.random.uniform(-0.025, 0.025)
            znoise = np.random.uniform(-0.025, 0.025)
            HAND_GOALS[key] = [
                np.array([-0.26+xnoise, 0.3, 1.9+znoise]),
                np.array([-0.26+xnoise, 0.3, 2.2+znoise]),
                np.array([-0.26+xnoise, 0.65, 2.1+znoise]),
                np.array([-0.23+xnoise, 0.75, 1.9+znoise]),
            ]

KETTLE_INIT = np.array([-0.269, 0.35, 1.62, 0.99, 0.0, 0.0, 0.0])
HAND_POS = np.array([-0.48, 0.10, 2.05, -0.38, 0.10, 2.05])
LEFTHAND_POS = np.array([-0.48, 0.10, 2.05])
INIT_LEFTHAND_POS = np.array([-0.48, 0.10, 2.05])
RIGHTHAND_POS = np.array([-0.38, 0.10, 2.05])
INIT_RIGHTHAND_POS = np.array([-0.38, 0.10, 2.055])


def interpolate(start, end, ease_function, duration):
    """
    Interpolate between the start and end points using a desired ease function
    """
    # breakpoint()
    steps = (np.array(range(duration))) / (duration - 1)
    # ease = np.vectorize(ease_function)
    # breakpoint()
    # try:
    #     easedValues = ease(steps)
    # except:
    #     breakpoint()
    easedValues = steps
    # vectorize next line
    # res = start + (end - start) * easedValues
    res = np.array([start + (end - start) * easedValues[i] for i in range(len(easedValues))])
    return res


def set_goal(
    positions,
    handpos,
    active_hand,
    action_item,
    start_time,
    time_count,
    pauses,
    completion,
    easeFunction=ease_linear,
):
    """
    Updates positions and orientations in the environment according to an action

    Parameters:
    ----------
    positions (ndarray): 2D array containing coordinates of the placement of objects in the kitchen
    action_item (set): dictionary of the action to be performed
    start_time (int): the number of time units from the beginning of the simulation to start performing the action
    time_count (int list): the number of time units it takes to perform each subaction
    pauses (int list): the number of time units to pause after each subaction
    completion (float): the fraction to open the object for hinge action, slide action, and microwave action

    Returns:
    ----------
    numpy array updated with this action
    """
    global LEFTHAND_POS, RIGHTHAND_POS
    goal = ACTION_GOALS[action_item]
    action_index = ACTION_INDICES[action_item]


    for i in range(len(action_index)):
        position_index = action_index[i]
        start = start_time
        for j in range(len(goal)):
            duration = time_count[j]
            pause = pauses[j]
            goal_position = goal[j][i] * float(completion)
            # breakpoint()
            if i==0:
                if active_hand=="left":
                    if pause > 0:
                        hand_movement_start = interpolate(LEFTHAND_POS, HAND_GOALS[action_item][j], easeFunction, pause)
                        handpos[start:start+pause, :3] = hand_movement_start
                    else:
                        handpos[start:start+pause, :3] = LEFTHAND_POS
                    hand_movement_goal = interpolate(HAND_GOALS[action_item][j], HAND_GOALS[action_item][j+1], easeFunction, duration)
                    LEFTHAND_POS = HAND_GOALS[action_item][j+1]
                    handpos[start+pause:start+pause+duration, :3] = hand_movement_goal
                else:
                    if pause > 0:
                        hand_movement_start = interpolate(RIGHTHAND_POS, HAND_GOALS[action_item][j], easeFunction, pause)
                        handpos[start:start+pause, 3:] = hand_movement_start
                    else:
                        handpos[start:start+pause, 3:] = RIGHTHAND_POS
                    hand_movement_goal = interpolate(HAND_GOALS[action_item][j], HAND_GOALS[action_item][j+1], easeFunction, duration)
                    RIGHTHAND_POS = HAND_GOALS[action_item][j+1]
                    handpos[start+pause:start+pause+duration, 3:] = hand_movement_goal
                    

            change = interpolate(
                positions[start][position_index], goal_position, easeFunction, duration
            )
            end_of_action = start + duration + pause
            positions[start:start+pause, position_index] = positions[start, position_index]
            positions[start+pause:end_of_action, position_index] = change
            positions[end_of_action:, position_index] = goal_position
            start = end_of_action
    return positions, handpos


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
    order=[0, 1, 2, 3, 4, 5, 6],
):
    """
    create the positions to render a series of actions

    Parameters:
    ----------
    actions (dict list): each dictionary represents an action to be performed
    order (int list): the order to perform the list of actions. the action at index i will be performed order[i] into the sequence. Use duplicate values to indicate actions starting at the same time.

    Returns:
    ----------
    numpy array representing the full sequence of actions
    """
    assert len(order) == len(actions)
    # breakpoint()
    groups = defaultdict(list)
    for ind in range(len(actions)):
        groups[order[ind]].append(actions[ind])
    action_groups = sorted(groups.items())
    action_groups = [list(i) for i in zip(*action_groups)][1]
    durations = []
    for group in action_groups:
        duration = 0
        pause = 0
        for action in group:
            duration = max(
                duration, np.sum(action["durations"]) + np.sum(action["pause"])
            )
        durations.append(duration)
        # breakpoint()
    eps_len = np.sum(durations)
    res = np.array([[0.0] * 30 for i in range(eps_len)], dtype="f")
    handpos = np.array([[0.0] * 6 for i in range(eps_len)], dtype="f")
    handpos[:, :3] = LEFTHAND_POS
    handpos[:, 3:] = RIGHTHAND_POS
    # res[:, :7] = init_qpos
    res[:, 23] = np.array([KETTLE_INIT[0]] * eps_len, dtype="f")
    res[:, 24] = np.array([KETTLE_INIT[1]] * eps_len, dtype="f")
    res[:, 25] = np.array([KETTLE_INIT[2]] * eps_len, dtype="f")
    res[:, 26] = np.array([KETTLE_INIT[3]] * eps_len, dtype="f")
    start_time = 0

    for group_number in range(len(action_groups)):
        # for action in action_groups[group_number]:
        lefthand_righthand_assign_dist = 0
        left_action = action_groups[group_number][0]["action"]
        left_goal = HAND_GOALS[left_action][0]
        lefthand_righthand_assign_dist += np.linalg.norm(LEFTHAND_POS - left_goal)
        if len(action_groups[group_number]) == 2:
            right_action = action_groups[group_number][1]["action"]
            right_goal = HAND_GOALS[right_action][0]
            lefthand_righthand_assign_dist += np.linalg.norm(RIGHTHAND_POS - right_goal)
        else:
            lefthand_righthand_assign_dist += 0

        righthand_left_assign_dist = 0
        if len(action_groups[group_number]) == 2:
            left_action = action_groups[group_number][1]["action"]
            left_goal = HAND_GOALS[left_action][0]
            righthand_left_assign_dist += np.linalg.norm(LEFTHAND_POS - left_goal)
        else:
            righthand_left_assign_dist += 0
        right_action = action_groups[group_number][0]["action"]
        right_goal = HAND_GOALS[right_action][0]
        righthand_left_assign_dist += np.linalg.norm(RIGHTHAND_POS - right_goal)

        if lefthand_righthand_assign_dist < righthand_left_assign_dist:
            left_first = True
        else:
            left_first = False
        # for action in action_groups[group_number]:
        for i in range(len(action_groups[group_number])):
            action = action_groups[group_number][i]
            if i == 0 and left_first or i==1 and not left_first:
                active_hand = "left"
            else:
                active_hand = "right"
            action_name = action["action"]
            duration = action["durations"]
            pause = action["pause"]
            # breakpoint()
            open_completion = action["completion"]
            ease = action["ease"]
            res, handpos = set_goal(
                res,
                handpos,
                active_hand,
                action_name,
                start_time,
                duration,
                pause,
                open_completion,
                ease,
            )
        if len(action_groups[group_number]) == 1:
            if left_first:
                handpos[start_time:start_time + durations[group_number], 3:] = RIGHTHAND_POS
            else:
                handpos[start_time:start_time + durations[group_number], :3] = LEFTHAND_POS
        start_time = start_time + durations[group_number]
    # breakpoint()
    return res, handpos


# @hydra.main(
#     version_base=None,
#     config_path="../config/simulation",
#     config_name="generate_kitchen",
# )
# def create_renders(cfg: DictConfig):
#     env = KitchenAllV0(use_abs_action=True, use_sphere_agent=False, use_none=True)
#     store_video, video_path = cfg.store_video, cfg.video_path
#     if store_video:
#         import imageio

#     env.reset()
#     frames = []

#     # create some example action sequences
#     burner_microwave = create_pos([top_burner_action, half_microwave_action], [1, 2])
#     reset_pos_full_microwave = create_pos(
#         [full_microwave_action, full_hinge_action, full_slide_action], [1, 2, 1]
#     )
#     reset_pos_full_microwave_halved = create_pos(
#         [half_hinge_action, half_microwave_action, quarter_slide_action], [3, 2, 1]
#     )
#     reset_pos = create_pos(
#         [
#             half_microwave_action,
#             quarter_slide_action,
#             top_burner_action,
#             lift_kettle_action,
#             light_action,
#             fast_hinge_action,
#         ],
#         [1, 2, 3, 3, 5, 4],
#     )

#     everything_everywhere_all_at_once = create_pos(order=[1, 1, 1, 1, 1, 1, 1])

#     # render each action sequence in a different mp4 file
#     scenes = np.array(
#         [
#             # burner_microwave,
#             # reset_pos_full_microwave,
#             # reset_pos_full_microwave_halved,
#             reset_pos,
#             # everything_everywhere_all_at_once,
#         ],
#         dtype=object,
#     )

#     # renders and saves each action sequence in scenes
#     episode_idx = 0
#     for reset_pos in scenes:
#         for i in range(len(reset_pos)):
#             env.robot.reset(env, reset_pos[i], env.init_qvel[:].copy())
#             image_observations = env.render(width=cfg.res, height=cfg.res)
#             image_observations = Image.fromarray(image_observations)
#             frames.append(image_observations)

#         if store_video:
#             video_filename = f"test_configs_{episode_idx}.mp4"
#             video_filepath = os.path.join(video_path, video_filename)
#             # Save the frames as a video using imageio
#             imageio.mimsave(video_filepath, frames, fps=30)
#             frames = []
#         episode_idx = episode_idx + 1
#     env.close()

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="generate_kitchen",
)
def generate_render(cfg: DictConfig):
    global LEFTHAND_POS, RIGHTHAND_POS, INIT_LEFTHAND_POS, INIT_RIGHTHAND_POS
    task_completions_list = read_json(f"/share/portal/kk837/xskill/datasets/kitchen_dataset/task_completions.json")
    # breakpoint()
    for eps_idx, task_list in enumerate(task_completions_list):
        randomize_handgoals()
        LEFTHAND_POS = INIT_LEFTHAND_POS
        RIGHTHAND_POS = INIT_RIGHTHAND_POS
        actions = []
        for task in task_list:
            if task == "kettle":
                actions.append(lift_kettle_action)
            elif task == "light switch":
                actions.append(light_action)
            elif task == "slide cabinet":
                actions.append(slide_action)
            elif task == "microwave":
                actions.append(microwave_action)
            elif task == "hinge cabinet":
                actions.append(hinge_action)
            elif task == "bottom burner":
                actions.append(bottom_burner_action)
            elif task == "top burner":
                actions.append(top_burner_action)
            else:
                raise NotImplementedError
        # actions.append(microwave_action)
        # actions.append(top_burner_action)
        # actions.append(bottom_burner_action)
        env = KitchenAllV0(use_abs_action=True, 
                           use_sphere_agent=False, 
                           use_none=False,
                           use_single_hand=False,
                           use_two_hands=True)
        # video_path = cfg.video_path
        # breakpoint()
        env.reset()
        frames = []
        # breakpoint()
        # actions.pop()
        # breakpoint()
        if len(actions) == 3:
            reset_pos, handpos = create_pos(actions, [1, 1, 3])
        else:
            reset_pos, handpos = create_pos(actions, [1, 1, 3, 3])

        for i in range(len(reset_pos)):
            env.sim.model.body_pos[-2][:3] = handpos[i][:3]
            env.sim.model.body_pos[-1][:3] = handpos[i][3:]
            env.robot.reset(env, reset_pos[i], env.init_qvel[:].copy())
            image_observations = env.render(width=cfg.res, height=cfg.res)
            image_observations = Image.fromarray(image_observations)
            video_filepath = os.path.join(cfg.video_path, f'{eps_idx}/{i}.png')
            os.makedirs(os.path.join(cfg.video_path, f'{eps_idx}'), exist_ok=True)
            image_observations.save(video_filepath)
        env.close()

        eps_path = os.path.join(cfg.video_path, f'{eps_idx}')
        with open(os.path.join(eps_path, "states.json"), "w") as f:
            json.dump(reset_pos.tolist(), f)

        # for i in range(len(reset_pos)):
        # # for i in range(100):
        #     env.sim.model.body_pos[-2][:3] = handpos[i][:3]
        #     env.sim.model.body_pos[-1][:3] = handpos[i][3:]
        #     env.robot.reset(env, reset_pos[i], env.init_qvel[:].copy())
        #     image_observations = env.render(width=400, height=400)
        #     image_observations = Image.fromarray(image_observations)
        #     frames.append(image_observations)
        # # make a gif grom frames
        #     # video_filename = f"test_configs_{eps_idx}.mp4"
        #     # video_filepath = os.path.join(cfg.video_path, video_filename)
        # Save the frames as a video using imageio
        # frames[0].save("test.gif", save_all=True, 
        # append_images=frames[1:], optimize=False, duration=100, loop=0)


            # video_filepath = os.path.join(video_path, f'{eps_idx}/{i}.png')
            # os.makedirs(os.path.join(video_path, f'{eps_idx}'), exist_ok=True)
            # image_observations.save(video_filepath)
        # env.close()
        # break

if __name__ == "__main__":
    # create_renders()
    generate_render()

        

