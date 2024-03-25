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

# Create the environment


@hydra.main(
    version_base=None, config_path="../config/simulation", config_name="replay_kitchen"
)
def create_dataset(cfg: DictConfig):
    kitchen_dataset = KitchenMjlLowdimDataset(dataset_dir=cfg.dataset_dir)
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

    # total_episode = kitchen_dataset.replay_buffer.n_episodes
    # assert cfg.end_eps<=total_episode

    eps_data = kitchen_dataset.replay_buffer.get_episode(0)
    eps_len = len(eps_data["obs"])
    eps_data["obs"][:, 11] = np.array([2.4577741e-05] * 255, dtype="f")
    eps_data["obs"][:, 12] = np.array([2.9558922e-07] * 255, dtype="f")
    eps_data["obs"][:, 15] = np.array([2.4577741e-05] * 255, dtype="f")
    eps_data["obs"][:, 16] = np.array([2.9558922e-07] * 255, dtype="f")
    eps_data["obs"][:, 17] = np.array([2.1619626e-05] * 255, dtype="f")
    eps_data["obs"][:, 18] = np.array([5.0807366e-06] * 255, dtype="f")
    eps_data["obs"][:, 19] = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.00792082,
            0.01885369,
            0.03381699,
            0.05614009,
            0.08872676,
            0.12292692,
            0.15741105,
            0.1892251,
            0.2171601,
            0.24284855,
            0.2633817,
            0.28117293,
            0.29449543,
            0.3056916,
            0.31296116,
            0.31701618,
            0.3182473,
            0.3179537,
            0.3177452,
            0.31757164,
            0.3175716,
            0.31757164,
            0.2822859,
            0.24700016,
            0.21171443,
            0.17642869,
            0.14114295,
            0.10585721,
            0.07057148,
            0.03528574,
            0.0,
            0.0,
            0.00792082,
            0.01885369,
            0.03381699,
            0.05614009,
            0.08872676,
            0.12292692,
            0.15741105,
            0.1892251,
            0.2171601,
            0.24284855,
            0.2633817,
            0.28117293,
            0.29449543,
            0.3056916,
            0.31296116,
            0.31701618,
            0.3182473,
            0.3179537,
            0.3177452,
            0.31757164,
            0.3175716,
            0.31757164,
            0.3177452,
            0.3179537,
            0.3182473,
            0.31701618,
            0.31296116,
            0.3056916,
            0.29449543,
            0.28117293,
            0.2633817,
            0.24284855,
            0.2171601,
            0.1892251,
            0.15741105,
            0.12292692,
            0.08872676,
            0.05614009,
            0.03381699,
            0.01885369,
            0.00792082,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype="f",
    )
    eps_data["obs"][:, 20] = np.array([0.0] * 255, dtype="f")
    eps_data["obs"][:, 21] = np.array([0.0] * 255, dtype="f")
    eps_data["obs"][:, 22] = np.array([0.0] * 255, dtype="f")
    eps_data["obs"][:, 23] = np.array([-0.269] * 255, dtype="f")
    eps_data["obs"][:, 24] = np.array([0.35] * 255, dtype="f")
    eps_data["obs"][:, 25] = np.array([1.6192839] * 255, dtype="f")
    eps_data["obs"][:, 26] = np.array([1.0] * 255, dtype="f")
    eps_data["obs"][:, 27] = np.array([1.95423656e-19] * 255, dtype="f")
    eps_data["obs"][:, 28] = np.array([-1.13061060e-05] * 255, dtype="f")
    eps_data["obs"][:, 29] = np.array([-8.45423254e-19] * 255, dtype="f")
    for i in range(eps_len):
        reset_pos = np.concatenate([eps_data["obs"][i, :9], eps_data["obs"][i, 9:30]])
        env.robot.reset(env, reset_pos, env.init_qvel[:].copy())
        image_observations = env.render(width=cfg.res, height=cfg.res)
        image_observations = Image.fromarray(image_observations)
        frames.append(image_observations)

    if store_video:
        video_filename = f"rollout_test.mp4"
        video_filepath = os.path.join(video_path, video_filename)
        # Save the frames as a video using imageio
        imageio.mimsave(video_filepath, frames, fps=30)

    env.close()


if __name__ == "__main__":
    create_dataset()
