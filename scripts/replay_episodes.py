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


@hydra.main(version_base=None,
            config_path="../config/simulation",
            config_name="replay_kitchen")
def create_dataset(cfg: DictConfig):
    kitchen_dataset = KitchenMjlLowdimDataset(dataset_dir=cfg.dataset_dir)
    if cfg.embodiment == 'robot':
        env = KitchenAllV0(use_abs_action=True, use_sphere_agent=False)
    elif cfg.embodiment == 'human':
        env = KitchenAllV0(use_abs_action=True, use_sphere_agent=True)
    else:
        raise NotImplementedError

    env.reset()

    total_episode = kitchen_dataset.replay_buffer.n_episodes
    # assert cfg.end_eps<=total_episode

    for eps_idx in tqdm(range(cfg.start_eps, cfg.end_eps)):
        eps_data = kitchen_dataset.replay_buffer.get_episode(eps_idx)
        eps_len = len(eps_data['obs'])
        for i in range(eps_len):
            reset_pos = np.concatenate(
                [eps_data['obs'][i, :9], eps_data['obs'][i, 9:30]])
            env.robot.reset(env, reset_pos, env.init_qvel[:].copy())
            image_observations = env.render(width=cfg.res, height=cfg.res)
            image_observations = Image.fromarray(image_observations)
            env.render(custom=False)


if __name__ == "__main__":
    create_dataset()
