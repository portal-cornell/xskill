import os
import pickle
import uuid
import hydra
import numpy as np
import torch
import torch.nn as nn
# import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
import random
from torch.utils.data import Dataset
from PIL import Image
import json

class ImgPredDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data = self.collect_data()
        
    def collect_data(self):
        data = []
        for folder in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder)
            if os.path.isdir(folder_path):
                img_filenames = [img for img in os.listdir(folder_path) if img.endswith('.png')]
                images = []
                for file in img_filenames:
                    img = Image.open(os.path.join(folder_path, file))
                    images.append(np.array(img))
                
                actions_file = os.path.join(folder_path, 'actions.json')
                with open(actions_file, 'r') as f:
                    actions = json.load(f)
                for i in range(len(images) - 1):
                    data.append({
                        'image': images[i],
                        'goal_image': images[-1],
                        'action': actions[i]
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="skill_transfer_composing",
)
def train_diffusion_bc(cfg: DictConfig):
    # create save dir
    unique_id = str(uuid.uuid4())
    save_dir = os.path.join(cfg.save_dir, unique_id)
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")
    # Set up logger
    # wandb.init(project=cfg.project_name)
    # wandb.config.update(OmegaConf.to_container(cfg))

    #set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    print("loading dataset")
    dataset = ImgPredDataset("datasets/small_kitchen_dataset/robot")
    print(len(dataset))
    print(dataset[0])




    # save training data statistics (min, max) for each dim
    # stats = dataset.stats
    # # open a file for writing in binary mode
    # with open(os.path.join(save_dir, "stats.pickle"), "wb") as f:
    #     # write the dictionary to the file
    #     pickle.dump(stats, f)

    # # create dataloader
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=cfg.batch_size,
    #     num_workers=cfg.num_workers,
    #     shuffle=True,
    #     # accelerate cpu-gpu transfer
    #     pin_memory=cfg.pin_memory,
    #     # don't kill worker process afte each epoch
    #     persistent_workers=cfg.persistent_workers,
    # )

    # # visualize data in batch
    # batch = next(iter(dataloader))
    # print("batch['obs'].shape:", batch["obs"].shape)
    # print("batch['actions'].shape", batch["actions"].shape)
    # print("batch['images'].shape", batch["images"].shape)



if __name__ == "__main__":
    train_diffusion_bc()