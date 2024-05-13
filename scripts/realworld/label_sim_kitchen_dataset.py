import numpy as np
import os
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import torch
import omegaconf
from tqdm import tqdm
from xskill.utility.transform import get_transform_pipeline
import concurrent.futures
from pathlib import Path
import cv2
from PIL import Image
import json


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(
        eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(
        os.path.join(cfg.exp_path, '.hydra/config.yaml'))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f'epoch={cfg.ckpt}.ckpt')
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model

def load_state_and_to_tensor(vid):
    state_path = os.path.join(vid, "states.json")
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data = np.array(state_data, dtype=np.float32)
    return state_data

def load_images(folder_path, resize_shape=None):
    images = []  # initialize an empty list to store the images

    # get a sorted list of filenames in the folder
    filenames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # loop through all PNG files in the sorted list
    for filename in filenames:
        # open the image file using PIL library
        img = Image.open(os.path.join(folder_path, filename))
        # convert the image to a NumPy array
        img_arr = np.array(img)
        if resize_shape is not None:
            img_arr = cv2.resize(img_arr, resize_shape)
        images.append(img_arr)  # add the image array to the list

    # convert the list of image arrays to a NumPy array
    images_arr = np.array(images)
    return images_arr


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor

def label_data(cfg, model, data_path, pipeline, human=False):
    all_folders = os.listdir(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    for folder_path in tqdm(all_folders):
        save_folder = os.path.join(
                    cfg.exp_path, f"{'human_' if human else ''}encode_protos", f"ckpt_{cfg.ckpt}", os.path.basename(os.path.normpath(data_path)), folder_path
                )
        os.makedirs(save_folder, exist_ok=True)

        data_folder = os.path.join(data_path, folder_path)
        if not human:
            data_folder = os.path.join(data_folder, 'overhead')
        images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)
        
        images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

        eps_len = images_tensor.shape[0]
        im_q = torch.stack(
            [
                images_tensor[j : j + model.slide + 1]
                for j in range(eps_len - model.slide)
            ]
        )  # (b,slide+1,c,h,w)

        z = model.encoder_q(im_q, None)
        softmax_z = torch.softmax(z / model.T, dim=1)
        affordance_emb = model.skill_prior(im_q[:, : model.stack_frames], None)

        state_representation = model.encoder_q.get_state_representation(im_q, None)
        traj_representation = model.encoder_q.get_traj_representation(
            state_representation
        )
        traj_representation = repeat_last_proto(traj_representation, eps_len)
        traj_representation = traj_representation.detach().cpu().numpy()
        traj_representation = np.array(traj_representation).tolist()

        encode_protos = repeat_last_proto(z, eps_len)
        encode_protos = encode_protos.detach().cpu().numpy()
        encode_protos = np.array(encode_protos).tolist()

        softmax_encode_protos = repeat_last_proto(softmax_z, eps_len)
        softmax_encode_protos = softmax_encode_protos.detach().cpu().numpy()
        softmax_encode_protos = np.array(softmax_encode_protos).tolist()

        affordance_state_embs = affordance_emb.detach().cpu().numpy()
        affordance_state_embs = np.array(affordance_state_embs).tolist()

        with open(os.path.join(save_folder, "encode_protos.json"), "w") as f:
            json.dump(encode_protos, f)

        with open(
            os.path.join(save_folder, "softmax_encode_protos.json"), "w"
        ) as f:
            json.dump(softmax_encode_protos, f)

        with open(
            os.path.join(save_folder, "affordance_state_embs.json"), "w"
        ) as f:
            json.dump(affordance_state_embs, f)

        with open(os.path.join(save_folder, "traj_representation.json"), "w") as f:
            json.dump(traj_representation, f)



@hydra.main(version_base=None,
            config_path="../../config/realworld",
            config_name="label_sim_kitchen_dataset")
def label_dataset(cfg: DictConfig):

    model = load_model(cfg)
    pipeline = get_transform_pipeline(cfg.augmentations)

    resize_shape = cfg.resize_shape
    
    for data_path in cfg.robot_dataset._allowed_dirs:
        label_data(cfg, model, data_path, pipeline)
    for data_path in cfg.human_dataset._allowed_dirs:
        label_data(cfg, model, data_path, pipeline, human=True)




if __name__ == "__main__":
    label_dataset()
