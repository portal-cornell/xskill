import numpy as np
from PIL import Image
import os
from skimage.transform import resize
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import json
import torch
import torchvision.transforms as Tr
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch import nn, einsum
import omegaconf
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import io


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


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


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor

def gif_of_clip(cfg, demo_type, ep_num, clip_num, slide, output_dir):
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    sub_imgs = images_arr[clip_num:clip_num + slide + 1]

    pil_imgs = [Image.fromarray(img) for img in sub_imgs]
    pil_imgs[0].save(os.path.join(output_dir, f'{demo_type}_{ep_num}_index{clip_num}.gif'), save_all=True, append_images=pil_imgs[1:], duration=200, loop=0)

    return pil_imgs

def load_gif(cfg, demo_type, ep_num):
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)
    pil_imgs = [Image.fromarray(img) for img in images_arr]
    return pil_imgs

def traj_representations(cfg, model, pipeline, demo_type, ep_num, frame_list=None):
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

    eps_len = images_tensor.shape[0]
    im_q = torch.stack(
        [
            images_tensor[j : j + model.slide + 1]
            for j in range(eps_len - model.slide)
        ]
    )  # (b,slide+1,c,h,w)

    clips = im_q
    if frame_list is not None:
        clips = clips[frame_list]

    z = model.encoder_q(im_q, None)
    state_rep = model.encoder_q.get_state_representation(clips, None)
    traj_rep = model.encoder_q.get_traj_representation(state_rep)

    return traj_rep, z



@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
)
def label_dataset(cfg: DictConfig):
    model = load_model(cfg)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

    clip_list = [20, 150, 247]
    frame_nums = [0, 40, 180]

    tasks_list = json.load(open(f'{cfg.data_path}/task_completions.json'))
    clip_diff = 0
    for clip_num in clip_list:
        for frame_num in frame_nums:
            output_dir = f'gif_outputs{"_diff" if clip_diff != 0 else ""}/{clip_num}_{frame_num}'
            os.makedirs(output_dir, exist_ok=True)
            human_clip_num = clip_num - clip_diff
            robot_clip_traj, robot_z = traj_representations(cfg, model, pipeline, 'robot', clip_num, frame_list=[frame_num])
            robot_gif = gif_of_clip(cfg, 'robot', clip_num, frame_num, 8, output_dir)
            human_clip_traj, human_z = traj_representations(cfg, model, pipeline, 'human', human_clip_num)
            dists = torch.norm(human_clip_traj - robot_clip_traj, dim=1)
            print(f"dists shape: {robot_clip_traj.shape}")

            closest_clip = torch.argmin(dists)
            human_gif = gif_of_clip(cfg, 'human', human_clip_num, closest_clip.item(), 8, output_dir)
            dictionary = {
                'robot_clip_num': clip_num,
                'human_clip_num': human_clip_num,
                'robot_clip_tasks': tasks_list[clip_num],
                'human_clip_tasks': tasks_list[human_clip_num]
            }
            with open(os.path.join(output_dir, "output.json"), "w") as outfile:
                json.dump(dictionary, outfile)

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
)
def visualize_nn(cfg: DictConfig):
    model = load_model(cfg)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

    clip_list = [247]
    frame_nums = [180]

    tasks_list = json.load(open(f'{cfg.data_path}/task_completions.json'))
    clip_diff = 0
    for clip_num in clip_list:
        for frame_num in frame_nums:
            output_dir = f'similarity_outputs{"_diff" if clip_diff != 0 else ""}/{clip_num}_{frame_num}'
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
            
            os.makedirs(output_dir, exist_ok=True)
            human_clip_num = clip_num - clip_diff
            robot_clip_traj, robot_z = traj_representations(cfg, model, pipeline, 'robot', clip_num, frame_list=[frame_num])
            human_clip_traj, human_z = traj_representations(cfg, model, pipeline, 'human', human_clip_num)
            dists = torch.norm(human_clip_traj - robot_clip_traj, dim=1)
            dists = dists.cpu().detach().numpy()
            num_top_dists = 3
            sorted_dists = np.argsort(dists)[:num_top_dists]
            sorted_dists = np.sort(sorted_dists)

            gif_of_clip(cfg, 'robot', clip_num, frame_num, 8, output_dir)
            robot_gif = Image.open(os.path.join(output_dir, f'robot_{clip_num}_index{frame_num}.gif'))
            # robot_gif = robot_gif.resize(((int)(robot_gif.size[0]/2), (int)(robot_gif.size[1]/2)))
            bars = plt.bar(range(len(dists)), dists, color='skyblue')
            for i in range(len(sorted_dists)):
                bars[sorted_dists[i]].set_color('orange')
            plt.axis('off')
            buf = io.BytesIO()
            buf.seek(0)
            buf.truncate(0)
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            img.paste(robot_gif, ((int)(img.size[0]/2), (int)(img.size[1]/8)))

            # # load images from human video
            # human_video_imgs = load_gif(cfg, 'human', clip_num)
            # final_gif = []
            # # edit matplot graph to highlight corresponding bar
            # for i in range(len(human_video_imgs)):
            #     print(i)
            #     bars = plt.bar(len(dists), dists, color='skyblue')
            #     bars[i].set_color('orange')
            #     plt.axis('off')
            #     buf = io.BytesIO()
            #     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            #     buf.seek(0)
            #     cur_frame = Image.open(buf)
            #     cur_frame.paste(robot_gif, ((int)(cur_frame.size[0]/2), (int)(cur_frame.size[1]/8)))
            #     # cur_frame.paste(human_video_imgs[i], ((int)(cur_frame.size[0]/4), (int)(cur_frame.size[1]/8)))
            #     final_gif.append(cur_frame)

                
            # # add to list of images in GIF

            

            draw = ImageDraw.Draw(img)
            for i in range(num_top_dists):
                clip_idx = sorted_dists[i]
                gif_of_clip(cfg, 'human', human_clip_num, clip_idx, 8, output_dir)
                human_gif = Image.open(os.path.join(output_dir, f'human_{clip_num}_index{clip_idx}.gif'))
                human_gif = human_gif.resize(((int)(img.size[0]/5), (int)(img.size[1]/5)))

                bar_x = (int)(img.size[0] / 2 + 2 * (clip_idx - len(dists) / 2))
                bar_y = (int)(img.size[1] * 8/9)
                human_gif_x = (int)(img.size[0]/4 + img.size[0]/2 * (i / num_top_dists))
                human_gif_y = (int)(img.size[1] * 3/4)
                img.paste(human_gif, (human_gif_x, human_gif_y))
                # draw.line([(human_gif_x + human_gif.size[0]/2, human_gif_y), (bar_x, bar_y)], fill="black", width=2)
            # img[0].save(os.path.join(output_dir, "plot.gif"), format='GIF', save_all=True, append_images=final_gif[1:], loop=0)
            img.save(os.path.join(output_dir, "plot.gif"))

if __name__ == "__main__":
    visualize_nn()