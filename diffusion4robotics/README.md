# Basic commands for octo baselines

#### Preparing data

The goal of this step is to turn the data collected using manimo into a single buffer that is to be used for training. We assume the data is organized according to the following directory structure:

```
├── irl_squared_data
├── ├── sim_data
├── ├── ├── <task_name_1>
├── ├── ├── ├── optimal
├── ├── ├── ├── ├── traj_00001.pkl
├── ├── ├── ├── ├── traj_00002.pkl
├── ├── ├── ├── ├── ...
├── ├── ├── ├── truncate
├── ├── ├── ├── recovery
├── ├── ├── <task_name_2>
├── ├── real_data
├── ├── ├── <task_name_1>
├── ├── ├── ├── optimal
├── ├── ├── ├── truncate
├── ├── ├── ├── recovery
├── ├── ├── <task_name_2>
```

A brief description on the three buckets:
* `optimal` trajectories are performed by an expert that accomplishes the task efficiently and without error
* `truncate` trajectories do not complete the task at hand, and are cut short somepoint before
* `recovery` trajectories initially fail to complete the task (e.g. veers off course, knocks cup over), but eventually recovers and completes the task.

The first step is to combine all trajectories under a certain task into a single replay buffer. To do so, please run `combine_trajectories_to_buffer.py`, in which one can specific any combination of `optimal`, `truncate`, and `recovery` trajectories to include for the specified task.

For example, to create a combination of `optimal` and  `recovery` trajectories collected in the simulator on the `lift` task, one would run:
```
python combine_trajectories_to_buffer.py --root_dir /path/to/irl_squared_data --skill_name lift --is_sim --subdir optimal recovery
```

#### Visualizing training data

We also provide scripts to visualize the training data and image data, which one can see by running
```
python buffer_to_video.py --root_dir /path/to/irl_squared_data --skill_name lift --is_sim --name lift-optimal
```

#### Running training

For robobuf data conversion example, check out [this repo](https://github.com/AGI-Labs/r2d2_to_robobuf)

**Note:** If CPU image augmentation is a major bottleneck during data loading, set `train_transform=gpu_medium`. This will move data aug to the GPU. 

```
# vc-1 training command (velocity action space, gaussian mlp policy)
nice -n 19 python finetune.py agent/policy=gaussian_constant exp_name=octo_baselines wandb.name=vc1_baseline buffer_path=/path/to/vel/buf.pkl max_iterations=50000  task.train_buffer.cam_indexes=[<target_cam_id>] train_transform=hard agent.features.restore_path=/path/to/vc1.pth

# r3m training command (velocity action space, gaussian mlp policy)
nice -n 19 python finetune.py agent/features=r3m agent/policy=gaussian_constant exp_name=octo_baselines wandb.name=r3m_baseline buffer_path=/path/to/vel/buf.pkl max_iterations=50000  task.train_buffer.cam_indexes=[<target_cam_id>] train_transform=medium agent.features.size=50

# single-cam diffusion (position + r6 rotation action space)
nice -n 19 python finetune.py agent=diffusion_unet exp_name=octo_baselines wandb.name=diffusion_singlecam buffer_path=/path/to/abs_r6/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[<target_cam_id>] agent.features.feature_dim=256

# wrist-cam + 2-step obs diffusion (position + r6 rotation action space)
nice -n 19 python finetune.py agent=diffusion_unet exp_name=octo_baselines wandb.name=diffusion_multicam buffer_path=/path/to/abs_r6/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[<front_cam_id>, <wrist_cam_id>] task.train_buffer.cam_indexes=[0,2] img_chunk=2
```

# Eval Scripts

Checkout `eval_ptr.py` and `eval_diffusion.py` respectively for reference eval scripts.

Here is an example of `obs_config.yaml` (place in the checkpoint folder), used by the eval scripts
```
img: '26638268_left'
transform:
  _target_: data4robotics.transforms.get_transform_by_name
  name: preproc
```

# An Unbiased Look at Datasets for Visuo-Motor Pre-Training
[[Project Page]](https://data4robotics.github.io/)

This repository offers a minimal Behavior Cloning (BC) implementation using pre-trained representations from our CoRL project. All tests were conducted on a Franka Panda robot, using the [polymetis controller](https://facebookresearch.github.io/fairo/polymetis/). We've also verified that it works on the [R2D2 control stack](https://github.com/AlexanderKhazatsky/R2D2/tree/main).

If you find this codebase or our pre-trained representations useful at all, please cite:
```
@inproceedings{dasari2023datasets,
      title={An Unbiased Look at Datasets for Visuo-Motor Pre-Training},
      author={Dasari, Sudeep and Srirama, Mohan Kumar and Jain, Unnat and Gupta, Abhinav},
      booktitle={Conference on Robot Learning},
      year={2023},
      organization={PMLR}
    }
```
## Installation
Our repository is easy to install using miniconda or anaconda:

```
conda env create -f env.yml
conda activate data4robotics
pip install git+https://github.com/AGI-Labs/robobuf.git
pip install git+https://github.com/facebookresearch/r3m.git
pip install -e ./
```

## Using Pre-Trained Features
You can easily download our pre-trained represenations using the provided script: `./download_features.sh` 

The features are very modular, and easy to use in your own code-base! Please refer to the [example code](https://github.com/SudeepDasari/data4robotics/blob/main/pretrained_networks_example.py) if you're interested in this.

## Training BC Policies
First, you're going to need to convert your training trajectories into our [robobuf](https://github.com/AGI-Labs/robobuf/tree/main) format (pseudo-code below).
```
def _resize_and_encode(rgb_img, size=(256,256)):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded

def convert_trajectories(input_trajs, out_path):
    out_buffer = []
    for traj in tqdm(input_trajs):
        out_traj = []
        for in_obs, in_ac, in_reward in enumerate(data):
            out_obs = dict(state=np.array(in_obs['state']).astype(np.float32),
                           enc_cam_0=_resize_and_encode(in_obs['image']))
            out_action = np.array(in_ac).astype(np.float32)
            out_reward = float(in_reward)
            out_traj.append((out_obs, out_action, out_reward))
        out_buffer.append(out_traj)

    with open(os.path.join(out_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(out_trajs, f)
```

Once the conversion is complete, you can run the example command below:
```
python finetune.py exp_name=test agent.features.restore_path=/path/to/SOUP_1M_DH.pth buffer_path=/data/path/buffer.pkl
```
This will result in a policy checkpoint saved in the `bc_finetune/<exp_name>` folder.
