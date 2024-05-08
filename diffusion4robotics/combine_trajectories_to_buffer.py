import argparse
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union

from pathlib import Path
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
import json



import pytorch3d.transforms as pt
import torch
import numpy as np
import functools

from eval_diffusion import rot6d_to_euler


class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def _gaussian_norm(all_acs):
    print('Using gaussian norm')
    all_acs_arr = np.array(all_acs)
    mean = np.mean(all_acs_arr, axis=0)
    std =  np.std(all_acs_arr, axis=0)
    if not std.all(): # handle situation w/ all 0 actions
        std[std == 0] = 1e-17

    for a in all_acs:
        a -= mean
        a /= std

    return dict(loc=mean.tolist(), scale=std.tolist())


def _max_min_norm(all_acs):
    print('Using max min norm')
    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr, axis=0)
    min_ac = np.min(all_acs_arr, axis=0)

    for a in all_acs:
        #a -= mid
        #a /= delta
        a -= min_ac
        a /= (max_ac - min_ac)
        a = a*2 - 1
    return dict(minimum = min_ac, maximum=max_ac)


def process_traj(traj):
    res = []

    for step in traj[0]:
        obs, act, reward = step
        d = dict()
        d['state'] = obs['state']
        d['actor'] = obs['actor']
        d['cam_rs'] = obs['cam_rs'][0]
        d['cam_zed_right'] = obs['cam_zed_right'][0]

        res.append((d, act, reward))

    return [res]

def convert_trajectories(
    pkl_files: List[Path], output_path: Path, image_postprocess: Any = None, remove_images: bool = False
):
    """
    Combine individual pickle trajectories into a single buffer
    """
    print(f"Working with {len(pkl_files)} files and saving to {output_path}")

    tf = RotationTransformer('quaternion', 'rotation_6d')
    out_buffer = None
    all_acs = []
    all_obs = []
    for file in tqdm(pkl_files, total=len(pkl_files)):
        with open(file, "rb") as f:
            traj = pickle.load(f)
            #print(traj)
            traj_buffer = ReplayBuffer.load_traj_list(process_traj(traj))
            if out_buffer:
                out_buffer.append_traj_list(traj_buffer.to_traj_list())
            else:
                out_buffer = traj_buffer

    if remove_images:
        new_traj_list = []
        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Processing stats",
            total=len(out_buffer._traj_starts),
        ):
            new_traj = ReplayBuffer()
            isFirst = True
            print(len(old_traj))
            for i, (old_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
                    new_obs = old_obs.copy()
                    original_keys = list(new_obs.keys())
                    for obs_key in original_keys:
                        if "cam" in obs_key:
                            new_obs.pop(obs_key)
                    all_obs.append(new_obs['state'])
                    all_acs.append(action)
                    

            all_acs_arr = np.array(all_acs)
            max_ac = np.max(all_acs_arr.copy(), axis=0)
            min_ac = np.min(all_acs_arr.copy(), axis=0)
            all_obs_arr = np.array(all_obs)
            max_ob = np.max(all_obs_arr.copy(), axis=0)
            min_ob = np.min(all_obs_arr.copy(), axis=0)
        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Removing images",
            total=len(out_buffer._traj_starts),
        ):
            for i, (old_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
                    new_obs = old_obs.copy()
                    original_keys = list(new_obs.keys())
                    for obs_key in original_keys:
                        if "cam" in obs_key:
                            new_obs.pop(obs_key)

                    
                    new_obs['state'] = (new_obs['state'] - min_ob)/(max_ob-min_ob)
                    
                    action = (action - min_ac)/(max_ac-min_ac)
                    new_obs['state'] = new_obs['state']*2 - 1
                    actionoptimal = action*2 - 1
                    new_traj.add(
                        Transition(ObsWrapper(new_obs), action, reward),
                        is_first=(isFirst),
                    )
                    isFirst = False
            new_traj_list += new_traj.to_traj_list()
        out_buffer = ReplayBuffer.load_traj_list(new_traj_list)
    
    elif image_postprocess:
        new_traj_list = []
        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Processing stats",
            total=len(out_buffer._traj_starts),
        ):
            for i, (old_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
    
                    all_obs.append(old_obs['state'])
                    all_acs.append(action)
        all_acs_arr = np.array(all_acs)
        max_ac = np.max(all_acs_arr.copy(), axis=0)
        min_ac = np.min(all_acs_arr.copy(), axis=0)
        all_obs_arr = np.array(all_obs)
        max_ob = np.max(all_obs_arr.copy(), axis=0)
        min_ob = np.min(all_obs_arr.copy(), axis=0)
        


        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Postprocessing",
            total=len(out_buffer._traj_starts),
        ):
            new_traj = ReplayBuffer()
            isFirst = True
            for i, (old_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
                    new_obs = {}
                    for obs_key in old_obs.keys():
                        if "cam" in obs_key:
                            new_obs[obs_key.replace("enc_", "")] = image_postprocess(
                                old_obs[obs_key]
                            )
                        #elif obs_key == 'state' and len(old_obs[obs_key]) > 9:
                        #    # EEF Pose(3), EEF Rot(3), Gripper pos (2), gripper width (1)
                        #    new_obs[obs_key] = old_obs[obs_key][7:]  
                        else:
                            new_obs[obs_key] = old_obs[obs_key]
                    
                    new_obs['state'] = (new_obs['state'] - min_ob)/(max_ob-min_ob)
                    action = (action - min_ac)/(max_ac-min_ac)
                    new_obs['state'] = new_obs['state']*2 - 1
                    action = action*2 - 1
                    new_traj.add(
                        Transition(ObsWrapper(new_obs), action, reward),
                        is_first=(isFirst),
                    )
                    isFirst = False
            new_traj_list += new_traj.to_traj_list()
        out_buffer = ReplayBuffer.load_traj_list(new_traj_list)
    
    else:
        new_traj_list = []
        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Processing stats",
            total=len(out_buffer._traj_starts),
        ):
            for i, (old_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
    
                    all_obs.append(old_obs['state'])
                    pos, angle, grip = action[:3], action[3:7], action[[7]]
                    r6_angle = tf.forward(angle)
                    angle_new = tf.inverse(r6_angle)
                    action = np.concatenate((pos, r6_angle, grip))
                    all_acs.append(action)
        all_acs_arr = np.array(all_acs)
        mean_ac = np.mean(all_acs_arr.copy(), axis=0)
        std_ac = np.std(all_acs_arr.copy(), axis=0)
        max_ac = np.max(all_acs_arr.copy(), axis=0)
        max_ac[3:9] = 1
        min_ac = np.min(all_acs_arr.copy(), axis=0)
        min_ac[3:9] = -1
        all_obs_arr = np.array(all_obs)
        max_ob = np.max(all_obs_arr.copy(), axis=0)
        min_ob = np.min(all_obs_arr.copy(), axis=0)

        for old_traj in tqdm(
            out_buffer.to_traj_list(),
            desc="Postprocessing",
            total=len(out_buffer._traj_starts),
        ):
            new_traj = ReplayBuffer()
            isFirst = True
            for i, (new_obs, action, reward) in enumerate(old_traj):
                if not (action[0:6] == [0.,0.,0.,0.,0.,0.]).all():
                    new_obs['state'] = (new_obs['state'] - min_ob)/(max_ob-min_ob)
                    pos, angle, grip = action[:3], action[3:7], action[[7]]
                    r6_angle = tf.forward(angle)
                    angle_new = tf.inverse(r6_angle)
                    action = np.concatenate((pos, r6_angle, grip))
                    action = (action - min_ac)/(max_ac-min_ac)
                    new_obs['state'] = new_obs['state']*2 - 1
                    action = action*2 - 1
                    new_traj.add(
                        Transition(ObsWrapper(new_obs), action, reward),
                        is_first=(isFirst),
                    )
                    isFirst = False
            new_traj_list += new_traj.to_traj_list()
        out_buffer = ReplayBuffer.load_traj_list(new_traj_list)
    
    traj_list = out_buffer.to_traj_list()
    num_traj = len(traj_list[0])
    max_min_norm = True


    ob_dict={'maximum':list(max_ob), 'minimum':list(min_ob)}
    ac_dict={'maximum':list(max_ac), 'minimum':list(min_ac)}
    print(traj_list[0][-1][0]['state'])
    for i in range(num_traj):
        assert np.max(traj_list[0][i][0]['state']) <= 1        
        assert np.min(traj_list[0][i][0]['state']) >= -1 
        assert np.min(traj_list[0][i][1]) >= -1
        assert np.max(traj_list[0][i][1]) <= 1


    for k in ac_dict:
        ac_dict[k] = list(ac_dict[k])
        ob_dict[k] = list(ob_dict[k])
    ac_norm_path = Path(output_path.parent, 'ac_norm.json')
    ob_norm_path = Path(output_path.parent, 'ob_norm.json')
    with open(str(ac_norm_path), 'w') as f:
        json.dump(ac_dict, f)
    with open(str(ob_norm_path), 'w') as f:
        json.dump(ob_dict, f)


    print(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(traj_list, f)
    print(
        f"Saved {len(out_buffer._traj_starts)} trajectories with {len(out_buffer)} transitions total to {output_path}"
    )

def sim_image_postprocess(encoded_image):
    """
    Flip image 180 degrees
    """
    encoded_image_np = np.frombuffer(encoded_image, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
    rgb_image = bgr_image[:, :, ::-1]
    return cv2.rotate(rgb_image, cv2.ROTATE_180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/share/irl_squared_data",
        help="Root directory for separate trajectories",
    )
    parser.add_argument("--skill_name", type=str, default="lift", help="Skill name")
    parser.add_argument(
        "--is_sim",
        action="store_true",
        default=False,
        help="Whether or not data is from simulation",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        nargs="+",
        default=["optimal"],
        help="Subdirectories to convert",
        #choices=["optimal", "truncate", "recovery"],
    )
    parser.add_argument(
        "--remove_imgs",
        action="store_true",
        default=False,
        help="Whether or not to include images in the buffer",
    )
    parser.add_argument("--num_trajs", default=-1, type=int, help="Number of trajs to convert")
    args = parser.parse_args()

    if args.is_sim:
        demo_type = "sim_data"
        image_postprocess = sim_image_postprocess
    else:
        demo_type = "real_data"
        image_postprocess = None

    # traj_dir = Path(args.root_dir, demo_type, args.skill_name)
    # print(traj_dir)
    traj_dir = Path("/share/portal/human_robot/mustard_tartar_raw")
    pkl_files = []
    for subdir in args.subdir:
        if args.num_trajs != -1 and len(pkl_files) >= args.num_trajs:
            break
        for pkl_file in Path(traj_dir, subdir).glob("*.pkl"):
            pkl_files.append(pkl_file)
            if args.num_trajs != -1 and len(pkl_files) >= args.num_trajs:
                break
    print(f"Converting {len(pkl_files)} trajectories")
    output_path = Path(
        args.root_dir,
        f"{demo_type}_buffers",
        f"{traj_dir.name}-{len(pkl_files)}trajs",
        f"{args.skill_name}-{'-'.join(args.subdir)}.pkl",
    )
    print(args.remove_imgs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_trajectories(pkl_files, output_path, image_postprocess=image_postprocess,remove_images=args.remove_imgs)
