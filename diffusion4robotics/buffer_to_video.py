import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from robobuf.buffers import ReplayBuffer


def decode_image(encoded_image):
    # Decode the encoded image byte string
    img_array = np.frombuffer(encoded_image, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def generate_video(trajs, video_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 50)
    fontScale = 0.5
    color = (0, 0, 225)  # Red color
    thickness = 1

    first_traj = trajs[0]
    first_obs, _, _ = first_traj[0]
    img = decode_image(first_obs["enc_cam_0"])
    image_width, image_height, _ = img.shape
    image_width *= len([k for k in first_obs.keys() if "cam" in k])

    video_save_path = Path(f"visualizations/buffers/{video_name}.mp4")
    video_save_path.parent.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(video_save_path),
        fourcc,
        20.0,
        (image_width, image_height),
    )

    for traj_idx, traj in tqdm(enumerate(trajs), total=len(trajs)):
        for i, (obs, _, reward) in enumerate(traj):
            if i % 10 == 0:
                images = []
                for key in obs.keys():
                    if "cam" in key:
                        im = decode_image(obs[key])
                        images.append(im)
                        # rotated_im = cv2.rotate(im, cv2.ROTATE_180)
                        # images.append(rotated_im)

                img = np.hstack(images)  # stack the images horizontally
                img_label = f"Traj: {traj_idx}. Reward: {reward:.5f}"
                img = cv2.putText(
                    img,
                    img_label,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                out.write(img)
    out.release()
    print(f"Video saved to {video_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/share/portal/irl_squared_data",
        help="Root directory for separate trajectories",
    )
    parser.add_argument("--skill_name", type=str, default="lift", help="Skill name")
    parser.add_argument(
        "--is_sim",
        action="store_true",
        default=True,
        help="Whether or not data is from simulation",
    )
    parser.add_argument(
        "--name", type=str, default="lift-optimal", help="Name of buffer file"
    )
    args = parser.parse_args()

    demo_type = "sim_data" if args.is_sim else "real_data"
    with open(f"{args.root_dir}/{demo_type}/buffers/{args.name}.pkl", "rb") as f:
        trajs = ReplayBuffer.load_traj_list(pickle.load(f)).to_traj_list()
    generate_video(trajs, video_name=args.name)
