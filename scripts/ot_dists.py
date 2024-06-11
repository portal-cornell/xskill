from collections import defaultdict, namedtuple
import numpy as np
import torch
from xskill.utility.file_utils import get_subdirs, get_files
import random
import collections
import os
import os.path as osp
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import cv2
from xskill.utility.eval_utils import gif_of_clip
from omegaconf import DictConfig
from xskill.utility.eval_utils import compute_optimal_transport_loss

# 1) Function that takes as input one robot trajectory and four human trajectories
#  rzs = z1,...,zT ; hzs = z1,...,zK
# 2) With rzs and hzs, compute the optimal transport distance (and assignments)
# 3) Save images of the frames which correspond to the 10 lowest distances (may need to filter to ignore the start and end of the clips)
#    Also save the distance numbers and probability masses of the assignments

with open("/share/portal/pd337/xskill/task_l.json", "r") as f:
    task_completions = json.load(f)
task_completions = task_completions

pretrain_path = "/share/portal/pd337/xskill/experiment/pretrain/no_pairing_twohands_2024-05-28_21-11-55"
robot_proto_path = os.path.join(pretrain_path, 'robot_segments_paired_twohands_encode_protos', 'ckpt_40')
human_proto_path = os.path.join(pretrain_path, 'twohands_segments_paired_twohands_encode_protos', 'ckpt_40')

robot_pairs = [["kettle", "bottom burner"], ["kettle", "light switch"]]
human_pairs = [["kettle", "bottom burner"], ["kettle", "light switch"], ["microwave", "bottom burner"], ["slide cabinet", "hinge cabinet"]]


pairs = [["kettle", "light switch"], ["kettle", "bottom burner"], ["top burner", "light switch"], ["slide cabinet", "hinge cabinet"]]
eps = [[] for _ in range(4)]

for j, pair in enumerate(pairs):
    for i, tasks in enumerate(task_completions):
        if pair[0] in tasks and pair[1] in tasks:
            eps[j].append(i)
            

"""
[101, 103, 105, 107, 109, 111, 114, 117, 119, 121, 123, 125, 127, 129, 132, 134, 136]

[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 84, 87, 89, 91, 93, 95, 97, 99, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 808, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831, 833, 835, 838, 840, 842, 844, 846, 849, 851, 853, 855, 857, 859, 861, 863, 865, 867, 869, 871, 873, 875, 877, 879, 881, 883, 885, 887, 889, 891, 893, 896, 898, 900, 902, 905, 907, 910, 912, 914, 917, 919, 922]

[550, 552, 554, 556, 558, 560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 856, 890, 892, 894, 897, 899, 901, 903, 906, 908, 911, 913, 915, 918, 920, 923, 1069, 1071, 1073, 1075, 1077, 1079, 1081, 1083, 1085, 1087, 1089, 1091, 1093, 1095, 1097, 1099, 1101, 1105, 1108, 1110]

[375, 377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415, 417, 419, 423, 425, 427, 429, 431]
"""
# eps = [106, 0, 52, 383]
# robot_ep_num = eps[0]
# with open(os.path.join(robot_proto_path, str(robot_ep_num), 'traj_representation.json'), "r") as f:
#     robot_proto_data = json.load(f)
# robot_proto_data = torch.Tensor(np.array(robot_proto_data, dtype=np.float32))  # (T,D)

# total_dists, total_assignments = [], []
# for human_ep_num in eps:
#     with open(os.path.join(human_proto_path, str(human_ep_num), 'traj_representation.json'), "r") as f:
#         human_proto_data = json.load(f)
#     human_proto_data = torch.Tensor(np.array(human_proto_data, dtype=np.float32))  # (T,D)
#     dists, assignments = compute_optimal_transport_loss(human_proto_data.unsqueeze(0), robot_proto_data.unsqueeze(0))
#     total_dists.extend(dists)
#     total_assignments.extend(assignments)
# 
# print(total_dists)

# 130, 42, 52, 382
total_dists, total_assignments = [], []

test = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 84, 87, 89, 91, 93, 95, 97, 99, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 808, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831, 833, 835, 838, 840, 842, 844, 846, 849, 851, 853, 855, 857, 859, 861, 863, 865, 867, 869, 871, 873, 875, 877, 879, 881, 883, 885, 887, 889, 891, 893, 896, 898, 900, 902, 905, 907, 910, 912, 914, 917, 919, 922]
# test = [109, 853, 911, 377]
test = eps[3]
robot_ep_num = 101
with open(os.path.join(robot_proto_path, str(robot_ep_num), 'traj_representation.json'), "r") as f:
    robot_proto_data = json.load(f)
robot_proto_data = torch.Tensor(np.array(robot_proto_data, dtype=np.float32))  # (T,D)
for human_ep_num in test:
    with open(os.path.join(human_proto_path, str(human_ep_num), 'traj_representation.json'), "r") as f:
        human_proto_data = json.load(f)
    human_proto_data = torch.Tensor(np.array(human_proto_data, dtype=np.float32))  # (T,D)
    dists, assignments = compute_optimal_transport_loss(human_proto_data.unsqueeze(0), robot_proto_data.unsqueeze(0))
    total_dists.append(dists[0][0].item())
    total_assignments.append(assignments[0].tolist())

print(total_dists)
idxs = torch.Tensor(total_dists).argsort()
print(torch.Tensor(test)[idxs])
# print(total_assignments)
# with open('dists.json', "w") as f:
#     json.dump(total_dists, f)
# with open('assignments.json', "w") as f:
#     json.dump(total_assignments, f)
breakpoint()

# dists: 23 (109), 27 (853), 49 (911), 59 (377)


