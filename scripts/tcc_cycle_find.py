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
from xskill.utility.eval_utils import compute_optimal_transport_loss, tcc_loss_

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

robot_pairs = [["kettle", "bottom burner"], ["slide cabinet", "hinge cabinet"]]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 84, 87, 89, 91, 93, 95, 97, 99, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 808, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831, 833, 835, 838, 840, 842, 844, 846, 849, 851, 853, 855, 857, 859, 861, 863, 865, 867, 869, 871, 873, 875, 877, 879, 881, 883, 885, 887, 889, 891, 893, 896, 898, 900, 902, 905, 907, 910, 912, 914, 917, 919, 922]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 48, 50, 102, 104, 106, 108, 110, 112, 115, 118, 120, 122, 124, 126, 128, 130, 133, 135, 137, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 433, 435, 437, 439, 441, 443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 697, 699, 701, 703, 705, 707, 709, 711, 713, 715, 717, 719, 721, 723, 725, 727, 729, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752, 754, 1028, 1030, 1032, 1034, 1036, 1039, 1041, 1043, 1045, 1047, 1049, 1051, 1053, 1055, 1057, 1059, 1061, 1063, 1065, 1067]


human_pairs = [["kettle", "light switch"], ["top burner", "light switch"], ["top burner", "slide cabinet"]]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 62, 65, 67, 69, 71, 73, 75, 77, 79, 81, 84, 87, 89, 91, 93, 95, 97, 99, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 808, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831, 833, 835, 838, 840, 842, 844, 846, 849, 851, 853, 855, 857, 859, 861, 863, 865, 867, 869, 871, 873, 875, 877, 879, 881, 883, 885, 887, 889, 891, 893, 896, 898, 900, 902, 905, 907, 910, 912, 914, 917, 919, 922]
# [375, 377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415, 417, 419, 423, 425, 427, 429, 431]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 48, 50, 102, 104, 106, 108, 110, 112, 115, 118, 120, 122, 124, 126, 128, 130, 133, 135, 137, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 433, 435, 437, 439, 441, 443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 697, 699, 701, 703, 705, 707, 709, 711, 713, 715, 717, 719, 721, 723, 725, 727, 729, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752, 754, 1028, 1030, 1032, 1034, 1036, 1039, 1041, 1043, 1045, 1047, 1049, 1051, 1053, 1055, 1057, 1059, 1061, 1063, 1065, 1067]



human_pairs = [["kettle", "light switch"], ["kettle", "bottom burner"], ["top burner", "slide cabinet"]]
test1 = [101, 103, 105, 107, 109, 111, 114, 117, 119, 121, 123, 125, 127, 129, 132, 134, 136]
test2 = [550, 552, 554, 556, 558, 560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 856, 890, 892, 894, 897, 899, 901, 903, 906, 908, 911, 913, 915, 918, 920, 923, 1069, 1071, 1073, 1075, 1077, 1079, 1081, 1083, 1085, 1087, 1089, 1091, 1093, 1095, 1097, 1099, 1101, 1105, 1108, 1110]
test3 = [139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675, 677, 679, 681, 683, 685, 687, 689, 691, 693, 695]

# pairs = [["kettle", "light switch"], ["kettle", "bottom burner"], ["top burner", "light switch"], ["slide cabinet", "hinge cabinet"]]
# eps = [[] for _ in range(4)]

# pairs = robot_pairs
# eps = [[] for _ in range(2)]

## need to find a case where at some point in the robot video it's in the first half but the cycle back is far away

pairs = human_pairs
eps = [[] for _ in range(3)]

for j, pair in enumerate(pairs):
    for i, tasks in enumerate(task_completions):
        if pair[0] in tasks and pair[1] in tasks:
            eps[j].append(i)
            
# breakpoint()
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

# test = [109, 853, 911, 377]
assignments_for_plot = []
test_a = [109, 911, 667]
test_b = [121, 574, 691]
test = test_a

test = test1

# robot_ep_num = 665
robot_ep_num = 101
with open(os.path.join(robot_proto_path, str(robot_ep_num), 'traj_representation.json'), "r") as f:
    robot_proto_data = json.load(f)
robot_proto_data = torch.Tensor(np.array(robot_proto_data, dtype=np.float32))  # (T,D)
for human_ep_num in test:
    with open(os.path.join(human_proto_path, str(human_ep_num), 'traj_representation.json'), "r") as f:
        human_proto_data = json.load(f)
    human_proto_data = torch.Tensor(np.array(human_proto_data, dtype=np.float32))  # (T,D)
    tcc_l, dists, idxs, start_robot, mid_human, end_robot = tcc_loss_(robot_proto_data.unsqueeze(0), human_proto_data.unsqueeze(0))   
    print(f'tcc: {start_robot}, {end_robot}')
    breakpoint()
     

print(sorted(total_dists))
idxs = torch.Tensor(total_dists).argsort()
print(torch.Tensor(test)[idxs])
# print(total_assignments)
# with open('dists.json', "w") as f:
#     json.dump(total_dists, f)
# with open('assignments.json', "w") as f:
#     json.dump(total_assignments, f)
breakpoint()
# import numpy as np

# tensor_list = assignments_for_plot

# # Convert tensors to numpy arrays
# numpy_array_list = [tensor.numpy() for tensor in tensor_list]

# # Stack numpy arrays
# stacked_array = np.stack(numpy_array_list)

# # Save numpy array
# np.save('top_row.npy', stacked_array)



