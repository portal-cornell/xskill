import os
import shutil
import random

# Define paths
original_folder = '/share/portal/pd337/xskill/datasets/kitchen_dataset/human_segments_paired'
new_folder = '/share/portal/pd337/xskill/datasets/kitchen_dataset/human_segments_paired_sample'

# Create new folder if it doesn't exist
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# Get list of subfolders in original folder
subfolders = [subfolder for subfolder in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, subfolder))]

# Shuffle subfolders and select 500
random.shuffle(subfolders)
selected_subfolders = subfolders[:500]

# Copy selected subfolders to new folder with ordered names
for i, subfolder in enumerate(selected_subfolders):
    original_subfolder_path = os.path.join(original_folder, subfolder)
    new_subfolder_path = os.path.join(new_folder, str(i))
    shutil.copytree(original_subfolder_path, new_subfolder_path)

print("Sample of 500 folders copied successfully.")
