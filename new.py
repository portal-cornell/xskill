import os
import cv2

# Define the root directory
root_dir = '/share/portal/pd337/xskill/datasets/portal_table_human'

# Function to resize images in a directory
def resize_images_in_directory(directory, dir_name):
    # Get file names and sort them based on the numeric part
    vid_names = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))
    # Create a new directory to store resized images
    resized_dir = os.path.join(root_dir, 'resized', dir_name)
    os.makedirs(resized_dir, exist_ok=True)
    # Resize each image
    for vid in vid_names:
        vid_path = os.path.join(directory, vid)
        for i, file_name in enumerate(os.listdir(vid_path)):
            img = cv2.imread(os.path.join(vid_path, file_name))
            if img is not None:
                img_resized = cv2.resize(img, (256, 256))
                os.makedirs(os.path.join(resized_dir, vid), exist_ok=True)
                cv2.imwrite(os.path.join(resized_dir, vid, str(i) + '.png'), img_resized)
                print(f"Resized {file_name} to {str(i)}.png")
            else:
                print(f"Failed to read {file_name}")

# Iterate through each directory in the root directory
for _, dir_names, _ in os.walk(root_dir):
    for dir_name in dir_names:
        resize_images_in_directory(os.path.join(root_dir, dir_name), dir_name)