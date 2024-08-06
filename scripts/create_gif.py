import os
import imageio
import re

from PIL import Image
import numpy as np

def create_gif_from_images(image_folder, output_gif, new_size):
    # Get all image files in the folder
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png') or file.endswith('.jpg')]

    # Sort image files by the number in their names
    image_files = sorted(image_files, key=lambda name: int(re.findall(r'\d+', name)[0]))

    # Read all images into a list and resize them
    images = [Image.open(image_file).resize(new_size) for image_file in image_files]

    # Convert PIL Image objects back to numpy arrays as imageio expects numpy arrays
    images = [np.array(image) for image in images]

    # Write images into a gif
    imageio.mimsave(output_gif, images)

# create a function to create mp4 from images
def create_mp4_from_images(image_folder, output_mp4, new_size):
    # Get all image files in the folder
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png') or file.endswith('.jpg')]

    # Sort image files by the number in their names
    image_files = sorted(image_files, key=lambda name: int(re.findall(r'\d+', name)[0]))

    # Read all images into a list and resize them
    images = [Image.open(image_file).resize(new_size) for image_file in image_files]

    # Convert PIL Image objects back to numpy arrays as imageio expects numpy arrays
    images = [np.array(image) for image in images]

    # Write images into a gif
    imageio.mimsave(output_mp4, images, fps=15)

# Usage
embodiment = 'singlehand'
idx = 583
filename = f'/share/portal/kk837/xskill/datasets/kitchen_dataset/{embodiment}/{idx}'
create_gif_from_images(filename, f'output_{embodiment}_{idx}.gif', (384, 384))
create_mp4_from_images(filename, f'output_{embodiment}_{idx}.mp4', (384, 384))