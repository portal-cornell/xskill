import cv2
import os

def images_to_video(image_folder, output_video, fps=10):
    # Get the list of images in the folder, assuming they are named sequentially
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('.')[0]))  # Sort based on the numeric part of the filename

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through the sorted image list and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved as {output_video}")

# Parameters
image_folder = '/share/portal/pd337/xskill/datasets/kitchen_dataset/robot/247'  # Replace with your image folder path
output_video = 'robot_247.mp4'  # Output video file name
fps = 10  # Frames per second

# Call the function
images_to_video(image_folder, output_video, fps)
