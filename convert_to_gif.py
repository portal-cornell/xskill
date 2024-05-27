import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

# Define the folder path
folder = '/share/portal/pd337/xskill/7'

# Sort the file names to ensure proper ordering
file_names = sorted(os.listdir(folder), key=lambda x: int(re.search(r'\d+', x).group()))[116:169]
# Create a figure
fig = plt.figure()

# Function to update the figure with each frame
def update(frame):
    plt.clf()  # Clear the previous frame
    img = plt.imread(os.path.join(folder, file_names[frame]))  # Read the image
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Turn off axes
    # plt.title('Frame {}'.format(frame))  # Set title to frame number

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(file_names), interval=100)

# Save the animation as a GIF
ani.save('animation.gif', writer='pillow', fps=15)  # Use PillowWriter for GIF format

plt.show()