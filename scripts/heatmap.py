import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

def duplicate_array(arr):
    # Duplicate along the rows and columns
    arr_dup = np.repeat(arr, 2, axis=0)  # Duplicate rows
    arr_dup = np.repeat(arr_dup, 2, axis=1)  # Duplicate columns
    return arr_dup


# Define custom colormap
colors = [(0.2, 0, 0.4), (0.9, 0.6, 0)]  # Dark purple to brighter orange
colors_new = [(12/255,7/255,134/255), (245/255,227/255,37/255)]
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors_new)
typea = 'top'

# Load the numpy array
array = np.load(f'{typea}_row.npy')
array = array[:,20:]

# Set the seed for reproducibility
np.random.seed(42)

# Randomly downsample so all plots are same shape
sampled_indices = np.sort(np.random.choice(len(array[0]), size=77, replace=False))

# Assuming the array has shape (3, 130, 60)
# Create heatmaps for each slice along the first dimension
for i in range(array.shape[0]):
    # interpolated_array = zoom(array[i][sampled_indices], zoom=2)  # Increase zoom factor as needed
    array_duplicated = duplicate_array(array[i][sampled_indices])

    plt.figure(figsize=(16, 12))
    # plt.imshow(interpolated_array, cmap=custom_cmap, interpolation='bicubic', vmin=0, vmax=0.05, aspect=1)
    # plt.imshow(array[i][sampled_indices], cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=0.05, aspect=1)
    plt.imshow(array_duplicated, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=0.1, aspect=1)
    plt.colorbar().remove()  # Remove the colorbar
    
    # Remove axis labels and ticks
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    # Add thicker black border
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0)
    plt.gca().spines['left'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    
    # Calculate and display Transport Plan Cost
    transport_plan_cost = np.sum(array[i])
    if (i == 0 and typea=='top') or (i==2 and typea == 'bottom'):
        bbox_props = dict(boxstyle="round,pad=0.3", fc="lime", ec="black", lw=1, alpha=0.9)
    else:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9)
    # plt.text(1, -8, f'Distance: {transport_plan_cost:.2f}', color='black', fontsize=20, fontweight='bold', bbox=bbox_props, verticalalignment='top')
    
    plt.savefig(f'experiment/figures/{typea}_heatmap_{i+1}.png', bbox_inches='tight', dpi=1200)
    plt.close()