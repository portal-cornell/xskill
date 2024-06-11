
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define custom colormap
colors = [(0.2, 0, 0.4), (1, 0.7, 0)]  # Dark purple to brighter orange
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
typea = 'bottom'

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
    plt.figure(figsize=(8, 6))
    plt.imshow(array[i][sampled_indices], cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=0.05, aspect=1)
    plt.colorbar().remove()  # Remove the colorbar
    
    # Remove axis labels and ticks
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    # Add thicker black border
    plt.gca().spines['top'].set_linewidth(10)
    plt.gca().spines['bottom'].set_linewidth(10)
    plt.gca().spines['left'].set_linewidth(10)
    plt.gca().spines['right'].set_linewidth(10)
    
    # Calculate and display Transport Plan Cost
    transport_plan_cost = np.sum(array[i])
    if (i == 0 and typea=='top') or (i==2 and typea == 'bottom'):
        bbox_props = dict(boxstyle="round,pad=0.3", fc="lime", ec="black", lw=1, alpha=0.9)
    else:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9)
    plt.text(1, -8, f'Distance: {transport_plan_cost:.2f}', color='black', fontsize=20, fontweight='bold', bbox=bbox_props, verticalalignment='top')
    
    plt.savefig(f'{typea}_heatmap_{i+1}.png', bbox_inches='tight', dpi=1200)
    plt.close()




