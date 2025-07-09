import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

# User Settings
platform = np.array([0.5, 0.5, 0.0]) # Platform position as percentage of grid
depth = 250 # Depth to seafloor
fetch = 250 # Length of ocean fetch (m/sqr)
ray_dir = np.array([0.0, 0.5, 0.5]) # Ray direction vector
step_size = 0.1  # Ray step
max_steps = 10000 # Max amount of steps until ray breaks (to avoid infinite loops)

# Generate grid
grid = utils.get_grid(fetch, depth)

# Get platform position on grid
init_pos = utils.platform_pos(platform, grid)

# Get sound speed field
C = utils.sound_speed_field(grid)

# Get pressure value and position for each step of ray
values, path = utils.traverse_grid(C, init_pos, ray_dir, step_size, max_steps)

# Normalize C values for colormap
C_ray = np.array(values) # Convert list to numpy array
norm = (C_ray - C_ray.min()) / (C_ray.max() - C_ray.min()) # Normalize C values
colors = cm.viridis(norm) # Apply viridis gradient

# Generate figure
fig = plt.figure(figsize=(8, 6)) # Create matplotlib figure object
ax = fig.add_subplot(111, projection='3d') # 3D graph

# Plot as colored line segments
segments = np.array([path[:-1], path[1:]]).transpose(1, 0, 2) # Create line segments from consecutive points along ray path
lc = Line3DCollection(segments, colors=colors[:-1], linewidth=2) # Create 3D Line Collection where each segment is colored according to corresponding C value
ax.add_collection3d(lc) # Add to plot

# Plot the starting point
ax.scatter([init_pos[0]], [init_pos[1]], [init_pos[2]], color='blue', s=50, label='Platform')

# Label axes, set title, and add legend
ax.set_xlabel('X Fetch')
ax.set_ylabel('Y Fetch')
ax.set_zlabel('Z Depth')
ax.set_title('Ray Traversal')
ax.legend()

# Set limits to predefined section of ocean
ax.set_xlim(0, fetch)
ax.set_ylim(0, fetch)
ax.set_zlim(0, depth)

# Invert z-axis to reflect positive depth downwards
ax.invert_zaxis()

# Add colorbar
mappable = cm.ScalarMappable(cmap='viridis')
mappable.set_array(C_ray)
fig.colorbar(mappable, ax=ax, label='Sound Speed (C)')

plt.savefig('/Users/marcel/Desktop/sonar-sim/docs/figures/3d-differential-analyzer.png', dpi=300)
plt.show()