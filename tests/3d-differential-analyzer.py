import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

# Variables
platform = np.array([0.5, 0.5, 0.0]) # Platform position as percentage of grid
depth = 3500 # Depth to seafloor
fetch = 100 # Length of ocean fetch (m/sqr)
max_steps = 10000

# Generate grid
grid = utils.get_grid(fetch, depth)

# Get platform position on grid
init_pos = utils.platform_pos(platform, grid)

# Get sound speed field
C = utils.sound_speed_field(grid)

# Initialize ray direction vector (straight down for test purposes)
ray_dir = np.array([0.0, 0.0, 1.0])

# At each step...
# Get current location
# Compute sound speed at current voxel
# Apply vectoized Snell's Law
# Update position and angle

# 3D Digital Differential Analyzer
# At each step, the algorithm checks which of the three axes the ray will cross next (which voxel boundary is closest along the ray's path), then increments the index along that axis by one.
def traverse_grid(C, init_pos, ray_dir, max_steps):
    shape = C.shape # Get shape of field
    idx = np.array(init_pos, dtype=float) # Initialize ray index from starting position
    values = [] # Create an empty array to return values of C for ray traversal
    path = [] # Initialize an empty array for ray path positions

    # Calculate step and tMax/tDelta for each axis
    step = np.sign(ray_dir).astype(int) # Increment, decrement, or do nothing along each axis as the ray moves
    tMax = np.zeros(3) # Initialize array that holds distance to next voxel boundary (variable)
    tDelta = np.zeros(3) # Initialize array that will hold distance between subsequent voxel boundaries (constant)
    grid_spacing = [1, 1, 1]  # Grid is uniform and each voxel is 1 unit in size

    # For each axis...
    for i in range(3):
        if ray_dir[i] != 0: # If moving along axis...
            if step[i] > 0: # And if stepping in positive direction...
                next_voxel_boundary = np.floor(idx[i] + 1) # Calculate next voxel boundary.
            else: # Else if stepping in negative direction...
                next_voxel_boundary = np.ceil(idx[i] - 1) # Calculate next voxel boundary
            tMax[i] = (next_voxel_boundary - idx[i]) / ray_dir[i]
            # Distance ray must travel to reach boundary. 
            tDelta[i] = grid_spacing[i] / abs(ray_dir[i])
            # Distance ray must travel between subsequent boundaries.
        else: # If the ray is not moving along axis...
            # Set tMax and tDelta to infinity so ray will not step this along this axis
            tMax[i] = np.inf
            tDelta[i] = np.inf

    # Loop up to max_steps limit to prevent infinite loops...
    for _ in range(max_steps):
        xi, yi, zi = idx.astype(int) # Break ray position index into x, y, and z index components.
        if not (0 <= xi < shape[0] and 0 <= yi < shape[1] and 0 <= zi < shape[2]): # If index exceeds boundaries of grid...
            break # Break loop
        values.append(C[xi, yi, zi]) # Get value at index.
        path.append([xi, yi, zi]) # Get index position of ray.
        # Step to next voxel
        axis = np.argmin(tMax) # Find axis along which next boundary will be crossed.
        idx[axis] += step[axis] # Move ray along chosen axis.
        tMax[axis] += tDelta[axis] # Update tMax for chosen axis.

    return values, np.array(path)

# Cast ray along grid, and get values and position index at each step
values_along_ray, ray_path = traverse_grid(C, init_pos, ray_dir, max_steps)

# Normalize C values for colormap
C_ray = np.array(values_along_ray) # Convert list to numpy array
norm = (C_ray - C_ray.min()) / (C_ray.max() - C_ray.min()) # Normalize C values
colors = cm.viridis(norm) # Apply viridis gradient

# Generate figure
fig = plt.figure(figsize=(8, 6)) # Create matplotlib figure object
ax = fig.add_subplot(111, projection='3d') # 3D graph

# Plot as colored line segments
segments = np.array([ray_path[:-1], ray_path[1:]]).transpose(1, 0, 2) # Create line segments from consecutive points along ray path
lc = Line3DCollection(segments, colors=colors[:-1], linewidth=2) # Create 3D Line Collection where each segment is colored according to corresponding C value
ax.add_collection3d(lc) # Add to plot

# Plot the starting point
ax.scatter([init_pos[0]], [init_pos[1]], [init_pos[2]], color='blue', s=50, label='Platform')

# Label axes, set title, and add legend
ax.set_xlabel('X Fetch')
ax.set_ylabel('Y Fetch')
ax.set_zlabel('Z Depth')
ax.set_title('3D Differential Analyzer')
ax.legend()

# Set limits to predefined section of ocean
ax.set_xlim(0, fetch)
ax.set_ylim(0, fetch)
ax.set_zlim(0, depth)

# Invert z-axis to reflect positive depth downwards
ax.invert_zaxis()

# Add colorbar
import matplotlib as mpl
mappable = cm.ScalarMappable(cmap='viridis')
mappable.set_array(C_ray)
fig.colorbar(mappable, ax=ax, label='Sound Speed (C)')

plt.savefig('/Users/marcel/Desktop/sonar-sim/docs/figures/3d-differential-analyzer.png', dpi=300)
plt.show()