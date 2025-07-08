import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Variables
platform = np.array([0.25, 0.5, 0.0]) # Platform position as percentage of grid
depth = 3500 # Depth to seafloor
fetch = 100 # Length of ocean fetch (m/sqr)

# Generate grid
grid = utils.get_grid(fetch, depth)

# Get platform position on grid
init_pos = utils.platform_pos(platform, grid)

# Get sound speed field
C = utils.sound_speed_field(grid)

# Initialize ray direction vector (straight down for test purposes)
ray_dir = np.array([0.0, 0.0, -1.0])

# At each step...
#for i in range(int(depth)):
#    x_idx = np.argmin(np.abs(x - ray_pos[0]))
#    y_idx = np.argmin(np.abs(y - ray_pos[1]))
#    z_idx = np.argmin(np.abs(z - ray_pos[2]))
#    print(x_idx, C[x_idx, y_idx, z_idx])
# Get current location
# Compute sound speed at current voxel
# Apply vectoized Snell's Law
# Update position and angle