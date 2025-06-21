import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Variables
platform = np.array([0.25, 0.5, 0.75]) # Platform positions based on normalized total area
depth = 3500 # Depth to seafloor
fetch = 100 # Length of ocean fetch (m/sqr)

# Generate arrays for horizontal and vertical extents
x = np.linspace(0, fetch, fetch) # X array
y = np.linspace(fetch, 0, fetch) # Y array
z = np.linspace(depth, 0, depth) # Z array
X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # Create 3D coordinate grid

# Get position of platform
def platform_position(p):
    x_new = fetch * p[0] # Calculate platform position on x
    y_new = fetch * p[1] # Calculate platform position on y
    z_new = fetch * p[2] # Calculate platform position on z
    return np.array([x_new, y_new, z_new])

# Basic temperature profile
def temperature_profile(z):
    z = np.asarray(z) # Convert input to array
    T = np.where(z <= 100, # If depth is less than 100m...
        20, # set to 20 degrees (C)
        np.where(z <= 1000, # If depth is less greater than 100 than 1000m...
            20 - 18 * ((z - 100) / 900), # Temperature decreases linearly from 20 to 2
            2 # If depth is greater than 1000m, set to 2 degrees (C)
        )
    )
    return T # Return temperature based on depth

# Basic salinity profile
def salinity_profile(z): 
    z = np.asarray(z) # Convert input to array
    S = np.where(z <= 100, # If depth is less than 100m....
        35, # Set to 35 Practical Salinity Units (PSU)
        np.where(z <= 1000, # If depth is greater than 100m and less than 1000m...
            35 + 0.5 * ((z - 100) / 900), # Salinity increases linearly from 35 to 35.5 PSU
            35.5 # If depth is greater than 1000m, set to 35.5 PSU
        )
    )
    return S # Return salinity based on depth

# Simple first order approximation based on Mackenzie (1981)
def sound_speed_field(x, y, z):
    T = temperature_profile(z) # Calculate temperature based on depth
    S = salinity_profile(z) # Calculate salinity based on depth
    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * z
        + 1.675e-7 * z**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * z**3
    )
    # Add minor variation with x and y
    c += 0.1 * (x / x.max())  # Small linear gradient in x
    c += 0.1 * (y / y.max())  # Small linear gradient in y
    c += 0.05 * np.sin(2 * np.pi * x / x.max()) * np.sin(2 * np.pi * y / y.max())  # Small sinusoidal perturbation
    return c

# Get sound speed field
C = sound_speed_field(X, Y, Z) # Apply sound speed field to 3D grid

# Platform
platform_x = platform_position(platform) # Platform x position
platform_z = 0 # Platform at surface

# Downsample your data for voxels
step = 40
Cv = C[::step, ::step, ::step]

# Compute voxel edges for each axis
xv = np.linspace(x[0], x[-1], Cv.shape[0]+1)
yv = np.linspace(y[0], y[-1], Cv.shape[1]+1)
zv = np.linspace(z[0], z[-1], Cv.shape[2]+1)

# Use np.meshgrid to create the grid of voxel edges
Xv, Yv, Zv = np.meshgrid(xv, yv, zv, indexing='ij')

# Surface and seafloor
N = Cv.shape[0]  # Number of x points after downsampling
M = Cv.shape[1]  # Number of y points after downsampling
x_surf = np.linspace(0, fetch, N)
y_surf = np.linspace(0, fetch, M)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf, indexing='ij')
Z_surf = np.zeros_like(X_surf)
seafloor = depth + 0.02 * (x_surf - x_surf[0])  # shape (N,)
Z_floor = np.tile(seafloor[:, np.newaxis], (1, M))

# Create a filled mask
filled = np.ones_like(Cv, dtype=bool)

# Plot voxels with physical coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis((Cv - Cv.min()) / (Cv.max() - Cv.min()))
ax.voxels(Xv, Yv, Zv, filled, facecolors=colors, edgecolor=None, alpha=0.5, zorder=1)
ax.plot_surface(X_surf, Y_surf, Z_surf, color='deepskyblue', alpha=0.5, linewidth=0, zorder=0, label='Sea Surface')
ax.plot_surface(X_surf, Y_surf, Z_floor, color='saddlebrown', alpha=0.5, linewidth=0, zorder=0, label='Seafloor')
ax.invert_zaxis()
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Depth (m)')
ax.set_title('Sound Speed Field')
plt.show()