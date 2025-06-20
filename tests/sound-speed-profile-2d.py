import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Define the horizontal and vertical extent
x = np.linspace(0, 2000, 2000) 
z = np.linspace(-2000, 0, 2000)

# Create 2D grid
X, Z = np.meshgrid(x, z)

# Simple first order approximation based on Mackenzie (1981)
def sound_speed_profile(x, z, T=10, S=35):
    z_pos = -z # Convert negative depth to positive depth to apply to formula
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
    return c

# Compute sound speed profile
C = sound_speed_profile(X, Z)

# Plot the sound speed field as a background
plt.figure(figsize=(10, 6)) # Plot figure
plt.contourf(x, z, C, levels=30, cmap='viridis') # Plot gradient
plt.colorbar(label='Sound Speed (m/s)') # Colorbar for sound speed gradient

# Overlay sea surface and seafloor
surface_z = np.zeros_like(x) # Generate flat sea surface
seafloor_z = -1000 - 0.02 * (x - x[0]) # Generate sloped sea floor
plt.plot(x, surface_z, color='blue', linewidth=3, label='Sea Surface') # Plot sea surface
plt.plot(x, seafloor_z, color='saddlebrown', linewidth=3, label='Seafloor') # Plot sea floor

# Platform
platform_x = 1000 # Platform x position
platform_z = 0 # Platform at surface
plt.scatter(platform_x, platform_z, color='black', s=80, zorder=5, label='Platform') # Plot platform position

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First subplot: 2D sound speed gradient
contour = ax1.contourf(x, z, C, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Sound Speed (m/s)')
ax1.plot(x, np.zeros_like(x), color='blue', linewidth=3, label='Sea Surface')
ax1.plot(x, -900 - 0.02 * (x - x[0]), color='saddlebrown', linewidth=3, label='Seafloor')
ax1.scatter(1000, 0, color='black', s=80, zorder=5, label='Platform')
ax1.set_xlabel('Horizontal Distance (m)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('Sound Speed Profile Across Ocean Cross-section')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Second subplot: Sound speed vs. depth at center
center_idx = len(x) // 2
ax2.plot(C[:, center_idx], z, color='red')
ax2.set_xlabel('Sound Speed (m/s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('Sound Speed Profile at x = {:.1f} m'.format(x[center_idx]))
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()