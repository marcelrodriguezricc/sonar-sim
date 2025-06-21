import numpy as np
import matplotlib.pyplot as plt

# Variables
platform = 0.5 # Platform position on horizon as percentage
depth = 4000 # Depth to seafloor
gradient_extent = 4500 # Draw extent of sound speed gradient
horiz = 2000 # Length of horizon

# Generate arrays for horizontal and vertical extents
x = np.linspace(0, horiz, horiz) # X array
z = np.linspace(gradient_extent, 0, gradient_extent) # Z array
X, Z = np.meshgrid(x, z) # Mesh into 2D array of [x, z].

# Get x position of platform
def platform_position(p):
    x_new = horiz * p # Multiply percentage by total extent to get platform position on x
    return int(x_new) # Return platform position on x as n integer

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
    return T # Return temperature

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
    return S

# Simple first order approximation based on Mackenzie (1981)
def sound_speed_profile(x, z):
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
    return c

# Compute sound speed profile
C = sound_speed_profile(X, Z)

# Overlay sea surface and seafloor
surface_z = np.zeros_like(x) # Generate flat sea surface
seafloor_z = depth + 0.02 * (x - x[0]) # Generate sloped sea floor

# Platform
platform_x = platform_position(platform) # Platform x position
platform_z = 0 # Platform at surface

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [4, 1]}
)

# First subplot: 2D sound speed gradient
contour = ax1.contourf(x, z, C, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Sound Speed (m/s)')
ax1.plot(x, surface_z, color='blue', linewidth=3, label='Sea Surface')
ax1.plot(x, seafloor_z, color='saddlebrown', linewidth=3, label='Seafloor')
ax1.fill_between(x, seafloor_z, gradient_extent, color='saddlebrown', alpha=1.0, zorder=2, label='_nolegend_')
ax1.scatter(platform_x, 0, color='black', s=80, zorder=3, label='Platform')
ax1.set_xlabel('Horizontal Distance (m)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('Sound Speed Gradient Cross-section')
ax1.legend()
ax1.invert_yaxis()
ax1.grid(True, linestyle='--', alpha=0.5)

# Second subplot: Sound speed vs. depth at platform
ax2.plot(C[:, platform_x], z, color='red')
ax2.set_xlabel('Sound Speed (m/s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('Sound Speed Profile')
ax2.invert_yaxis()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/Users/marcel/Desktop/sonar-sim/docs/figures/sound_speed_plot_1.png', dpi=300)
plt.show()