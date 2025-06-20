import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
source = np.array([0, 0, 0]) # XYZ position of source
heading = 100 # Heading of source
num_rays = 100 # Number of rays to emit
ray_length = 5 # Length of each ray
num_points = 100 # Number of points in ray trajectory

# Specify heading
heading_angle = np.deg2rad(heading)  # Convert degrees to radians
heading_xy = np.array([np.cos(heading_angle), np.sin(heading_angle), 0]) # Create 3D vector based on heading angle.

# Local axes
down = np.array([0, 0, -1]) # Down from source to floor
lateral = np.cross(heading_xy, down) # Cross product of heading and local down to find lateral vector

# Generate angle for each ray
angles = np.linspace(-np.pi/2, np.pi/2, num_rays)  # Fan 180°

# Create plot ahead of loop
fig = plt.figure(figsize=(8, 6)) # Generate figure
ax = fig.add_subplot(111, projection='3d') # Set 3D projection

# Heading
ax.quiver(
    source[0], source[1], source[2],
    heading_xy[0], heading_xy[1], heading_xy[2],
    length=1, color='g', linewidth=1, arrow_length_ratio=0.2, label='Heading'
)

# Lateral
ax.quiver(
    source[0], source[1], source[2],
    lateral[0], lateral[1], lateral[2],
    length=1, color='b', linewidth=1, arrow_length_ratio=0.2, label='Lateral'
)

# Down
ax.quiver(
    source[0], source[1], source[2],
    down[0], down[1], down[2],
    length=1, color='r', linewidth=1, arrow_length_ratio=0.2, label='Down'
)

# For each angle, compute the direction vector and endpoint
for theta in angles: # For each ray...
    # Calculate direction vector weighted by angle.
    direction = (
        np.sin(theta) * lateral +
        np.cos(theta) * down 
    )
    direction = direction / np.linalg.norm(direction)  # Normalize direction
    endpoint = source + direction * ray_length # Calculate endpoint based on direction and ray length from source

    # Generate ray
    t = np.linspace(0, 1, num_points) #  Generate array of specified number of points
    trajectory = np.outer(1 - t, source) + np.outer(t, endpoint) # Calculate trajectory points as percentage of source to target
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='y', alpha=0.7) # Plot ray trajectory

# Plot
ax.scatter(*source, color='black', s=50, label='Source') # Plot the source
ax.set_xlabel('X') # Label x-axis
ax.set_ylabel('Y') # Label y-axis
ax.set_zlabel('Z') # Label z-axis
ax.set_title(f'3D Ray Fan: {num_rays} Rays, 180° Spread') # Add title
ax.legend() # Add legend

# Set axis limits (change these values as needed)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 2)

plt.show() # Show in window.