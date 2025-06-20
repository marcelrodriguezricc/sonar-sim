import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define source and target points in 3D space
source = np.array([0, 0, 0]) # XYZ coordinates of source.
target = np.array([10, 5, -3]) # XYZ coordinates of target.

# Generate points along the ray
num_points = 100 # Number of points in trajectory line.
t = np.linspace(0, 1, num_points) # Generate array with 100 values from 0 to 1.
trajectory = np.outer(1 - t, source) + np.outer(t, target) # Generate trajectory points as percentage of source to target.

# Plot the trajectory
fig = plt.figure(figsize=(8, 6)) # Generate figure
ax = fig.add_subplot(111, projection='3d') # 3D projection
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Ray Path', color='b') # Plot ray trajectory
ax.scatter(*source, color='g', s=50, label='Source') # Source point
ax.scatter(*target, color='r', s=50, label='Target') # Target point
ax.set_xlabel('X') # Label x-axis
ax.set_ylabel('Y') # Label y-axis
ax.set_zlabel('Z') # Label z-axis
ax.set_title('Basic 3D Raytracing: Source to Target') # Add title to graphic
ax.legend() # Show legend
plt.show() # Show plot in window