import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D

# Parameters
heading = 0 # Heading of source
num_rays = 100 # Number of rays to emit
ray_length = 1 # Length of each ray
num_points = 1000 # Number of points in ray trajectory
arrow_length_pct = 0.1  # Percentage of distance to edge of height map for local axes

# Path to bathymetry XYZ file
xyz_path = "/Users/marcel/Desktop/sonar-sim/assets/bathymetry/CENWP_DIS_YB_00_YBA_20220915_CS/YB_00_YBA_20220915_CS_A.XYZ" # Set filepath

# Load data
data = np.loadtxt(xyz_path) # Load dataset
x = data[:, 0] # Extract first column (x-values) into 1D array
y = data[:, 1] # Extract second column (y-values) into 1D array
z = data[:, 2] # Extract third column (z-values) into 1D array

# Normalize x, y, z to [0, 1]
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())
z_norm = (z - z.min()) / (z.max() - z.min())
z_norm = z_norm * -1

# Define grid in normalized coordinates
xi = np.linspace(0, 1, 200)
yi = np.linspace(0, 1, 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z values onto grid
zi = griddata((x_norm, y_norm), z_norm, (xi, yi), method='linear')
zi = gaussian_filter(zi, sigma=2)
zi_masked = ma.masked_invalid(zi)

# Set source at (center_x, center_y, 0)
source = np.array([.5, .5, 0])

# Specify heading
heading_angle = np.deg2rad(heading)  # Convert degrees to radians
heading_xy = np.array([np.cos(heading_angle) + .5, np.sin(heading_angle) + .5, 0]) # Create 3D vector based on heading angle.

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
    length=.1, color='r', linewidth=.5, arrow_length_ratio=0.2, label='Heading'
)

# Lateral
ax.quiver(
    source[0], source[1], source[2],
    lateral[0], lateral[1], lateral[2],
    length=.1, color='g', linewidth=.5, arrow_length_ratio=0.2, label='Lateral'
)

# Down
ax.quiver(
    source[0], source[1], source[2],
    down[0], down[1], down[2],
    length=.1, color='b', linewidth=.5, arrow_length_ratio=0.2, label='Down'
)

# Create an interpolator for the bathymetry surface
bathy_interp = RegularGridInterpolator(
    (yi[:,0], xi[0,:]),  # grid axes (y, x)
    zi_masked,           # grid values
    bounds_error=False,
    fill_value=np.nan
)

intersection_points = []  # List to store intersection points

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
    t = np.linspace(0, 1, num_points) #  Generate array of specified number of points between 0 and 1.
    trajectory = np.outer(1 - t, source) + np.outer(t, endpoint) # Calculate trajectory points as percentage of source to target.
       # March along the ray and check for intersection
    for i, point in enumerate(trajectory):
        x_ray, y_ray, z_ray = point
        bathy_z = bathy_interp((y_ray, x_ray))  # Note: order is (y, x)
        if np.isnan(bathy_z):
            continue  # skip if outside bathy grid
        if z_ray <= bathy_z:
            # Plot the ray up to this point only
            ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], trajectory[:i+1, 2], color='y', alpha=0.7)
            intersection_points.append([x_ray, y_ray, z_ray])  # Store intersection
            break
    else:
        # If no intersection, plot the full ray
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='y', alpha=0.7)

# Plot
ax.scatter(*source, color='black', s=5, label='Source') # Plot the source
ax.set_xlabel('X') # Label x-axis
ax.set_ylabel('Y') # Label y-axis
ax.set_zlabel('Z') # Label z-axis
ax.set_title(f'3D Ray Fan: {num_rays} Rays, 180° Spread') # Add title
ax.legend() # Add legend

intersection_points = np.array(intersection_points)
if len(intersection_points) > 0:
    ax.scatter(
        intersection_points[:, 0],
        intersection_points[:, 1],
        intersection_points[:, 2],
        color='r', s=10, label='Intersections'
    )

surf = ax.plot_surface(
    xi, yi, zi_masked,
    cmap='viridis',
    alpha=0.7,         # Adjust transparency so rays and source are visible
    linewidth=0,
    antialiased=True
)

# Set axis limits (change these values as needed)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-1, 0)

plt.show() # Show in window.