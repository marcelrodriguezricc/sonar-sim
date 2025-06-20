import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# Path to bathymetry XYZ file
xyz_path = "/Users/marcel/Desktop/sonar-sim/assets/bathymetry/CENWP_DIS_YB_00_YBA_20220915_CS/YB_00_YBA_20220915_CS_A.XYZ" # Set filepath

# Load data
data = np.loadtxt(xyz_path) # Load dataset
x = data[:, 0] # Extract first column (x-values) into 1D array
y = data[:, 1] # Extract second column (y-values) into 1D array
z = data[:, 2] # Extract third column (z-values) into 1D array

# Define grid 
xi = np.linspace(x.min(), x.max(), 200) # Linear interpolation between minimum and maximum x values to set grid x points.
yi = np.linspace(y.min(), y.max(), 200) # Linear interpolation between minimum and maximum y values to set grid y points.
z = z * -1 # Invert z 
xi, yi = np.meshgrid(xi, yi) # Merge to 2D array

# Interpolate z values onto grid
zi = griddata((x, y), z, (xi, yi), method='linear') # Estimates z value at each grid point (xi, yi) by interpolating from the known (x, y, z) data.
zi = gaussian_filter(zi, sigma=2) # Apply gausian filter
zi_masked = ma.masked_invalid(zi)

# 2D Plot
plt.figure(figsize=(10, 8))
plt.imshow(zi_masked, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Depth (Z)')
plt.title('Bathymetry Height Map')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 3D Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi_masked, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Depth (Z)')
ax.set_title('Bathymetry 3D Surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (Z)')
plt.show()