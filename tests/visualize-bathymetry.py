import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# Path to bathymetry XYZ file
xyz_path = "/Users/marcel/Desktop/sonar-sim/assets/bathymetry/CENWP_DIS_YB_00_YBA_20220915_CS/YB_00_YBA_20220915_CS_A.XYZ"

# Load data
data = np.loadtxt(xyz_path)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Define grid 500 x 500
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate Z values onto grid
zi = griddata((x, y), z, (xi, yi), method='linear')
zi = gaussian_filter(zi, sigma=2)
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