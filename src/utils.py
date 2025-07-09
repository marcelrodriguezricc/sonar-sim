import numpy as np

# Get grid based on predefined fetch and depth
def get_grid(f, d):
    # Create an array of evenly spaced values for each dimensional component of grid
    x = np.linspace(0, f, f) # X-axis based on fetch
    y = np.linspace(f, 0, f) # Y-axis based on fetch
    z = np.linspace(d, 0, d) # Z-axis based on depth
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # Produce coordinate matrix (grid)
    return (X, Y, Z)

# Return platform position on grid based on predefined normalized position values
def platform_pos(p, grid):
    x, y, z = grid # Unpack grid
    # Calculate indexes as percentage of total voxels and convert to integer
    x_idx = int(p[0] * (len(x) - 1)) # X-index
    y_idx = int(p[1] * (len(y) - 1)) # Y-index
    z_idx = int(p[2] * (len(z) - 1)) # Z-index
    return np.array([x_idx, y_idx, z_idx]) # Return as array of indicies for platform position on grid

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

# Simple first order approximation of coherent pressure field based on Mackenzie (1981)
def sound_speed_field(grid):
    x, y, z = grid
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
    return c # Return sound speed values for each voxel in "grid"

# Trilinear interpolation to get pressure value based on fractional distance of ray within voxel and value at each corner point of voxel
def trilinear_interp(C, pos):
    x, y, z = pos # Break position on 3D grid into constituent parts
    x0, y0, z0 = np.floor([x, y, z]).astype(int) # Find lower corner of voxel.
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1 # Calculate upper corner of voxel.

    # Prevent attempts to access elements outside valid range of the array when point is near edge
    x0 = np.clip(x0, 0, C.shape[0] - 1)
    x1 = np.clip(x1, 0, C.shape[0] - 1)
    y0 = np.clip(y0, 0, C.shape[1] - 1)
    y1 = np.clip(y1, 0, C.shape[1] - 1)
    z0 = np.clip(z0, 0, C.shape[2] - 1)
    z1 = np.clip(z1, 0, C.shape[2] - 1)

    # Fractional distance from ray position to voxel bottom corner
    xd = x - x0 
    yd = y - y0
    zd = z - z0

    # Get pressure value at each corner of voxel cube
    c000 = C[x0, y0, z0]
    c001 = C[x0, y0, z1] 
    c010 = C[x0, y1, z0]
    c011 = C[x0, y1, z1]
    c100 = C[x1, y0, z0]
    c101 = C[x1, y0, z1]
    c110 = C[x1, y1, z0]
    c111 = C[x1, y1, z1]

    # Use weighted average to interpolate between pressure values at x-corners
    c00 = c000 * (1 - xd) + c100 * xd # Multiple each pressure value by it's weight based on fractional distance within voxel and add weighted values.
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate x-pairings along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate xy-pairings along z-axis
    c = c0 * (1 - zd) + c1 * zd

    return c # Return interpolated sound speed value

# Traverse grid, getting pressure value for each step 
def traverse_grid(c, init_pos, ray_dir, step_size, max_steps):
    shape = c.shape # Get shape of sound speed field (grid)
    num_steps = int(max_steps / step_size) # Calculate maximum number of steps based on size
    idx = np.array(init_pos, dtype=float) # Convert initial position (x, y, z) list into array of floats
    values = [] # Initialize array where pressure values will be stored
    path = [] # Initialize array where ray path points will be stored for graphing

    # Loop until ray is at maximum number of steps (to avoid infinite loops)
    for _ in range(num_steps):
        xi, yi, zi = idx # Break idx array into constituent x, y, and z parts
        if not (0 <= xi < shape[0] and 0 <= yi < shape[1] and 0 <= zi < shape[2]): # If ray steps out of bounds...
            break # Break loop
        values.append(trilinear_interp(c, idx)) # Apply trilinear interpolation to get value based on fractional distance of ray within voxel
        path.append(idx.copy()) # Append current ray position to path array
        idx += ray_dir * step_size  # Move by a fraction of a voxel
    return values, np.array(path)