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
