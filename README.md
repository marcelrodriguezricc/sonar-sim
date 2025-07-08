# Sonar Sim

A Python based 3D side-scan sonar simulation for development of real-time data processing tools.

## Outline
1. **Generate 3D physical environment**

   -  Generate sound speed field.

      - Calculate basic salinity and temperature profile based on depth.

      - Calculate coherent pressure calculation for each voxel based on first order approximation of Mackenzie (1981), with slight variation in away from center.

   - Assign material properties to bottom.

         - Scattering and absorption based on bottom material properties.

   - Assign geometry to surface.

      - Flat or rough.

2. **Define properties of acoustic source emissions**

   - Positionality of platform and transducers.

   - Frequency.

   - Waveform.

   - Directivity pattern.

   - Pulse width.

3. **Raytracing**

   - Launch a fan of rays.

   - Propagate rays through physical environment.

      - Gaussian beam technique to simulate sonar beam footprint, intensity variation across footprint, interactions with regions of complex bathymetry (high relief), and smoothing over of caustics for closer approximation of sonar behavior as opposed to classical raytracing.

      - Apply attenuation as ray travels through each voxel using a 3D Differential Analyzer (Museth, 2013).

   - Apply boundary interactions.

      - Detect intersection.

      - A apply reflection and scattering based on bottom material and surface turbulance.

      - Volume scattering through thermoclines.

4. **Receiver**

   - Sum pressure field contributions of each ray to reconstruct incident pressure field at transducers.

   - Store eigenray parameters

      - Arrival time.

      - Amplitude.

      - Angle.

5. **Output**

   - Sonar imagery of bottom contours.

   - Other graphical materials.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd sonar-sim
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv myEnv
   source myEnv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## License

This project is released under the MIT License. You are free to reuse, modify, and distribute.
