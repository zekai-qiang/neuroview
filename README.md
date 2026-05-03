# neuroview
A Python library for visualizing brain electrodes, volumes, and surfaces. 

Functionalities:
- Interactive Brain Scenes: Build 3D scenes with "glass brain" transparency, opaque anatomical structures, and custom lighting.
- Electrode Visualization: Render electrodes with uniform colors, group-based color coding (e.g., by anatomical label), or scalar weighting (e.g., spectral power or CCEP amplitude).
- MRI Slice Planes: Embed axial, coronal, or sagittal MRI slices into the 3D electrode space.
- Activity Mapping: Project electrode-level activity onto cortical surface meshes using Gaussian kernels.
- Anatomical Segmentations: Automatically extract and render isosurfaces from NIfTI label volumes using marching cubes.
- Multi-Format Support: Built-in readers for common neuroimaging formats including FreeSurfer, GIFTI, and NIfTI.

The library requires the following dependencies: plotly, pandas, numpy, scipy, scikit-image, kaleido

Zekai Qiang, May 2026
