"""
neuroview
=========
A Python library for visualising brain electrodes, volumes and surfaces.

Modules
-------
io                  — File readers (FreeSurfer binary, GIFTI, NIfTI)
transforms          — Coordinate system utilities (voxel↔world, affine)
surface             — Surface loaders and vertex-normal shading
surface_activity    — Surface activity mapping and plotting
volume              — NIfTI volume loading, axis-coordinate extraction, isosurface
electrodes          — Electrode data containers and grouping utilities
scene               — Plotly scene builder

See demo.py for a fully annotated usage example.
"""

from .io import (
    read_freesurfer_surface,
    read_gifti_surface,
    read_nifti,
    read_electrodes_tsv,
)
from .transforms import CoordTransforms
from .surface import Surface
from .volume import Volume
from .electrodes import ElectrodeSet
from .scene import BrainScene

__version__ = "1.0.0"
__all__ = [
    "read_freesurfer_surface",
    "read_gifti_surface",
    "read_nifti",
    "read_electrodes_tsv",
    "CoordTransforms",
    "Surface",
    "Volume",
    "ElectrodeSet",
    "BrainScene",
]