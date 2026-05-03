"""
neuroview.volume
================
NIfTI volume container and utilities.

Provides:
- World-coordinate axis arrays from affine
- Axial / coronal / sagittal slice extraction
- Marching-cubes isosurface extraction (requires scipy + scikit-image)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from .io import read_nifti
from .transforms import CoordTransforms


@dataclass
class Volume:
    """
    A 3-D volumetric dataset with spatial metadata.

    Attributes
    ----------
    data    : ndarray (X, Y, Z)   — voxel intensities, float32
    affine  : ndarray (4, 4)      — voxel → world (RAS mm) transform
    path    : Path or None        — source file path
    """
    data: np.ndarray
    affine: np.ndarray
    path: Optional[Path] = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Volume":
        """Load a NIfTI-1 volume (.nii or .nii.gz)."""
        path = Path(path)
        data, affine = read_nifti(path)
        return cls(data=data, affine=affine, path=path)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape[:3]

    @property
    def x_coords(self) -> np.ndarray:
        return self._axis_coords()[0]

    @property
    def y_coords(self) -> np.ndarray:
        return self._axis_coords()[1]

    @property
    def z_coords(self) -> np.ndarray:
        return self._axis_coords()[2]

    def _axis_coords(self):
        if not hasattr(self, "_cached_coords"):
            self._cached_coords = CoordTransforms.axis_coords_from_affine(
                self.shape, self.affine
            )
        return self._cached_coords

    @property
    def world_bounds(self) -> dict:
        return {
            "x": (float(self.x_coords.min()), float(self.x_coords.max())),
            "y": (float(self.y_coords.min()), float(self.y_coords.max())),
            "z": (float(self.z_coords.min()), float(self.z_coords.max())),
        }

    def get_slice(
        self, plane: str, coord_mm: float
    ) -> Tuple[np.ndarray, float, Tuple, str, str]:
        """
        Extract a 2-D slice at a given world coordinate.

        Returns
        -------
        image  : ndarray (H, W)
        actual : float — actual mm coordinate of nearest voxel
        extent : (x0, x1, y0, y1) for imshow
        xlabel, ylabel : str
        """
        p = plane.lower()
        if p in ("z", "axial"):
            coords = self.z_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            image = self.data[:, :, idx].T
            extent = (self.x_coords[0], self.x_coords[-1],
                      self.y_coords[0], self.y_coords[-1])
            return image, float(coords[idx]), extent, "X (mm)", "Y (mm)"
        elif p in ("y", "coronal"):
            coords = self.y_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            image = self.data[:, idx, :].T
            extent = (self.x_coords[0], self.x_coords[-1],
                      self.z_coords[0], self.z_coords[-1])
            return image, float(coords[idx]), extent, "X (mm)", "Z (mm)"
        elif p in ("x", "sagittal"):
            coords = self.x_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            image = self.data[idx, :, :].T
            extent = (self.y_coords[0], self.y_coords[-1],
                      self.z_coords[0], self.z_coords[-1])
            return image, float(coords[idx]), extent, "Y (mm)", "Z (mm)"
        else:
            raise ValueError(f"plane must be 'x'/'sagittal', 'y'/'coronal', or 'z'/'axial'")

    def slice_plane_grid(
        self, plane: str, coord_mm: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = plane.lower()

        if p in ("z", "axial"):
            coords = self.z_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            sc = self.data[:, :, idx].astype(float)
            xx, yy = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")
            zz = np.full_like(xx, float(coords[idx])) 

        elif p in ("y", "coronal"):
            coords = self.y_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            sc = self.data[:, idx, :].T.astype(float)
            zz, xx = np.meshgrid(self.z_coords, self.x_coords, indexing="ij")
            yy = np.full_like(xx, float(coords[idx])) 

        elif p in ("x", "sagittal"):
            coords = self.x_coords
            idx = int(np.argmin(np.abs(coords - coord_mm)))
            sc = self.data[idx, :, :].T.astype(float) 
            zz, yy = np.meshgrid(self.z_coords, self.y_coords, indexing="ij") 
            xx = np.full_like(yy, float(coords[idx]))

        else:
            raise ValueError(f"Invalid plane '{plane}'")

        lo, hi = float(sc.min()), float(sc.max())
        sc = (sc - lo) / (hi - lo) if hi > lo else np.zeros_like(sc)
        return xx, yy, zz, sc

    def extract_isosurface(
        self,
        label: int = 1,
        smooth_sigma: float = 0.8,
        iso_level: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract an isosurface mesh from a label segmentation volume.

        Requires: scipy and scikit-image.

        Returns
        -------
        vertices : ndarray (N, 3) in world mm
        faces    : ndarray (F, 3)
        """
        try:
            from scipy import ndimage
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError(
                "extract_isosurface requires scipy and scikit-image:\n"
                "  pip install scipy scikit-image"
            )
        mask = (self.data == label).astype(float)
        if smooth_sigma > 0:
            mask = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
        verts, faces, _, _ = marching_cubes(mask, level=iso_level, spacing=(1.0, 1.0, 1.0))
        verts_world = CoordTransforms.voxel_to_world(verts, self.affine)
        return verts_world, faces.astype(np.int64)

    def __repr__(self) -> str:
        b = self.world_bounds
        return (
            f"Volume(shape={self.shape}, "
            f"x=[{b['x'][0]:.0f},{b['x'][1]:.0f}] "
            f"y=[{b['y'][0]:.0f},{b['y'][1]:.0f}] "
            f"z=[{b['z'][0]:.0f},{b['z'][1]:.0f}])"
        )