"""
neuroview.surface
=================
Surface container and loader.

Supports FreeSurfer binary and GIFTI formats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .io import read_freesurfer_surface, read_gifti_surface
from .transforms import CoordTransforms


@dataclass
class Surface:
    """
    A triangulated 3-D surface mesh.

    Attributes
    ----------
    vertices  : ndarray (N, 3)   — vertex coordinates in mm (RAS)
    faces     : ndarray (F, 3)   — zero-indexed triangle indices
    name      : str              — human-readable label
    color     : str              — default hex colour  (e.g. '#B8C4D6')
    opacity   : float            — default opacity 0–1
    """
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "surface"
    color: str = "#B8C4D6"
    opacity: float = 0.20
    _shading: np.ndarray = field(default=None, init=False, repr=False)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        name: str | None = None,
        color: str = "#B8C4D6",
        opacity: float = 0.20,
    ) -> "Surface":
        """
        Load a surface from a FreeSurfer binary or GIFTI file.

        Format is inferred from extension:
        - .gii             → GIFTI
        - anything else    → FreeSurfer binary (lh.pial, rh.inflated, etc.)
        """
        path = Path(path)
        if name is None:
            name = path.name
        if path.suffix.lower() == ".gii":
            verts, faces = read_gifti_surface(path)
        else:
            verts, faces = read_freesurfer_surface(path)
        return cls(vertices=verts, faces=faces, name=name, color=color, opacity=opacity)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> dict:
        """Bounding box as {'x': (min, max), 'y': ..., 'z': ...}."""
        return {
            ax: (float(self.vertices[:, i].min()), float(self.vertices[:, i].max()))
            for i, ax in enumerate("xyz")
        }

    @property
    def shading(self) -> np.ndarray:
        """Per-vertex Lambertian shading intensity (cached)."""
        if self._shading is None:
            self._shading = CoordTransforms.vertex_shading(self.vertices, self.faces)
        return self._shading

    def left_hemisphere(self) -> "Surface":
        """Return a new Surface restricted to x < 0 vertices."""
        return self._filter_hemisphere("left")

    def right_hemisphere(self) -> "Surface":
        """Return a new Surface restricted to x > 0 vertices."""
        return self._filter_hemisphere("right")

    def _filter_hemisphere(self, side: str) -> "Surface":
        mask = self.vertices[:, 0] < 0 if side == "left" else self.vertices[:, 0] > 0
        keep_idx = np.where(mask)[0]
        idx_map = -np.ones(len(self.vertices), dtype=np.int64)
        idx_map[keep_idx] = np.arange(len(keep_idx))
        face_mask = np.all(mask[self.faces], axis=1)
        new_faces = idx_map[self.faces[face_mask]]
        return Surface(
            vertices=self.vertices[keep_idx].copy(),
            faces=new_faces,
            name=f"{self.name}_{side}",
            color=self.color,
            opacity=self.opacity,
        )

    def __repr__(self) -> str:
        b = self.bounds
        return (
            f"Surface('{self.name}', {self.n_vertices:,} verts, {self.n_faces:,} faces, "
            f"x=[{b['x'][0]:.0f},{b['x'][1]:.0f}] "
            f"y=[{b['y'][0]:.0f},{b['y'][1]:.0f}] "
            f"z=[{b['z'][0]:.0f},{b['z'][1]:.0f}])"
        )