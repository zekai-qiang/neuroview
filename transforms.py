"""
neuroview.transforms
====================
Coordinate transformation utilities.

All transforms operate in RAS (Right-Anterior-Superior) world-mm space,
standard for both FreeSurfer and NIfTI files.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class CoordTransforms:
    """Static coordinate transformation methods."""

    @staticmethod
    def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """
        Map voxel indices to world (mm) coordinates using a 4×4 affine.

        Parameters
        ----------
        voxel_coords : ndarray (N, 3)
        affine       : ndarray (4, 4)

        Returns
        -------
        ndarray (N, 3) — world coordinates in mm
        """
        vox = np.asarray(voxel_coords, dtype=float)
        if vox.ndim == 1:
            vox = vox[np.newaxis]
        hom = np.c_[vox, np.ones(len(vox))]
        return (affine @ hom.T).T[:, :3]

    @staticmethod
    def world_to_voxel(world_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Map world (mm) coordinates to fractional voxel indices."""
        inv = np.linalg.inv(affine)
        world = np.asarray(world_coords, dtype=float)
        if world.ndim == 1:
            world = world[np.newaxis]
        hom = np.c_[world, np.ones(len(world))]
        return (inv @ hom.T).T[:, :3]

    @staticmethod
    def axis_coords_from_affine(
        shape: Tuple[int, int, int],
        affine: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns 1-D world-coordinate arrays where coords[i] is the world
        coordinate of voxel index i along that dimension.
        Preserves voxel-index order — no sorting or flipping.
        """
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        origin = affine[:3, 3]

        # Column i of the affine rotation/scale block gives the world-space
        # step when incrementing voxel index along dimension i.
        sx = float(affine[0, 0])   # world-X step per voxel in dim 0
        sy = float(affine[1, 1])   # world-Y step per voxel in dim 1
        sz = float(affine[2, 2])   # world-Z step per voxel in dim 2

        x = origin[0] + np.arange(nx) * sx   # length == shape[0], index-matched
        y = origin[1] + np.arange(ny) * sy   # length == shape[1], index-matched
        z = origin[2] + np.arange(nz) * sz   # length == shape[2], index-matched

        return x, y, z

    @staticmethod
    def project_to_plane(
        coords: np.ndarray,
        plane: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3-D coordinates onto a 2-D anatomical plane.

        Parameters
        ----------
        plane : 'axial'/'z',  'coronal'/'y',  'sagittal'/'x'

        Returns
        -------
        (u, v) : two 1-D ndarrays — in-plane axes
        """
        p = plane.lower()
        if p in ("z", "axial"):
            return coords[:, 0], coords[:, 1]
        elif p in ("y", "coronal"):
            return coords[:, 0], coords[:, 2]
        elif p in ("x", "sagittal"):
            return coords[:, 1], coords[:, 2]
        else:
            raise ValueError(f"plane must be 'x'/'sagittal', 'y'/'coronal', or 'z'/'axial'")

    @staticmethod
    def left_hemisphere_mask(coords: np.ndarray) -> np.ndarray:
        """Boolean mask: x < 0 in RAS (left hemisphere)."""
        return np.asarray(coords)[:, 0] < 0

    @staticmethod
    def right_hemisphere_mask(coords: np.ndarray) -> np.ndarray:
        """Boolean mask: x > 0 in RAS (right hemisphere)."""
        return np.asarray(coords)[:, 0] > 0

    @staticmethod
    def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Smooth per-vertex normals by accumulating area-weighted face normals.

        Returns
        -------
        normals : ndarray (N, 3), unit length
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        vn = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(vn, faces[:, i], fn)
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.where(norms == 0, 1.0, norms)

    @staticmethod
    def vertex_shading(
        vertices: np.ndarray,
        faces: np.ndarray,
        light_dir: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Per-vertex Lambertian shading intensity in [0, 1].

        Parameters
        ----------
        light_dir : (3,) unit vector toward light source.
                    Defaults to upper-left-front.
        """
        if light_dir is None:
            light_dir = np.array([0.4, 0.6, 0.9])
        light_dir = np.asarray(light_dir, dtype=float)
        light_dir /= np.linalg.norm(light_dir)
        normals = CoordTransforms.compute_vertex_normals(vertices, faces)
        return np.clip(normals @ light_dir, 0.0, 1.0)