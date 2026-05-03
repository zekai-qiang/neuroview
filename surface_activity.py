
"""
neuroview.surface_activity
==========================
Gaussian projection of electrode activity onto the cortical surface.

Each electrode contributes a Gaussian "blob" of activity to nearby mesh vertices, 
which is then rendered as a per-vertex colour overlay on the glass-brain mesh.

"""

from __future__ import annotations
from typing import Optional
import numpy as np


def gaussian_electrode_projection(
    vertices: np.ndarray,
    electrode_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma: float = 50.0,
    cutoff_factor: float = 5.0,
) -> np.ndarray:
    """
    Project electrode activity onto surface vertices using a Gaussian kernel.

    Mirrors ``gaussian_proj_v2.m``:
      activity(v) = sum_k  w_k * exp(-||v - e_k||^2 / sigma)

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Surface vertex coordinates in mm (RAS).
    electrode_coords : ndarray (K, 3)
        Electrode positions in mm (RAS).
    weights : ndarray (K,) or None
        Per-electrode scalar weights (e.g. spectral power, CCEP amplitude).
        Pass ``None`` or an array of ones for a coverage / proximity map.
    sigma : float
        Gaussian spread parameter in mm^2 (default 50 -> ~7 mm FWHM).
        Larger values produce broader, smoother blobs.
    cutoff_factor : float
        Vertices farther than sqrt(cutoff_factor * sigma) mm from every
        electrode receive no contribution.

    Returns
    -------
    activity : ndarray (N,)
        Per-vertex activity values in the same units as *weights*.
    """
    vertices = np.asarray(vertices, dtype=float)
    elec = np.asarray(electrode_coords, dtype=float)
    if elec.ndim == 1:
        elec = elec[np.newaxis]

    if weights is None:
        weights = np.ones(len(elec), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(elec),):
            raise ValueError(
                f"weights must have length {len(elec)}, got {weights.shape}"
            )

    activity = np.zeros(len(vertices), dtype=float)
    radius_cutoff = np.sqrt(cutoff_factor * sigma)

    for epos, w in zip(elec, weights):
        diff = vertices - epos                        # (N, 3)
        dist_sq = np.einsum("ij,ij->i", diff, diff)  # (N,) — vectorised
        nearby = dist_sq < radius_cutoff ** 2
        if not np.any(nearby):
            continue
        activity[nearby] += w * np.exp(-dist_sq[nearby] / sigma)

    return activity