"""
neuroview.io
============
Low-level file readers.

Supported formats
-----------------
- FreeSurfer binary triangle surface  (lh.pial, rh.pial, lh.inflated, ...)
- GIFTI surface                        (.surf.gii)
- NIfTI volume                         (.nii / .nii.gz)
- Electrode table                      (.tsv, .csv)
"""

from __future__ import annotations

import gzip
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# FreeSurfer binary surface 

def read_freesurfer_surface(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a FreeSurfer binary triangle-mesh surface file.

    Parameters
    ----------
    path : str or Path
        Path to the surface file (e.g. ``lh.pial``).

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates in mm (RAS space).
    faces : ndarray, shape (F, 3)
        Zero-indexed triangle indices.

    Raises
    ------
    ValueError
        If the magic bytes do not match a triangle surface.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        magic = struct.unpack(">BBB", fh.read(3))
        if magic != (255, 255, 254):
            raise ValueError(
                f"{path.name}: not a FreeSurfer triangle surface "
                f"(magic={magic}, expected (255,255,254))"
            )
        # Skip two creator comment lines terminated by b'\n\n'
        buf = b""
        while not buf.endswith(b"\n\n"):
            buf += fh.read(1)

        n_verts, n_faces = struct.unpack(">II", fh.read(8))
        vertices = (
            np.frombuffer(fh.read(n_verts * 3 * 4), dtype=">f4")
            .reshape(n_verts, 3)
            .astype(np.float64)
        )
        faces = (
            np.frombuffer(fh.read(n_faces * 3 * 4), dtype=">i4")
            .reshape(n_faces, 3)
            .astype(np.int64)
        )
    return vertices, faces

# GIFTI surface  (.surf.gii) 

def read_gifti_surface(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a GIFTI surface file without nibabel.

    Supports plain XML GIFTI and gzip-compressed variants.

    Returns
    -------
    vertices : ndarray (N, 3)   — vertex coordinates
    faces    : ndarray (F, 3)   — triangle indices
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    root = ET.fromstring(raw.decode("utf-8", errors="replace"))

    vertices = faces = None
    for da in root.iter("DataArray"):
        intent = da.get("Intent", "")
        enc    = da.get("Encoding", "ASCII")
        dims   = [int(da.get(f"Dim{i}", 0)) for i in range(6) if da.get(f"Dim{i}")]

        data_el = da.find("Data")
        if data_el is None:
            continue
        text = (data_el.text or "").strip()

        if enc in ("ASCII", "TextDataArray"):
            arr = np.fromstring(text, sep="\n" if "\n" in text else " ")
        elif enc in ("Base64Binary", "GZipBase64Binary"):
            import base64
            b = base64.b64decode(text)
            if enc == "GZipBase64Binary":
                b = gzip.decompress(b)
            dt_str = da.get("DataType", "NIFTI_TYPE_FLOAT32")
            dtype  = np.float32 if "FLOAT" in dt_str else np.int32
            arr    = np.frombuffer(b, dtype=np.dtype(dtype).newbyteorder("<"))
        else:
            arr = np.fromstring(text, sep=" ")

        if dims:
            arr = arr.reshape(dims)

        if "POINTSET" in intent or "COORDINATES" in intent:
            vertices = arr.astype(np.float64)
        elif "TRIANGLE" in intent:
            faces = arr.astype(np.int64)

    if vertices is None or faces is None:
        raise ValueError(f"{path.name}: could not extract POINTSET/TRIANGLE arrays")

    return vertices, faces

# NIfTI volume  (.nii / .nii.gz)

_NII_DTYPES = {
    2:  np.uint8,
    4:  np.int16,
    8:  np.int32,
    16: np.float32,
    32: np.complex64,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}

def read_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a NIfTI-1 file (.nii or .nii.gz) without nibabel.

    Returns
    -------
    data   : ndarray (X, Y, Z [, T])   — voxel data, float32
    affine : ndarray (4, 4)            — voxel→world (RAS mm) transform
    """
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
    bo = "<" if sizeof_hdr == 348 else ">"

    datatype  = struct.unpack_from(f"{bo}H", raw, 70)[0]
    dim       = struct.unpack_from(f"{bo}8h", raw, 40)
    ndim      = dim[0]
    shape     = tuple(int(dim[i + 1]) for i in range(ndim))

    pixdim     = struct.unpack_from(f"{bo}8f", raw, 76)
    vox_offset = struct.unpack_from(f"{bo}f", raw, 108)[0]
    qform_code = struct.unpack_from(f"{bo}H", raw, 252)[0]
    sform_code = struct.unpack_from(f"{bo}H", raw, 254)[0]

    if sform_code > 0:
        srow_x = struct.unpack_from(f"{bo}4f", raw, 280)
        srow_y = struct.unpack_from(f"{bo}4f", raw, 296)
        srow_z = struct.unpack_from(f"{bo}4f", raw, 312)
        affine = np.array([srow_x, srow_y, srow_z, [0, 0, 0, 1]], dtype=np.float64)
    elif qform_code > 0:
        qb, qc, qd = struct.unpack_from(f"{bo}3f", raw, 256)
        qx, qy, qz = struct.unpack_from(f"{bo}3f", raw, 268)
        qfac = float(pixdim[0]) if pixdim[0] != 0 else 1.0
        qa = float(np.sqrt(max(1.0 - qb**2 - qc**2 - qd**2, 0.0)))
        dx, dy, dz = pixdim[1], pixdim[2], pixdim[3]
        R = np.array([
            [qa**2+qb**2-qc**2-qd**2, 2*(qb*qc-qa*qd),         2*(qb*qd+qa*qc)],
            [2*(qb*qc+qa*qd),         qa**2+qc**2-qb**2-qd**2, 2*(qc*qd-qa*qb)],
            [2*(qb*qd-qa*qc),         2*(qc*qd+qa*qb),         qa**2+qd**2-qb**2-qc**2],
        ], dtype=np.float64)
        R[:, 2] *= qfac
        affine = np.eye(4)
        affine[:3, :3] = R * np.array([dx, dy, dz])
        affine[:3, 3]  = [qx, qy, qz]
    else:
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0]).astype(np.float64)

    dtype  = _NII_DTYPES.get(datatype, np.float32)
    offset = int(max(vox_offset, 352))
    data   = np.frombuffer(raw[offset:], dtype=np.dtype(dtype).newbyteorder(bo))
    data = data[: int(np.prod(shape))].reshape(shape, order='F').astype(np.float32)

    return data, affine

# Electrode table  (.tsv / .csv)

def read_electrodes_tsv(path: str | Path, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Load an electrode coordinate table (BIDS TSV or generic CSV).

    Must contain at minimum columns: name, x, y, z.
    """
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    required = {"name", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Electrode file missing columns: {missing}")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    return df