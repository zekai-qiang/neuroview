"""
neuroview.electrodes
====================
Electrode data container and grouping utilities.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .io import read_electrodes_tsv
from .transforms import CoordTransforms


@dataclass
class ElectrodeSet:
    """
    A collection of intracranial electrodes with spatial coordinates
    and optional metadata (anatomical labels, weights, group colours).

    Attributes
    ----------
    df   : pd.DataFrame  — full table (must have name/x/y/z columns)
    name : str           — dataset label used in figure legends
    """
    df: pd.DataFrame
    name: str = "electrodes"

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None,
                  sep: str | None = None) -> "ElectrodeSet":
        """Load from a BIDS electrodes.tsv or CSV file."""
        path = Path(path)
        df = read_electrodes_tsv(path, sep=sep)
        return cls(df=df, name=name or path.stem)

    @classmethod
    def from_arrays(
        cls,
        coords: np.ndarray,
        names: Sequence[str],
        name: str = "electrodes",
        **extra_columns,
    ) -> "ElectrodeSet":
        """
        Construct from coordinate array and name list.

        Parameters
        ----------
        coords       : ndarray (N, 3)
        names        : sequence of str
        **extra_columns : keyword → list/array of length N
        """
        coords = np.asarray(coords, dtype=float)
        df = pd.DataFrame({"name": list(names),
                           "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
        for col, val in extra_columns.items():
            df[col] = val
        return cls(df=df, name=name)

    @property
    def coords(self) -> np.ndarray:
        """(N, 3) float64 coordinate array."""
        return self.df[["x", "y", "z"]].to_numpy(dtype=float)

    @property
    def names(self) -> List[str]:
        return self.df["name"].tolist()

    @property
    def n(self) -> int:
        return len(self.df)

    def left_hemisphere(self) -> "ElectrodeSet":
        """Subset: x < 0 (left hemisphere in RAS)."""
        mask = CoordTransforms.left_hemisphere_mask(self.coords)
        return ElectrodeSet(df=self.df[mask].reset_index(drop=True),
                            name=f"{self.name}_LH")

    def right_hemisphere(self) -> "ElectrodeSet":
        """Subset: x > 0 (right hemisphere in RAS)."""
        mask = CoordTransforms.right_hemisphere_mask(self.coords)
        return ElectrodeSet(df=self.df[mask].reset_index(drop=True),
                            name=f"{self.name}_RH")

    def filter_by(self, column: str, values) -> "ElectrodeSet":
        """Return subset where column is in values."""
        if isinstance(values, str):
            values = [values]
        mask = self.df[column].isin(values)
        return ElectrodeSet(df=self.df[mask].reset_index(drop=True), name=self.name)

    def exclude_by(self, column: str, values) -> "ElectrodeSet":
        """Return subset where column is NOT in values."""
        if isinstance(values, str):
            values = [values]
        mask = ~self.df[column].isin(values)
        return ElectrodeSet(df=self.df[mask].reset_index(drop=True), name=self.name)

    def get_electrode(self, name: str) -> Optional[np.ndarray]:
        """Return (3,) coordinate for a single electrode by name, or None."""
        row = self.df[self.df["name"] == name]
        if row.empty:
            return None
        return row[["x", "y", "z"]].to_numpy(dtype=float)[0]

    def group_coords(self, column: str) -> Dict[str, np.ndarray]:
        """Dict mapping each unique value in column → coordinate array (N_i, 3)."""
        return {
            str(g): sub[["x", "y", "z"]].to_numpy(dtype=float)
            for g, sub in self.df.groupby(column, sort=False)
        }

    def group_names(self, column: str) -> Dict[str, List[str]]:
        """Dict mapping each group → list of electrode names."""
        return {str(g): sub["name"].tolist()
                for g, sub in self.df.groupby(column, sort=False)}

    def assign_colors(self, column: str, cmap_name: str = "Turbo") -> Dict[str, str]:
        """
        Assign a hex colour to each unique value in column.

        Returns
        -------
        dict: group label → hex colour string
        """
        try:
            import plotly.express as px
            unique = self.df[column].dropna().unique().tolist()
            n = len(unique)
            scale = getattr(px.colors.sequential, cmap_name,
                            px.colors.sequential.Turbo)
            idxs = np.linspace(0, len(scale) - 1, max(n, 1)).astype(int)
            return {str(g): scale[i] for g, i in zip(unique, idxs)}
        except ImportError:
            import colorsys
            unique = self.df[column].dropna().unique().tolist()
            n = max(len(unique), 1)
            hues = np.linspace(0, 1, n, endpoint=False)
            colors = {}
            for g, h in zip(unique, hues):
                r, g_, b = colorsys.hsv_to_rgb(h, 0.85, 0.90)
                colors[str(g)] = f"#{int(r*255):02x}{int(g_*255):02x}{int(b*255):02x}"
            return colors

    def __repr__(self) -> str:
        return f"ElectrodeSet('{self.name}', {self.n} electrodes)"