"""
neuroview.scene
===============
Plotly scene builder for brain electrode visualisation.

The central class is BrainScene, which accumulates surfaces, volumes,
segmentations, and electrode sets, then compiles them into a Plotly figure.

Capabilities
------------
- Glass brain surface        (FreeSurfer binary or GIFTI)
- Additional opaque surfaces (segmentations, sub-structures)
- NIfTI label → isosurface   (marching cubes, requires scipy + scikit-image)
- MRI slice planes           (axial / coronal / sagittal) in 3-D
- Electrode scatter          (uniform colour, group-coloured, scalar-weighted)
- Highlighted electrodes     (recording / stimulation site)
- Legend, title, camera, save (HTML / PNG / SVG)
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    raise ImportError("neuroview requires plotly:  pip install plotly")

from .surface import Surface
from .volume import Volume
from .electrodes import ElectrodeSet
from .surface_activity import gaussian_electrode_projection

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _glass_lighting(cam_eye: dict) -> dict:
    """Camera-adaptive Gouraud lighting for transparent glass surfaces."""
    v = np.array([cam_eye.get("x", -1.5), cam_eye.get("y", 0.2), cam_eye.get("z", 1.2)])
    nrm = np.linalg.norm(v) or 1.0
    vz = abs(v[2] / nrm)
    return dict(ambient=0.20 + 0.15*(1-vz), diffuse=0.60 + 0.25*vz,
                specular=1.0, roughness=0.05, fresnel=1.0)


def _colormap_hex(n: int, cmap: str = "Turbo") -> List[str]:
    """n evenly-spaced hex colours from a Plotly sequential colorscale."""
    scale = getattr(px.colors.sequential, cmap, px.colors.sequential.Turbo)
    idxs = np.linspace(0, len(scale) - 1, max(n, 1)).astype(int)
    return [scale[i] for i in idxs]


def _scalar_colorscale(values: np.ndarray, colorscale: str = "RdBu_r") -> List[str]:
    """Map a float array to hex colours using a Plotly diverging colorscale."""
    try:
        import plotly.colors as pc
        normed = np.clip(
            (values - values.min()) / max(values.max() - values.min(), 1e-9), 0, 1
        )
        rgb_tuples = pc.sample_colorscale(colorscale, list(normed))
        out = []
        for rgb in rgb_tuples:
            nums = re.findall(r"\d+", rgb)
            r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
            out.append(f"#{r:02x}{g:02x}{b:02x}")
        return out
    except Exception:
        return ["#ff6666"] * len(values)


# ─── BrainScene ───────────────────────────────────────────────────────────────

class BrainScene:
    """
    Builder for interactive Plotly brain visualisations.

    Usage
    -----
    >>> scene = BrainScene()
    >>> scene.add_glass_brain(Surface.from_file("lh.pial", opacity=0.18))
    >>> scene.add_electrodes(elec_set)
    >>> scene.show()

    All add_* methods return self for optional method chaining.
    """

    def __init__(self, bg_color: str = "white", width: int = 1000, height: int = 800):
        self._traces: List[go.BaseTraceType] = []
        self._bg_color = bg_color
        self._width = width
        self._height = height
        self._cam_eye: dict = {"x": -1.8, "y": 0.1, "z": 0.6}
        self._title: str = ""

    # ── Camera ────────────────────────────────────────────────────────────────

    def set_camera(self, azimuth: float = -90, elevation: float = 0,
                   distance: float = 2.2) -> "BrainScene":
        """
        Set camera angle using MATLAB-compatible spherical conventions.

        Parameters
        ----------
        azimuth   : horizontal degrees (−90=left lateral, 0=frontal,
                    90=right lateral, 180=posterior)
        elevation : vertical degrees (0=equatorial, 90=top-down)
        distance  : camera distance (zoom)
        """
        az = np.radians(azimuth)
        el = np.radians(elevation)
        self._cam_eye = {
            "x": float(distance * np.cos(el) * np.cos(az)),
            "y": float(distance * np.cos(el) * np.sin(az)),
            "z": float(distance * np.sin(el)),
        }
        return self

    def set_camera_xyz(self, x: float, y: float, z: float) -> "BrainScene":
        """Set camera eye position directly in Plotly (x, y, z) coordinates."""
        self._cam_eye = {"x": x, "y": y, "z": z}
        return self

    # ── Surfaces ──────────────────────────────────────────────────────────────

    def add_glass_brain(self, surface: Surface, opacity: Optional[float] = None,
                        color: Optional[str] = None) -> "BrainScene":
        """
        Add a transparent 'glass brain' cortical surface.

        Parameters
        ----------
        surface : Surface loaded via Surface.from_file(...)
        opacity : override surface.opacity (0.12–0.22 typical)
        color   : override surface.color (hex)
        """
        op  = opacity if opacity is not None else surface.opacity
        col = color   if color   is not None else surface.color
        self._traces.append(go.Mesh3d(
            x=surface.vertices[:, 0], y=surface.vertices[:, 1], z=surface.vertices[:, 2],
            i=surface.faces[:, 0],   j=surface.faces[:, 1],   k=surface.faces[:, 2],
            color=col, opacity=op,
            flatshading=False,
            lighting=_glass_lighting(self._cam_eye),
            lightposition=dict(x=100, y=200, z=400),
            hoverinfo="skip", showlegend=False, name=surface.name,
        ))
        return self

    def add_surface(self, surface: Surface, opacity: Optional[float] = None,
                    color: Optional[str] = None, name: Optional[str] = None,
                    show_in_legend: bool = True) -> "BrainScene":
        """
        Add an opaque or semi-transparent anatomical surface
        (e.g. cingulate, hippocampus, thalamus).
        """
        op  = opacity if opacity is not None else surface.opacity
        col = color   if color   is not None else surface.color
        lbl = name    if name    is not None else surface.name
        self._traces.append(go.Mesh3d(
            x=surface.vertices[:, 0], y=surface.vertices[:, 1], z=surface.vertices[:, 2],
            i=surface.faces[:, 0],   j=surface.faces[:, 1],   k=surface.faces[:, 2],
            color=col, opacity=op, flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3,
                          roughness=0.4, fresnel=0.4),
            lightposition=dict(x=100, y=200, z=400),
            hoverinfo="text", hovertext=lbl,
            showlegend=show_in_legend, name=lbl,
        ))
        return self

    def add_segmentation(self, volume: Volume, label: int = 1,
                         smooth_sigma: float = 0.8, color: str = "#EDB085",
                         opacity: float = 0.75, name: str = "segmentation") -> "BrainScene":
        """
        Extract a label from a NIfTI segmentation and render as a surface.

        Requires scipy and scikit-image.
        """
        verts, faces = volume.extract_isosurface(label, smooth_sigma)
        surf = Surface(vertices=verts, faces=faces, name=name,
                       color=color, opacity=opacity)
        return self.add_surface(surf, opacity=opacity, color=color, name=name)

    # ── MRI slice planes ──────────────────────────────────────────────────────

    def add_mri_slice(self, volume: Volume, plane: str, coord_mm: float,
                      opacity: float = 0.90, colorscale: str = "Gray",
                      name: str = "") -> "BrainScene":
        """
        Render an MRI slice as a coloured plane in 3-D space.

        Parameters
        ----------
        plane      : 'axial'/'z',  'coronal'/'y',  'sagittal'/'x'
        coord_mm   : world-space mm coordinate of the slice
        colorscale : Plotly colorscale name ('Gray', 'Hot', etc.)
        """
        xx, yy, zz, sc = volume.slice_plane_grid(plane, coord_mm)
        self._traces.append(go.Surface(
            x=xx, y=yy, z=zz, surfacecolor=sc,
            cmin=0.0, cmax=1.0, colorscale=colorscale,
            showscale=False, opacity=opacity,
            hoverinfo="skip", showlegend=False, name=name,
        ))
        return self

    # ── Electrodes ────────────────────────────────────────────────────────────

    def add_electrodes(self, elec: ElectrodeSet, color: str = "gold",
                       size: int = 7, show_labels: bool = True,
                       label_size: int = 9, name: Optional[str] = None) -> "BrainScene":
        """Add all electrodes in a uniform colour."""
        coords = elec.coords
        lbl = name if name is not None else elec.name
        self._traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text" if show_labels else "markers",
            text=elec.names,
            textposition="top right",
            textfont=dict(size=label_size, color="black"),
            marker=dict(size=size, color=color,
                        line=dict(color="black", width=1), symbol="circle"),
            hovertext=elec.names, hoverinfo="text",
            name=lbl, showlegend=True,
        ))
        return self

    def add_electrodes_grouped(self, elec: ElectrodeSet, group_column: str,
                                cmap: str = "Turbo", size: int = 7,
                                show_labels: bool = True,
                                label_size: int = 9) -> "BrainScene":
        """
        Plot electrodes colour-coded by an anatomical or clinical grouping column.

        Parameters
        ----------
        group_column : column name in elec.df (e.g. 'Destrieux_label_text')
        cmap         : Plotly sequential colormap name
        """
        if group_column not in elec.df.columns:
            raise KeyError(f"Column '{group_column}' not found. "
                           f"Available: {list(elec.df.columns)}")
        color_map = elec.assign_colors(group_column, cmap)
        for grp, coords in elec.group_coords(group_column).items():
            names_g  = elec.group_names(group_column).get(grp, [])
            hex_col  = color_map.get(grp, "#888888")
            self._traces.append(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers+text" if show_labels else "markers",
                text=names_g, textposition="top right",
                textfont=dict(size=label_size, color="black"),
                marker=dict(size=size, color=hex_col,
                            line=dict(color="black", width=1)),
                hovertext=names_g, hoverinfo="text",
                name=grp, showlegend=True,
            ))
        return self

    def add_electrodes_weighted(self, elec: ElectrodeSet, weights: np.ndarray,
                                 colorscale: str = "RdBu_r",
                                 size_range: Tuple[int, int] = (5, 16),
                                 show_labels: bool = True, label_size: int = 9,
                                 name: str = "weighted",
                                 weight_max: Optional[float] = None) -> "BrainScene":
        """
        Plot electrodes with size and colour encoding a scalar weight
        (e.g. alpha-prime, CCEP amplitude, significance).

        Parameters
        ----------
        weights     : (N,) array of scalar values
        colorscale  : Plotly diverging colorscale
        size_range  : (min_px, max_px) for marker scaling
        weight_max  : clip weights to ±weight_max before scaling
        """
        weights = np.asarray(weights, dtype=float)
        wmax = weight_max if weight_max else float(np.nanmax(np.abs(weights)) or 1.0)
        wc   = np.clip(weights, -wmax, wmax)
        hex_colors = _scalar_colorscale(wc, colorscale)
        rel  = np.abs(wc) / wmax
        s_min, s_max = size_range
        sizes = (s_min + (s_max - s_min) * rel).tolist()
        coords = elec.coords
        self._traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text" if show_labels else "markers",
            text=elec.names, textposition="top right",
            textfont=dict(size=label_size, color="black"),
            marker=dict(size=sizes, color=hex_colors, colorscale=colorscale,
                        colorbar=dict(title=name, thickness=12, len=0.4),
                        showscale=True, line=dict(color="black", width=0.8)),
            hovertext=[f"{n}: {w:.3f}" for n, w in zip(elec.names, weights)],
            hoverinfo="text", name=name, showlegend=True,
        ))
        return self

    def add_highlight_electrode(self, coord: np.ndarray, label: str = "highlight",
                                 color: str = "red", size: int = 14,
                                 edge_color: str = "darkred") -> "BrainScene":
        """
        Mark a single electrode prominently (e.g. recording / stimulation site).

        Parameters
        ----------
        coord : (3,) array [x, y, z] in mm
        """
        coord = np.atleast_2d(np.asarray(coord, dtype=float))
        self._traces.append(go.Scatter3d(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            mode="markers+text",
            text=[label], textposition="top right",
            textfont=dict(size=11, color="black"),
            marker=dict(size=size, color=color,
                        line=dict(color=edge_color, width=2), symbol="circle"),
            hovertext=[label], hoverinfo="text",
            name=label, showlegend=True,
        ))
        return self
    
    def add_surface_activity(
        self,
        surface: "Surface",
        electrode_coords: "np.ndarray",
        weights: "Optional[np.ndarray]" = None,
        sigma: float = 50.0,
        cutoff_factor: float = 5.0,
        colorscale: str = "RdBu_r",
        symmetric_clim: bool = True,
        opacity: float = 1.0,
        show_colorbar: bool = True,
        colorbar_title: str = "Activity",
        name: str = "surface activity",
    ) -> "BrainScene":
        """
        Render a cortical surface coloured by Gaussian-projected electrode
        activity. 

        Parameters
        ----------
        surface : Surface
            Triangulated cortical mesh (FreeSurfer or GIFTI).
        electrode_coords : ndarray (K, 3)
            Electrode positions in RAS mm.
        weights : ndarray (K,) or None
            Per-electrode scalar weights.  Pass ``None`` for a plain
            proximity/coverage map (all weights = 1).
        sigma : float
            Gaussian spread in mm^2 (default 50 -> ~7 mm FWHM).
            Higher values produce broader, smoother blobs. 
        symmetric_clim : bool
            If True, colour limits are +-max(|activity|) so zero maps to
            the midpoint of the colorscale (best for signed data).
        opacity : float
            Surface opacity 0-1.  
        """
        from .surface_activity import gaussian_electrode_projection

        electrode_coords = np.asarray(electrode_coords, dtype=float)
        if electrode_coords.ndim == 1:
            electrode_coords = electrode_coords[np.newaxis]

        activity = gaussian_electrode_projection(
            surface.vertices,
            electrode_coords,
            weights=weights,
            sigma=sigma,
            cutoff_factor=cutoff_factor,
        )

        if symmetric_clim:
            vmax = float(np.max(np.abs(activity))) or 1.0
            cmin, cmax = -vmax, vmax
        else:
            cmin = float(activity.min())
            cmax = float(activity.max()) or 1.0

        self._traces.append(go.Mesh3d(
            x=surface.vertices[:, 0],
            y=surface.vertices[:, 1],
            z=surface.vertices[:, 2],
            i=surface.faces[:, 0],
            j=surface.faces[:, 1],
            k=surface.faces[:, 2],
            intensity=activity.tolist(),
            intensitymode="vertex",
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                thickness=14,
                len=0.45,
                x=1.02,
            ) if show_colorbar else None,
            opacity=opacity,
            flatshading=False,
            lighting=dict(ambient=0.35, diffuse=0.75, specular=0.4,
                          roughness=0.5, fresnel=0.3),
            lightposition=dict(x=100, y=200, z=400),
            hoverinfo="text",
            hovertext=[f"{v:.4f}" for v in activity],
            showlegend=True,
            name=name,
        ))
        return self

    # ── Figure compilation ───────────────────────────────────────────────────

    def set_title(self, title: str) -> "BrainScene":
        self._title = title
        return self

    def build(self) -> go.Figure:
        """Compile all added traces into a Plotly Figure."""
        layout = go.Layout(
            title=dict(text=self._title, font=dict(size=15), x=0.5)
                  if self._title else None,
            width=self._width, height=self._height,
            paper_bgcolor=self._bg_color,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor=self._bg_color,
                aspectmode="data",
                camera=dict(eye=self._cam_eye, up=dict(x=0, y=0, z=1)),
            ),
            legend=dict(font=dict(size=10), x=0.82, y=0.95,
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="lightgrey", borderwidth=1,
                        itemsizing="constant"),
            margin=dict(l=0, r=0, t=40 if self._title else 0, b=0),
        )
        return go.Figure(data=list(self._traces), layout=layout)

    # ── Output ───────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Open the interactive figure in the browser or notebook."""
        self.build().show()

    def save_html(self, path: str | Path,
                  include_plotlyjs: str = "cdn") -> None:
        """
        Save as a self-contained interactive HTML file.

        Parameters
        ----------
        include_plotlyjs : 'cdn' (small, needs internet) or
                           'inline' (large, fully offline)
        """
        path = Path(path)
        self.build().write_html(str(path), include_plotlyjs=include_plotlyjs)
        print(f"[neuroview] Saved HTML → {path}")

    def save_png(self, path: str | Path, scale: int = 2) -> None:
        """Save as PNG. Requires kaleido:  pip install kaleido"""
        path = Path(path)
        try:
            self.build().write_image(str(path), scale=scale)
            print(f"[neuroview] Saved PNG → {path}")
        except Exception as e:
            raise RuntimeError(
                f"PNG export failed: {e}\nInstall kaleido:  pip install kaleido"
            ) from e

    def save_svg(self, path: str | Path, scale: int = 2) -> None:
        """Save as SVG. Requires kaleido:  pip install kaleido"""
        path = Path(path)
        try:
            self.build().write_image(str(path), format="svg", scale=scale)
            print(f"[neuroview] Saved SVG → {path}")
        except Exception as e:
            raise RuntimeError(
                f"SVG export failed: {e}\nInstall kaleido:  pip install kaleido"
            ) from e