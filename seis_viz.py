import numpy as np
import pandas as pd
import xarray as xr
import segyio
from scipy.interpolate import griddata
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_seismic_cube(sgy_path):
    """Load a 3D SEG-Y file into an xarray Dataset."""
    with segyio.open(sgy_path, iline=189, xline=193) as f:
        ilines = f.ilines.copy()
        xlines = f.xlines.copy()
        twt = f.samples.copy()

        # Read 3D amplitude data
        data = segyio.tools.cube(f)

        # Read CDP coordinates per trace
        n_il, n_xl = len(ilines), len(xlines)
        cdp_x = np.zeros((n_il, n_xl), dtype=np.float32)
        cdp_y = np.zeros((n_il, n_xl), dtype=np.float32)

        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        scale = 1.0 / abs(scalar) if scalar < 0 else float(max(scalar, 1))

        for i, il in enumerate(ilines):
            for j, xl in enumerate(xlines):
                idx = i * n_xl + j
                h = f.header[idx]
                cdp_x[i, j] = h[segyio.TraceField.CDP_X] * scale
                cdp_y[i, j] = h[segyio.TraceField.CDP_Y] * scale

    cube = xr.Dataset(
        {"data": (["iline", "xline", "twt"], data)},
        coords={
            "iline": ilines,
            "xline": xlines,
            "twt": twt,
            "cdp_x": (["iline", "xline"], cdp_x),
            "cdp_y": (["iline", "xline"], cdp_y),
        },
    )
    return cube


def load_horizon(dat_path):
    """Load a whitespace-delimited horizon file (cdp_x, cdp_y, twt)."""
    hrz = pd.read_csv(
        dat_path, names=["cdp_x", "cdp_y", "twt"], sep=r"\s+"
    )
    return hrz


def map_horizon_to_grid(cube, hrz):
    """Interpolate scattered horizon points onto the cube's inline/crossline grid."""
    cdp_x_grid = cube.cdp_x.values
    cdp_y_grid = cube.cdp_y.values

    twt_grid = griddata(
        points=hrz[["cdp_x", "cdp_y"]].values,
        values=hrz["twt"].values,
        xi=(cdp_x_grid, cdp_y_grid),
        method="linear",
    )

    hrz_mapped = xr.Dataset(
        coords={
            "iline": cube.iline,
            "xline": cube.xline,
            "twt": (["iline", "xline"], twt_grid),
            "cdp_x": cube.cdp_x,
            "cdp_y": cube.cdp_y,
        }
    )
    return hrz_mapped


def get_corner_points(cube):
    """Return the 4 corner CDP coordinates of the survey as an (5, 2) array.

    Order: top-left, top-right, bottom-right, bottom-left, top-left (closed polygon).
    """
    cx = cube.cdp_x.values
    cy = cube.cdp_y.values
    corners = np.array([
        [cx[0, 0], cy[0, 0]],
        [cx[0, -1], cy[0, -1]],
        [cx[-1, -1], cy[-1, -1]],
        [cx[-1, 0], cy[-1, 0]],
        [cx[0, 0], cy[0, 0]],
    ])
    return corners


def get_affine_transform(cube):
    """Compute an Affine2D mapping inline/crossline indices to CDP x/y.

    Uses 3 corner points to solve the affine system:
        cdp_x = a * iline + b * xline + tx
        cdp_y = c * iline + d * xline + ty
    """
    il = cube.iline.values.astype(float)
    xl = cube.xline.values.astype(float)
    cx = cube.cdp_x.values
    cy = cube.cdp_y.values

    src = np.array([
        [il[0], xl[0], 1],
        [il[0], xl[-1], 1],
        [il[-1], xl[0], 1],
    ])
    dst_x = np.array([cx[0, 0], cx[0, -1], cx[-1, 0]])
    dst_y = np.array([cy[0, 0], cy[0, -1], cy[-1, 0]])

    coeffs_x = np.linalg.solve(src, dst_x)
    coeffs_y = np.linalg.solve(src, dst_y)

    # from_values uses SVG convention: matrix is [[a,c,e],[b,d,f]]
    # so x_out = a*x + c*y + e, y_out = b*x + d*y + f
    return Affine2D.from_values(
        coeffs_x[0], coeffs_y[0], coeffs_x[1], coeffs_y[1],
        coeffs_x[2], coeffs_y[2],
    )


def plot_map_and_section(cube, hrz_mapped, corners, inline_val, fig=None):
    """Plot a dual-panel figure: map view (left) + seismic section (right).

    Parameters
    ----------
    cube : xr.Dataset
        Seismic cube from load_seismic_cube().
    hrz_mapped : xr.Dataset
        Mapped horizon from map_horizon_to_grid().
    corners : np.ndarray
        Survey corners from get_corner_points(), shape (5, 2).
    inline_val : int
        Inline number to display.
    fig : optional
        Reuse existing figure (for animation). If None, creates a new one.

    Returns
    -------
    fig, axes : matplotlib Figure and array of two Axes
    """
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.2, 1]})
    else:
        fig.clear()
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})

    tform = get_affine_transform(cube)
    ax_map, ax_seis = axes

    # --- Left panel: map view ---
    # All map elements are plotted in CDP coordinate space (no runtime transforms)
    # to avoid matplotlib rendering issues with pcolormesh + affine transforms.

    # Helper: transform iline/xline points to CDP x/y
    def to_cdp(*il_xl_pairs):
        pts = np.array(il_xl_pairs, dtype=float)
        return tform.transform(pts)

    # Horizon surface as pcolormesh in CDP coords
    il_vals = hrz_mapped.iline.values.astype(float)
    xl_vals = hrz_mapped.xline.values.astype(float)
    IL, XL = np.meshgrid(il_vals, xl_vals, indexing="ij")
    cdp_pts = tform.transform(np.column_stack([IL.ravel(), XL.ravel()]))
    CDP_X = cdp_pts[:, 0].reshape(IL.shape)
    CDP_Y = cdp_pts[:, 1].reshape(IL.shape)

    ax_map.pcolormesh(CDP_X, CDP_Y, hrz_mapped.twt.values, shading="auto", cmap="viridis_r")
    ax_map.set_aspect(1)
    ax_map.axis("off")

    # Survey boundary
    ax_map.plot(corners[:, 0], corners[:, 1], lw=4, c="white")

    # Current inline indicator
    il_min, il_max = float(cube.iline[0]), float(cube.iline[-1])
    xl_min, xl_max = float(cube.xline[0]), float(cube.xline[-1])
    il_line = to_cdp([inline_val, xl_min], [inline_val, xl_max])
    ax_map.plot(il_line[:, 0], il_line[:, 1], color="k", lw=2)

    props = dict(facecolor="white", edgecolor="k", alpha=1)
    label_pt = to_cdp([inline_val, xl_max])[0]
    ax_map.annotate(
        str(inline_val), xy=label_pt, fontsize=8, bbox=props,
        xytext=(5, 10), textcoords="offset points",
    )

    # Grid overlay
    n_gridlines = 5
    grid_il = np.round(np.linspace(il_min, il_max, n_gridlines).astype("int64"), -1)
    grid_xl = np.round(np.linspace(xl_min, xl_max, n_gridlines).astype("int64"), -1)

    for i in range(n_gridlines):
        # Crossline grid line (constant xline, varying iline)
        gl = to_cdp([il_min, grid_xl[i]], [il_max, grid_xl[i]])
        ax_map.plot(gl[:, 0], gl[:, 1], color="grey", lw=0.5)
        # Inline grid line (constant iline, varying xline)
        gl = to_cdp([grid_il[i], xl_min], [grid_il[i], xl_max])
        ax_map.plot(gl[:, 0], gl[:, 1], color="grey", lw=0.5)
        # Crossline label
        label_pt = to_cdp([il_max, grid_xl[i]])[0]
        ax_map.annotate(
            str(grid_xl[i]), xy=label_pt, fontsize=10,
            xytext=(8, 0), textcoords="offset points",
        )

    # Axis labels centered along their edges, rotated to follow the edge
    survey_center = corners[:4].mean(axis=0)
    offset_dist = 200  # meters from edge midpoint

    def readable_angle(deg):
        """Normalize rotation so text is never upside-down."""
        deg = deg % 360
        if deg > 90 and deg <= 270:
            deg -= 180
        return deg

    # Inline label along the left edge (constant xl_min, varying iline)
    il_start = to_cdp([il_min, xl_min])[0]
    il_end = to_cdp([il_max, xl_min])[0]
    il_mid = (il_start + il_end) / 2
    il_angle = np.degrees(np.arctan2(il_end[1] - il_start[1], il_end[0] - il_start[0]))
    outward = il_mid - survey_center
    outward = outward / np.linalg.norm(outward) * offset_dist
    ax_map.text(
        il_mid[0] + outward[0], il_mid[1] + outward[1],
        "Inline #", fontsize=10, ha="center", va="center",
        rotation=readable_angle(il_angle), rotation_mode="anchor",
    )

    # Crossline label along the top edge (constant il_min, varying xline)
    xl_start = to_cdp([il_min, xl_min])[0]
    xl_end = to_cdp([il_min, xl_max])[0]
    xl_mid = (xl_start + xl_end) / 2
    xl_angle = np.degrees(np.arctan2(xl_end[1] - xl_start[1], xl_end[0] - xl_start[0]))
    outward = xl_mid - survey_center
    outward = outward / np.linalg.norm(outward) * offset_dist
    ax_map.text(
        xl_mid[0] + outward[0], xl_mid[1] + outward[1],
        "Crossline #", fontsize=10, ha="center", va="center",
        rotation=readable_angle(xl_angle), rotation_mode="anchor",
    )

    # Set map limits to survey extent
    pad = 500
    ax_map.set_xlim(corners[:, 0].min() - pad, corners[:, 0].max() + pad)
    ax_map.set_ylim(corners[:, 1].min() - pad, corners[:, 1].max() + pad)

    # --- Right panel: seismic section ---
    section = cube.data.sel(iline=inline_val, twt=slice(2300, 3000))
    section.plot.imshow(
        ax=ax_seis,
        x="xline", y="twt",
        add_colorbar=True, interpolation="spline16", robust=True,
        yincrease=False, cmap="RdBu", vmin=-5, vmax=5, alpha=0.8,
    )

    # Horizon overlay
    hrz_il = hrz_mapped.sel(iline=inline_val)
    ax_seis.plot(hrz_il.xline.values, hrz_il.twt.values, color="k", lw=3)
    ax_seis.text(
        hrz_il.xline.values[11], hrz_il.twt.values[0],
        s="Top Hugin", bbox=props, fontsize=6,
    )
    ax_seis.invert_xaxis()
    ax_seis.set_xlabel("Crossline #")
    ax_seis.set_ylabel("TWT (ms)")
    ax_seis.set_title(f"Inline #{inline_val}")

    return fig, axes


def create_animation(cube, hrz_mapped, corners, output_path, fps=10, fmt="gif"):
    """Create a GIF or MP4 animation sweeping through all inlines.

    Parameters
    ----------
    cube, hrz_mapped, corners : as returned by the loading/mapping functions
    output_path : str
        Output file path (without extension — extension is added based on fmt).
    fps : int
        Frames per second.
    fmt : str
        "gif" or "mp4".
    """
    fig = plt.figure(figsize=(20, 8))
    ilines = cube.iline.values

    def update(frame):
        plot_map_and_section(cube, hrz_mapped, corners, ilines[frame], fig=fig)

    anim = FuncAnimation(fig, update, frames=len(ilines), interval=1000 // fps)
    full_path = f"{output_path}.{fmt}"

    if fmt == "gif":
        anim.save(full_path, writer="pillow", fps=fps)
    else:
        anim.save(full_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Saved animation to {full_path}")
