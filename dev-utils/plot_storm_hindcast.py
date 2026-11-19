"""
Figures of an ANNA storm hindcast (`python -m anna_inference` with
`--initial-state-source danra`), compared with DANRA, from its workdir:

- `<prefix>_wind_mslp.png`: 10 m wind speed (0-40 m/s) with mean sea level
  pressure contours at one valid time, DANRA next to ANNA
- `<prefix>_timeseries.png`: the minimum mean sea level pressure and the
  maximum 10 m wind speed over Denmark (land points of Denmark, Natural
  Earth 10m) by valid time, DANRA and ANNA, with observed values marked.
  The y-axes are fixed (`--mslp-ylim`, `--wind-ylim`), so that figures of
  other models (e.g. the CLARA hindcast of the same case) can be overlaid
- with `--frames`: `frames/<prefix>_wind_<valid time>.png`, the wind speed
  maps of every valid time, and `<prefix>_wind.gif` animating them

DANRA is taken from the workdir (`inputs/interior_single_levels.zarr`, which
covers the initial states and all valid times). Winds are relative to the
DANRA grid in both DANRA and ANNA; the speed doesn't depend on that (wind
vectors would need rotating to east/north first).

Ported from the CLARA container's `dev-utils/plot_storm_hindcast.py` (same
map, colour scale and contours), for a deterministic forecast.

Usage (cartopy downloads Natural Earth data on first use), e.g. for the 3
December 1999 storm:

    uv run python dev-utils/plot_storm_hindcast.py \\
        --workdir ~/anna-hindcasts/1999-12-03_storm/init-1999-12-03T00 \\
        --valid-time 1999-12-03T18:00 \\
        --observed-min-mslp 952 --observed-max-wind 41.2 --frames
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from PIL import Image

# categorical slot 2 (as CLARA's member mean), text and surface
SERIES_ANNA = "#eb6834"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d6d5d0"
# the DANRA projection (extra.projection in the interior datastore config)
DANRA_CRS = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=(56.7, 56.7),
    globe=ccrs.Globe(semimajor_axis=6367470.0, semiminor_axis=6367470.0),
)
# Denmark-centred map extent (lon_min, lon_max, lat_min, lat_max)
MAP_EXTENT = [3.5, 17.5, 53.3, 59.0]
WIND_STYLE = dict(cmap="YlGnBu", vmin=0.0, vmax=40.0)
MSLP_LEVELS = np.arange(940, 1041, 4)
MSLP_YLIM = (940.0, 1010.0)
WIND_YLIM = (0.0, 50.0)
FIGSIZE = (16, 9)
DPI = 150
GIF_FRAME_MS = 700


def calculate_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate the 10 m wind speed.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with `u10m` and `v10m` (m/s).

    Returns
    -------
    xr.DataArray
        The wind speed (m/s).
    """
    return np.hypot(ds.u10m, ds.v10m)


def create_denmark_mask(ds_grid: xr.Dataset) -> xr.DataArray:
    """
    Create a mask of the grid points in Denmark (Natural Earth 10m).

    Parameters
    ----------
    ds_grid : xr.Dataset
        Dataset with 2D `lat` and `lon` coordinates.

    Returns
    -------
    xr.DataArray
        True for the points inside Denmark's land area.
    """
    reader = shpreader.Reader(
        shpreader.natural_earth("10m", "cultural", "admin_0_countries")
    )
    denmark = shapely.union_all(
        [
            r.geometry
            for r in reader.records()
            if r.attributes["ADMIN"] == "Denmark"
        ]
    )
    inside = shapely.contains_xy(
        denmark, ds_grid.lon.values, ds_grid.lat.values
    )
    return xr.DataArray(inside, dims=ds_grid.lat.dims)


def add_map_axes(fig: plt.Figure, rect: list[float]) -> plt.Axes:
    """
    Add a Denmark-centred map axes in the DANRA projection.

    Parameters
    ----------
    fig : plt.Figure
        The figure.
    rect : list[float]
        The axes position [left, bottom, width, height].

    Returns
    -------
    plt.Axes
        The map axes.
    """
    ax = fig.add_axes(rect, projection=DANRA_CRS, facecolor=SURFACE)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=INK, linewidth=0.6
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    return ax


def draw_wind_mslp(
    ax: plt.Axes,
    da_wind: xr.DataArray,
    da_mslp: xr.DataArray | None,
    title: str,
):
    """
    Draw the wind speed, with mean sea level pressure contours.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    da_wind : xr.DataArray
        10 m wind speed (m/s), dims (y, x).
    da_mslp : xr.DataArray or None
        Mean sea level pressure (hPa), dims (y, x); no contours if None.
    title : str
        Panel title; the maximum wind (and minimum pressure) is added.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The wind speed mesh (for the colour bar).
    """
    mesh = ax.pcolormesh(
        da_wind.x.values,
        da_wind.y.values,
        da_wind.values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **WIND_STYLE,
    )
    summary = f"max {float(np.nanmax(da_wind.values)):.0f} m/s"
    if da_mslp is not None:
        cs = ax.contour(
            da_mslp.x.values,
            da_mslp.y.values,
            da_mslp.values,
            levels=MSLP_LEVELS,
            colors=INK,
            linewidths=0.8,
            transform=DANRA_CRS,
        )
        ax.clabel(cs, fmt="%d", fontsize=10)
        # the pressure minimum within the map
        i_min = np.unravel_index(np.nanargmin(da_mslp.values), da_mslp.shape)
        ax.plot(
            da_mslp.x.values[i_min[1]],
            da_mslp.y.values[i_min[0]],
            marker="x",
            color=INK,
            markersize=10,
            markeredgewidth=2,
            transform=DANRA_CRS,
        )
        summary = f"min {float(np.nanmin(da_mslp.values)):.0f} hPa, {summary}"
    ax.set_title(f"{title}  ({summary})", color=INK, fontsize=16, loc="left")
    return mesh


def crop_to_map(ds: xr.Dataset) -> xr.Dataset:
    """
    Crop a DANRA-grid dataset to the points within the map extent.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 2D `lat`/`lon` coordinates.

    Returns
    -------
    xr.Dataset
        The dataset on the smallest x/y box covering the map extent.
    """
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    inside = (
        (ds.lon >= lon_min - 2)
        & (ds.lon <= lon_max + 2)
        & (ds.lat >= lat_min - 1)
        & (ds.lat <= lat_max + 1)
    ).values
    j, i = np.nonzero(inside)
    return ds.isel(
        y=slice(j.min(), j.max() + 1), x=slice(i.min(), i.max() + 1)
    )


def describe_inputs(ds_anna: xr.Dataset) -> str:
    """
    Describe an ANNA forecast's input sources briefly.

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA output, with the `initial_state_source` and `boundary_source`
        attributes (DINI for both if absent, as before they were recorded).

    Returns
    -------
    str
        E.g. "DANRA initial states, ERA5 boundary".
    """
    initial = ds_anna.attrs.get("initial_state_source", "dini").upper()
    boundary = ds_anna.attrs.get("boundary_source", "dini").upper()
    return f"{initial} initial states, {boundary} boundary"


def get_init_time(ds_anna: xr.Dataset) -> np.datetime64:
    """
    Get the forecast start: the second initial state, analysis time + 3 h.

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA output, with the `analysis_time` attribute.

    Returns
    -------
    np.datetime64
        The forecast start time.
    """
    return np.datetime64(
        ds_anna.attrs["analysis_time"], "ns"
    ) + np.timedelta64(3, "h")


def plot_side_by_side(
    ds_anna: xr.Dataset,
    ds_danra: xr.Dataset,
    valid_time: np.datetime64,
    fp: Path,
    case_name: str,
    with_mslp: bool = True,
) -> None:
    """
    Plot DANRA next to ANNA at one valid time (16:9).

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA single levels (`outputs/single_levels.zarr`).
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    valid_time : np.datetime64
        The valid time.
    fp : Path
        The PNG to write.
    case_name : str
        The case, for the title.
    with_mslp : bool
        Whether to draw the MSLP contours.
    """
    ds_anna_t = crop_to_map(ds_anna.sel(time=valid_time)).load()
    ds_danra_t = crop_to_map(ds_danra.sel(time=valid_time)).load()
    init_time = get_init_time(ds_anna)
    lead_h = int((valid_time - init_time) / np.timedelta64(1, "h"))

    panels = [
        ("DANRA", ds_danra_t),
        (f"ANNA +{lead_h} h", ds_anna_t),
    ]
    fig = plt.figure(figsize=FIGSIZE, facecolor=SURFACE)
    for k, (title, ds) in enumerate(panels):
        ax = add_map_axes(fig, [0.02 + k * 0.49, 0.15, 0.47, 0.72])
        mesh = draw_wind_mslp(
            ax,
            calculate_wind_speed(ds),
            ds.pres_seasurface / 100 if with_mslp else None,
            title,
        )
    cb = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.30, 0.075, 0.40, 0.022]),
        orientation="horizontal",
    )
    cb.set_label(
        "10 m wind speed (m/s)"
        + ("; contours: MSLP every 4 hPa" if with_mslp else ""),
        color=INK_2,
        fontsize=13,
    )
    cb.ax.tick_params(colors=INK_2, labelsize=12)
    cb.outline.set_edgecolor(GRID)
    fig.text(
        0.02,
        0.97,
        f"{case_name}: valid {str(valid_time)[:13]}Z, ANNA from "
        f"{str(init_time)[:13]}Z ({describe_inputs(ds_anna)})",
        color=INK,
        fontsize=18,
        ha="left",
        va="top",
    )
    fig.savefig(fp, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


def calculate_denmark_extremes(
    ds: xr.Dataset, da_mask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the minimum MSLP and maximum 10 m wind speed over Denmark.

    Parameters
    ----------
    ds : xr.Dataset
        Single levels with `pres_seasurface`, `u10m`, `v10m`.
    da_mask : xr.DataArray
        True for the points in Denmark, dims (y, x).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Minimum MSLP (hPa) and maximum wind speed (m/s), over (y, x).
    """
    da_mslp = (ds.pres_seasurface / 100).where(da_mask)
    da_wind = calculate_wind_speed(ds).where(da_mask)
    return da_mslp.min(("y", "x")).load(), da_wind.max(("y", "x")).load()


def plot_timeseries(
    ds_anna: xr.Dataset,
    ds_danra: xr.Dataset,
    da_mask: xr.DataArray,
    fp: Path,
    case_name: str,
    mslp_ylim: tuple[float, float] = MSLP_YLIM,
    wind_ylim: tuple[float, float] = WIND_YLIM,
    observed_min_mslp: float | None = None,
    observed_max_wind: float | None = None,
    observed_time: np.datetime64 | None = None,
) -> None:
    """
    Plot the minimum MSLP and maximum wind over Denmark by valid time (16:9).

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA single levels.
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    da_mask : xr.DataArray
        True for the points in Denmark.
    fp : Path
        The PNG to write.
    case_name : str
        The case, for the title.
    mslp_ylim, wind_ylim : tuple[float, float]
        The y-axis ranges (hPa, m/s).
    observed_min_mslp : float, optional
        Observed minimum pressure (hPa) to mark.
    observed_max_wind : float, optional
        Observed maximum mean wind (m/s) to mark.
    observed_time : np.datetime64, optional
        The time of the observed values.
    """
    da_mslp, da_wind = calculate_denmark_extremes(ds_anna, da_mask)
    # DANRA up to the last forecast time (the inputs have extra steps)
    da_mslp_danra, da_wind_danra = calculate_denmark_extremes(
        ds_danra.sel(time=slice(None, ds_anna.time.values[-1])), da_mask
    )
    init_time = get_init_time(ds_anna)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE)
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.83, bottom=0.12, wspace=0.18
    )
    panels = [
        (
            axes[0],
            da_mslp,
            da_mslp_danra,
            "Minimum MSLP over Denmark (hPa)",
            observed_min_mslp,
            mslp_ylim,
        ),
        (
            axes[1],
            da_wind,
            da_wind_danra,
            "Maximum 10 m wind speed over Denmark (m/s)",
            observed_max_wind,
            wind_ylim,
        ),
    ]
    for ax, da_anna, da_danra, title, observed, ylim in panels:
        ax.set_facecolor(SURFACE)
        ax.plot(
            da_danra.time.values,
            da_danra.values,
            color=INK,
            linewidth=2.5,
            label="DANRA",
        )
        ax.plot(
            da_anna.time.values,
            da_anna.values,
            color=SERIES_ANNA,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="ANNA",
        )
        if observed is not None and observed_time is not None:
            ax.plot(
                observed_time,
                observed,
                marker="*",
                markersize=18,
                color=INK,
                markerfacecolor="white",
                linestyle="none",
                label=f"observed ({observed:g})",
            )
        ax.axvline(init_time, color=INK_2, linewidth=1.0, linestyle="--")
        ax.text(
            init_time,
            0.02,
            " forecast start",
            transform=ax.get_xaxis_transform(),
            color=INK_2,
            fontsize=12,
        )
        ax.set_ylim(*ylim)
        ax.set_title(title, color=INK, fontsize=16, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H UTC"))
        ax.tick_params(colors=INK_2, labelsize=13)
        ax.grid(color=GRID, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.legend(fontsize=13, frameon=False, labelcolor=INK)
    fig.text(
        0.06,
        0.95,
        f"{case_name}: ANNA from {str(init_time)[:13]}Z "
        f"({describe_inputs(ds_anna)}) and DANRA, land points of Denmark",
        color=INK,
        fontsize=17,
        ha="left",
        va="top",
    )
    fig.savefig(fp, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


def plot_frames(
    ds_anna: xr.Dataset,
    ds_danra: xr.Dataset,
    out_dir: Path,
    prefix: str,
    case_name: str,
) -> None:
    """
    Plot the wind speed maps of every valid time, and animate them as a GIF.

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA single levels.
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    out_dir : Path
        Directory for `frames/` and the GIF.
    prefix : str
        The file name prefix.
    case_name : str
        The case, for the titles.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = []
    for valid_time in ds_anna.time.values:
        fp = frames_dir / (
            f"{prefix}_wind_{str(valid_time)[:13].replace(':', '')}.png"
        )
        plot_side_by_side(
            ds_anna, ds_danra, valid_time, fp, case_name, with_mslp=False
        )
        fps.append(fp)
    images = [
        Image.open(fp).convert("P", palette=Image.ADAPTIVE) for fp in fps
    ]
    fp_gif = out_dir / f"{prefix}_wind.gif"
    images[0].save(
        fp_gif,
        save_all=True,
        append_images=images[1:],
        duration=GIF_FRAME_MS,
        loop=0,
    )
    print(f"wrote {fp_gif}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--valid-time", required=True, type=np.datetime64)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="directory for the figures (default: <workdir>/figures)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="file name prefix (default: anna_hindcast_<forecast start>)",
    )
    parser.add_argument(
        "--case-name",
        default="3 December 1999 storm",
        help="the case, for the titles (default: %(default)s)",
    )
    parser.add_argument("--observed-min-mslp", type=float)
    parser.add_argument("--observed-max-wind", type=float)
    parser.add_argument(
        "--observed-time",
        type=np.datetime64,
        help="time of the observed values (default: --valid-time)",
    )
    parser.add_argument(
        "--mslp-ylim",
        type=float,
        nargs=2,
        default=MSLP_YLIM,
        help="time series MSLP axis range, hPa (default: %(default)s)",
    )
    parser.add_argument(
        "--wind-ylim",
        type=float,
        nargs=2,
        default=WIND_YLIM,
        help="time series wind axis range, m/s (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="also plot the wind maps of every valid time, and a GIF",
    )
    args = parser.parse_args()

    workdir = args.workdir.expanduser()
    ds_anna = xr.open_zarr(workdir / "outputs" / "single_levels.zarr")
    ds_danra = xr.open_zarr(workdir / "inputs" / "interior_single_levels.zarr")
    valid_time = np.datetime64(args.valid_time, "ns")
    init_time = get_init_time(ds_anna)
    prefix = args.prefix or (
        f"anna_hindcast_{str(init_time)[:13].replace(':', '')}"
    )
    out_dir = args.out_dir or workdir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_side_by_side(
        ds_anna,
        ds_danra,
        valid_time,
        out_dir / f"{prefix}_wind_mslp.png",
        args.case_name,
    )
    plot_timeseries(
        ds_anna,
        ds_danra,
        create_denmark_mask(ds_danra),
        out_dir / f"{prefix}_timeseries.png",
        args.case_name,
        mslp_ylim=tuple(args.mslp_ylim),
        wind_ylim=tuple(args.wind_ylim),
        observed_min_mslp=args.observed_min_mslp,
        observed_max_wind=args.observed_max_wind,
        observed_time=(
            np.datetime64(args.observed_time or args.valid_time, "ns")
        ),
    )
    if args.frames:
        plot_frames(ds_anna, ds_danra, out_dir, prefix, args.case_name)


if __name__ == "__main__":
    main()
