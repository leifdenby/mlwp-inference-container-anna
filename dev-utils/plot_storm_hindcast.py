"""
Figures of an ANNA storm hindcast (`python -m anna_inference` with
`--initial-state-source danra`), compared with DANRA, from its workdir:

- `<prefix>_wind_mslp.png`: 10 m wind speed (0-40 m/s) with mean sea level
  pressure contours at one valid time, 2x2: ERA5 (hourly ARCO-ERA5) and
  DANRA on top, ANNA and the DMI station observations of wind speed (DMI
  open data metObs API, cached in `<prefix>_stations_<valid time>.csv`)
  below
- `<prefix>_timeseries.png`: the minimum mean sea level pressure and the
  maximum 10 m wind speed over Denmark (land points of Denmark, Natural
  Earth 10m) by valid time, DANRA and ANNA, with observed values marked.
  The y-axes are fixed (`--mslp-ylim`, `--wind-ylim`), so that figures of
  other models (e.g. the CLARA hindcast of the same case) can be overlaid
- with `--frames`: `frames/<prefix>_wind_<valid time>.png`, the 2x2 maps of
  every valid time, and `<prefix>_wind.gif` animating them
- with `--single`: `<prefix>_wind_single.gif`, ANNA on its own over its
  whole domain (no title, for a slide), holding `--valid-time`, which is
  also saved as `<prefix>_wind_single_<HH>Z.png`

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
import pandas as pd
import requests
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
# the single-panel animation (--single): ANNA's domain without the outer
# grid points (25 km), for a slide box of ~3.6 x 2.6 in
WHITE = "#ffffff"
SINGLE_TRIM_POINTS = 10
SINGLE_SIZE_PX = (1600, 1150)
SINGLE_FRAME_MS = 800
SINGLE_HOLD_MS = 2500
# GIF palette: colours of the wind colour map evenly over 0-40 m/s (an
# adaptive palette spends them on the common light colours and loses the
# wind maxima), and greys from white to black for lines and text
SINGLE_GIF_CMAP_COLORS = 160
SINGLE_GIF_GREYS = 32
# hourly ERA5 (the 6-hourly WeatherBench2 store of the boundary is derived
# from it); zarr v2, as the environment has zarr < 3
ARCO_ERA5 = (
    "gs://gcp-public-data-arco-era5/ar/"
    "1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
)
# DMI open data, station observations
METOBS_URL = (
    "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
)


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
    transform: ccrs.Projection = DANRA_CRS,
    dims: tuple[str, str] = ("y", "x"),
    title_size: int = 16,
):
    """
    Draw the wind speed, with mean sea level pressure contours.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    da_wind : xr.DataArray
        10 m wind speed (m/s), dims `dims`.
    da_mslp : xr.DataArray or None
        Mean sea level pressure (hPa), dims `dims`; no contours if None.
    title : str
        Panel title; the maximum wind (and minimum pressure) is added.
    transform : ccrs.Projection
        The projection of the fields' coordinates (DANRA's x/y, or
        PlateCarree for ERA5's latitude/longitude).
    dims : tuple[str, str]
        The names of the fields' (y, x) dims, which are also coordinates.
    title_size : int
        Font size of the title.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The wind speed mesh (for the colour bar).
    """
    y_name, x_name = dims
    mesh = ax.pcolormesh(
        da_wind[x_name].values,
        da_wind[y_name].values,
        da_wind.transpose(*dims).values,
        transform=transform,
        shading="auto",
        rasterized=True,
        **WIND_STYLE,
    )
    summary = f"max {float(np.nanmax(da_wind.values)):.0f} m/s"
    if da_mslp is not None:
        da_mslp = da_mslp.transpose(*dims)
        cs = ax.contour(
            da_mslp[x_name].values,
            da_mslp[y_name].values,
            da_mslp.values,
            levels=MSLP_LEVELS,
            colors=INK,
            linewidths=0.8,
            transform=transform,
        )
        ax.clabel(cs, fmt="%d", fontsize=10)
        # the pressure minimum within the map
        i_min = np.unravel_index(np.nanargmin(da_mslp.values), da_mslp.shape)
        ax.plot(
            da_mslp[x_name].values[i_min[1]],
            da_mslp[y_name].values[i_min[0]],
            marker="x",
            color=INK,
            markersize=10,
            markeredgewidth=2,
            transform=transform,
        )
        summary = f"min {float(np.nanmin(da_mslp.values)):.0f} hPa, {summary}"
    ax.set_title(
        f"{title}  ({summary})", color=INK, fontsize=title_size, loc="left"
    )
    return mesh


def load_era5(
    valid_time: np.datetime64, source: str = ARCO_ERA5
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load ERA5's 10 m wind speed and MSLP around the map at one valid time.

    Parameters
    ----------
    valid_time : np.datetime64
        The valid time (any hour with the hourly ARCO-ERA5 store).
    source : str
        The ERA5 zarr store (WeatherBench2/ARCO-ERA5 layout), read
        anonymously from gs://.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The wind speed (m/s) and MSLP (hPa), dims (latitude, longitude).
    """
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    # no dask: the hourly store has ~550,000 time chunks per variable
    ds = xr.open_zarr(source, storage_options=storage_options, chunks=None)
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    ds = ds[
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ]
    ].sel(time=valid_time)
    ds = ds.sel(
        latitude=slice(lat_max + 2, lat_min - 2),
        longitude=slice(lon_min - 3, lon_max + 3),
    ).load()
    da_wind = np.hypot(
        ds["10m_u_component_of_wind"], ds["10m_v_component_of_wind"]
    )
    return da_wind, ds["mean_sea_level_pressure"] / 100


def fetch_station_wind(
    valid_time: np.datetime64, fp_cache: Path | None = None
) -> pd.DataFrame:
    """
    Fetch the DMI station observations of 10 m wind speed at one time.

    From DMI's open data metObs API (which dmiparquet.denby.eu republishes,
    from 2020 only). `wind_speed` is the 10-minute mean wind speed. Stations
    without a location in the API for that time are left out.

    Parameters
    ----------
    valid_time : np.datetime64
        The observation time.
    fp_cache : Path, optional
        A CSV file to read the observations from if it exists, or to save
        them to.

    Returns
    -------
    pd.DataFrame
        Columns `station_id`, `lon`, `lat` and `wind_speed` (m/s).
    """
    if fp_cache is not None and fp_cache.exists():
        return pd.read_csv(fp_cache, dtype={"station_id": str})
    t = f"{str(valid_time)[:19]}Z"
    response = requests.get(
        METOBS_URL,
        params=dict(
            parameterId="wind_speed", datetime=f"{t}/{t}", limit=10000
        ),
        timeout=60,
    )
    response.raise_for_status()
    features = response.json()["features"]
    rows = [
        dict(
            station_id=f["properties"]["stationId"],
            lon=f["geometry"]["coordinates"][0],
            lat=f["geometry"]["coordinates"][1],
            wind_speed=f["properties"]["value"],
        )
        for f in features
        if f["geometry"] is not None
    ]
    df = pd.DataFrame(rows)
    print(
        f"{len(df)} station observations at {t} ({len(features) - len(df)} "
        "without a location left out)"
    )
    if fp_cache is not None:
        df.to_csv(fp_cache, index=False)
    return df


def draw_stations(ax: plt.Axes, df_obs: pd.DataFrame, title: str) -> None:
    """
    Draw station observations of wind speed as coloured dots.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    df_obs : pd.DataFrame
        Columns `lon`, `lat` and `wind_speed` (m/s).
    title : str
        Panel title; the number of stations and the maximum are added.
    """
    df_obs = df_obs.sort_values("wind_speed")
    ax.scatter(
        df_obs.lon,
        df_obs.lat,
        c=df_obs.wind_speed,
        s=70,
        edgecolors=INK,
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=3,
        **WIND_STYLE,
    )
    ax.set_title(
        f"{title}  ({len(df_obs)}, 10-min mean, max "
        f"{df_obs.wind_speed.max():.0f} m/s)",
        color=INK,
        fontsize=14,
        loc="left",
    )


def plot_four_panels(
    ds_anna: xr.Dataset,
    ds_danra: xr.Dataset,
    valid_time: np.datetime64,
    fp: Path,
    case_name: str,
    fp_obs_cache: Path | None = None,
) -> None:
    """
    Plot ERA5 and DANRA (top), ANNA and the station observations (bottom) at
    one valid time (2x2, 16:9).

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
    fp_obs_cache : Path, optional
        CSV cache of the station observations (see `fetch_station_wind`).
    """
    ds_anna_t = crop_to_map(ds_anna.sel(time=valid_time)).load()
    ds_danra_t = crop_to_map(ds_danra.sel(time=valid_time)).load()
    da_wind_era5, da_mslp_era5 = load_era5(valid_time)
    df_obs = fetch_station_wind(valid_time, fp_obs_cache)
    init_time = get_init_time(ds_anna)
    lead_h = int((valid_time - init_time) / np.timedelta64(1, "h"))

    fig = plt.figure(figsize=FIGSIZE, facecolor=SURFACE)
    width, height = 0.46, 0.37
    rects = [
        [0.03, 0.50, width, height],
        [0.51, 0.50, width, height],
        [0.03, 0.08, width, height],
        [0.51, 0.08, width, height],
    ]
    axes = [add_map_axes(fig, rect) for rect in rects]
    # the maps keep their aspect ratio, so close the columns up in the middle
    for ax, anchor in zip(axes, ["E", "W", "E", "W"]):
        ax.set_anchor(anchor)
        ax.apply_aspect()
    ax_era5, ax_danra, ax_anna, ax_obs = axes
    draw_wind_mslp(
        ax_era5,
        da_wind_era5,
        da_mslp_era5,
        "ERA5",
        transform=ccrs.PlateCarree(),
        dims=("latitude", "longitude"),
        title_size=14,
    )
    for ax, (title, ds) in zip(
        [ax_danra, ax_anna],
        [("DANRA", ds_danra_t), (f"ANNA +{lead_h} h", ds_anna_t)],
    ):
        mesh = draw_wind_mslp(
            ax,
            calculate_wind_speed(ds),
            ds.pres_seasurface / 100,
            title,
            title_size=14,
        )
    draw_stations(ax_obs, df_obs, "DMI stations")
    pos_top, pos_bottom = axes[1].get_position(), axes[3].get_position()
    cax = fig.add_axes(
        [
            pos_top.x1 + 0.04,
            pos_bottom.y0,
            0.012,
            pos_top.y1 - pos_bottom.y0,
        ]
    )
    cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cb.set_label(
        "10 m wind speed (m/s); contours: MSLP every 4 hPa",
        color=INK_2,
        fontsize=12,
    )
    cb.ax.tick_params(colors=INK_2, labelsize=11)
    cb.outline.set_edgecolor(GRID)
    fig.text(
        0.03,
        0.975,
        f"{case_name}: valid {str(valid_time)[:13]}Z, ANNA from "
        f"{str(init_time)[:13]}Z ({describe_inputs(ds_anna)})",
        color=INK,
        fontsize=17,
        ha="left",
        va="top",
    )
    fig.savefig(fp, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


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
        # 6-hourly labels (3-hourly ones overlap)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(3, 24, 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H UTC"))
        ax.grid(which="minor", color=GRID, linewidth=0.4)
        ax.tick_params(colors=INK_2, labelsize=13)
        ax.grid(color=GRID, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.legend(fontsize=13, frameon=False, labelcolor=INK)
    fig.text(
        0.06,
        0.95,
        f"{case_name}: ANNA from {str(init_time)[:13]}Z and DANRA\n"
        f"{describe_inputs(ds_anna)}; land points of Denmark",
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
    Plot the 2x2 maps (see `plot_four_panels`) of every valid time, and
    animate them as a GIF.

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
        t_name = str(valid_time)[:13]
        fp = frames_dir / f"{prefix}_wind_{t_name}.png"
        plot_four_panels(
            ds_anna,
            ds_danra,
            valid_time,
            fp,
            case_name,
            fp_obs_cache=frames_dir / f"{prefix}_stations_{t_name}.csv",
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


def create_gif_palette() -> Image.Image:
    """
    Create the GIF palette of the single-panel animation.

    Returns
    -------
    Image.Image
        A palette image: `SINGLE_GIF_CMAP_COLORS` colours of the wind colour
        map, evenly over its range, and `SINGLE_GIF_GREYS` greys from white
        to black.
    """
    cmap = plt.get_cmap(WIND_STYLE["cmap"])
    rgb = [
        tuple(int(round(255 * c)) for c in cmap(f)[:3])
        for f in np.linspace(0, 1, SINGLE_GIF_CMAP_COLORS)
    ]
    rgb += [
        (int(g),) * 3 for g in np.linspace(255, 0, SINGLE_GIF_GREYS).round()
    ]
    flat = [c for colour in rgb for c in colour]
    flat += [0] * (768 - len(flat))
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat)
    return palette


def plot_single_animation(
    ds_anna: xr.Dataset,
    out_dir: Path,
    prefix: str,
    hold_time: np.datetime64,
) -> None:
    """
    Animate ANNA on its own over its whole domain, for a slide (no title).

    One map of the 10 m wind speed with MSLP contours (styled as the
    comparison maps) on the DANRA grid without its outer
    `SINGLE_TRIM_POINTS` points, a large time stamp, and a slim colour bar.
    1600 x 1150 px on white, fonts at least ~4% of the height. Every valid
    time is a frame of `SINGLE_FRAME_MS`, `hold_time` is held for
    `SINGLE_HOLD_MS` and also saved as a PNG.

    Parameters
    ----------
    ds_anna : xr.Dataset
        ANNA single levels (`outputs/single_levels.zarr`).
    out_dir : Path
        Directory for `<prefix>_wind_single.gif` and
        `<prefix>_wind_single_<HH>Z.png`.
    prefix : str
        The file name prefix.
    hold_time : np.datetime64
        The valid time to hold (and save as a PNG).
    """
    n = SINGLE_TRIM_POINTS
    ds = ds_anna.isel(y=slice(n, -n), x=slice(n, -n)).load()
    init_time = get_init_time(ds_anna)
    x, y = ds.x.values, ds.y.values
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    width_px, height_px = SINGLE_SIZE_PX
    palette = create_gif_palette()
    frames, durations = [], []
    for valid_time in ds.time.values:
        ds_t = ds.sel(time=valid_time)
        da_wind = calculate_wind_speed(ds_t)
        da_mslp = ds_t.pres_seasurface / 100
        fig = plt.figure(
            figsize=(width_px / 100, height_px / 100),
            dpi=100,
            facecolor=WHITE,
        )
        # the map fills the height above the colour bar; it is narrower than
        # the figure (DANRA's aspect is 1.34), the sides are white
        ax = fig.add_axes(
            [0.0, 0.2, 1.0, 0.8], projection=DANRA_CRS, facecolor=WHITE
        )
        ax.set_extent(extent, crs=DANRA_CRS)
        ax.set_anchor("N")
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"), edgecolor=INK, linewidth=1.0
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        mesh = ax.pcolormesh(
            x,
            y,
            da_wind.values,
            transform=DANRA_CRS,
            shading="auto",
            rasterized=True,
            **WIND_STYLE,
        )
        cs = ax.contour(
            x,
            y,
            da_mslp.values,
            levels=MSLP_LEVELS,
            colors=INK,
            linewidths=1.2,
            transform=DANRA_CRS,
        )
        ax.clabel(cs, fmt="%d", fontsize=18)
        i_min = np.unravel_index(np.nanargmin(da_mslp.values), da_mslp.shape)
        ax.plot(
            x[i_min[1]],
            y[i_min[0]],
            marker="x",
            color=INK,
            markersize=18,
            markeredgewidth=3.5,
            transform=DANRA_CRS,
        )
        t = pd.Timestamp(valid_time)
        lead_h = int((valid_time - init_time) / np.timedelta64(1, "h"))
        ax.text(
            0.015,
            0.975,
            f"{t.day} {t:%b %Y %H} UTC (+{lead_h} h)",
            transform=ax.transAxes,
            fontsize=36,
            color=INK,
            ha="left",
            va="top",
            bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.8, pad=6),
        )
        ax.apply_aspect()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.1, 0.12, pos.width - 0.2, 0.022])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label("10 m wind speed (m/s)", color=INK, fontsize=34)
        cb.ax.tick_params(colors=INK, labelsize=34)
        cb.outline.set_edgecolor(GRID)
        fig.canvas.draw()
        image = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        hold = valid_time == np.datetime64(hold_time, "ns")
        if hold:
            fp_png = out_dir / f"{prefix}_wind_single_{t:%H}Z.png"
            image.save(fp_png)
            print(f"wrote {fp_png}")
        plt.close(fig)
        # no dithering: its noise makes the GIF much larger
        frames.append(
            image.quantize(palette=palette, dither=Image.Dither.NONE)
        )
        durations.append(SINGLE_HOLD_MS if hold else SINGLE_FRAME_MS)
    fp_gif = out_dir / f"{prefix}_wind_single.gif"
    frames[0].save(
        fp_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {fp_gif} ({fp_gif.stat().st_size / 1e6:.1f} MB)")


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
        help="also plot the 2x2 maps of every valid time, and a GIF",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="also animate ANNA on its own over its domain, for a slide, "
        "holding --valid-time",
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
    t_name = str(valid_time)[:13]
    plot_four_panels(
        ds_anna,
        ds_danra,
        valid_time,
        out_dir / f"{prefix}_wind_mslp.png",
        args.case_name,
        fp_obs_cache=out_dir / f"{prefix}_stations_{t_name}.csv",
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
    if args.single:
        plot_single_animation(ds_anna, out_dir, prefix, valid_time)


if __name__ == "__main__":
    main()
