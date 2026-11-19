"""
Take ANNA's boundary from ERA5 (WeatherBench2), the boundary ANNA was trained
with, for hindcasts of past cases (`--boundary-source era5`).

The ERA5 analyses are laid out as a forecast in the IFS boundary contract
(`anna_inference/boundary/ifs_7deg_model1_config.yaml`): one 00/12 UTC cycle,
the latest at or before the analysis time, with each lead time the ERA5
analysis at its valid time. `create_inference_dataset` turns that back into
the valid-time layout of training, so the model sees ERA5 at the same valid
times (00/06/12/18 UTC) as in training. The statics (`land_sea_mask`,
`geopotential_at_surface`) are ERA5's, as in training.

Written as `{output}/boundary.zarr`, the file `regrid_dini.py` writes for a
DINI boundary.

Usage:
    python -m anna_inference.era5_boundary \\
        --analysis-time 1999-12-02T21:00 --forecast-duration PT27H \\
        --output inference_workdir/inputs
"""
import argparse
import datetime
import time
from pathlib import Path

import dask
import dask.diagnostics
import isodate
import numpy as np
import xarray as xr
from loguru import logger

from anna_inference.create_inference_dataset import (
    BOUNDARY_STEP,
    boundary_times,
)
from anna_inference.regrid_dini import BOUNDARY_LATS, BOUNDARY_LONS, LEVELS

# as in the training boundary config (era_7deg_model1_config.yaml in the
# artifact) and artifact/era5_boundary_stats.py
WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SL_VARS = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
PL_VARS = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]
CYCLE_STEP = datetime.timedelta(hours=12)


def get_cycle(analysis_time: datetime.datetime) -> datetime.datetime:
    """
    Get the 00/12 UTC cycle the ERA5 boundary is laid out as.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).

    Returns
    -------
    datetime.datetime
        The latest 00/12 UTC time at or before `analysis_time`, the cycle
        `create_inference_dataset.boundary_to_valid_time` picks.
    """
    midnight = datetime.datetime.combine(analysis_time.date(), datetime.time())
    return analysis_time - (analysis_time - midnight) % CYCLE_STEP


def get_valid_times(
    analysis_time: datetime.datetime, forecast_duration: datetime.timedelta
) -> tuple[datetime.datetime, list[datetime.datetime]]:
    """
    Get the cycle and the valid times of the ERA5 boundary of a forecast.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).
    forecast_duration : datetime.timedelta
        The forecast duration from `analysis_time`.

    Returns
    -------
    tuple[datetime.datetime, list[datetime.datetime]]
        The cycle (see `get_cycle`), and the valid times from it every 6 h
        to the last boundary time the forecast uses (see
        `create_inference_dataset.boundary_times`).
    """
    cycle = get_cycle(analysis_time)
    used, _ = boundary_times(analysis_time, forecast_duration)
    n = int((used[-1] - cycle) / BOUNDARY_STEP)
    return cycle, [cycle + k * BOUNDARY_STEP for k in range(n + 1)]


def select_era5_as_forecast(
    ds_era5: xr.Dataset,
    analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
) -> xr.Dataset:
    """
    Select ERA5 for a forecast, laid out as a forecast in the IFS contract.

    Parameters
    ----------
    ds_era5 : xr.Dataset
        ERA5 in WeatherBench2 layout: dims `time`, `level`, `latitude` and
        `longitude` (0..360), with the statics on (`latitude`, `longitude`).
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).
    forecast_duration : datetime.timedelta
        The forecast duration from `analysis_time`.

    Returns
    -------
    xr.Dataset
        The contract layout (lazy): dims `time` (the one cycle),
        `prediction_timedelta`, `level`, `latitude` (72..40) and `longitude`
        (-27..39.5).

    Raises
    ------
    ValueError
        If ERA5 doesn't have all the valid times needed.
    """
    cycle, valid_times = get_valid_times(analysis_time, forecast_duration)
    valid_times_ns = np.array(valid_times, dtype="datetime64[ns]")
    available = ds_era5.time.values.astype("datetime64[ns]")
    missing = valid_times_ns[~np.isin(valid_times_ns, available)]
    if missing.size:
        raise ValueError(
            f"ERA5 has no valid times {[str(t)[:16] for t in missing]}, "
            f"needed for a {forecast_duration} forecast from {analysis_time}"
        )
    # the time range first: label selection on the full WB2 store builds a
    # dask graph over all its time chunks (see era5_boundary_stats.py)
    ds = ds_era5[SL_VARS + PL_VARS].sel(time=valid_times_ns)
    ds = xr.merge([ds, ds_era5[STATIC_VARS]])
    ds = ds.sel(
        level=LEVELS,
        latitude=BOUNDARY_LATS,
        longitude=BOUNDARY_LONS % 360,
    ).assign_coords(longitude=BOUNDARY_LONS)
    leads = valid_times_ns - np.datetime64(cycle, "ns")
    ds = ds.rename(time="prediction_timedelta").assign_coords(
        prediction_timedelta=leads
    )
    for v in SL_VARS + PL_VARS:
        ds[v] = ds[v].expand_dims(time=[np.datetime64(cycle, "ns")])
    ds["level"].attrs = {"units": "hPa"}
    for v in ds.variables:
        ds[v].encoding = {}
    logger.info(
        f"ERA5 boundary as a forecast from {cycle:%Y-%m-%dT%H:%M}: valid "
        f"times {valid_times[0]:%Y-%m-%dT%H:%M} .. "
        f"{valid_times[-1]:%Y-%m-%dT%H:%M}"
    )
    return ds


def extract_era5_boundary(
    analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    output: Path,
    source: str = WB2_ERA5,
) -> None:
    """
    Write the ERA5 boundary of a forecast as `{output}/boundary.zarr`.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).
    forecast_duration : datetime.timedelta
        The forecast duration from `analysis_time`.
    output : Path
        Directory for `boundary.zarr`.
    source : str
        The ERA5 zarr store (WeatherBench2 layout), read anonymously from
        gs://.
    """
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    ds_era5 = xr.open_zarr(source, storage_options=storage_options)
    ds = select_era5_as_forecast(ds_era5, analysis_time, forecast_duration)
    ds.attrs = dict(
        source=source,
        description="ERA5 (WeatherBench2) laid out as a forecast in the IFS "
        "boundary contract, the boundary ANNA was trained with "
        "(era5_boundary.py)",
    )
    logger.info("reading ERA5")
    start = time.perf_counter()
    with dask.diagnostics.ProgressBar(dt=10):
        (ds,) = dask.compute(ds)
    logger.info(f"read in {time.perf_counter() - start:.0f} s")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(output / "boundary.zarr", mode="w", consolidated=True)
    logger.info(f"wrote {output / 'boundary.zarr'}")


def _parse_analysis_time(s: str) -> datetime.datetime:
    t = isodate.parse_datetime(s)
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=_parse_analysis_time,
        help="time of the first initial state, ISO8601 (UTC if no timezone)",
    )
    parser.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source",
        default=WB2_ERA5,
        help="ERA5 zarr store (default: %(default)s)",
    )
    args = parser.parse_args()
    extract_era5_boundary(
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        output=args.output,
        source=args.source,
    )


if __name__ == "__main__":
    main()
