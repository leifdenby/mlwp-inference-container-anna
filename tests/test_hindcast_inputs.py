"""
The hindcast inputs: DANRA as the interior (`anna_inference.danra_interior`)
and ERA5 as the boundary (`anna_inference.era5_boundary`), on synthetic data
in the layouts of DANRA v0.5.0 and WeatherBench2 ERA5.
"""
import datetime

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from anna_inference import danra_interior, era5_boundary
from anna_inference.convert_output import source_attrs
from anna_inference.create_inference_dataset import boundary_to_valid_time
from anna_inference.regrid_dini import BOUNDARY_LATS, BOUNDARY_LONS, LEVELS

# the 3 December 1999 storm: initial states 02T21Z and 03T00Z, predictions
# 03T03Z .. 04T00Z
ANALYSIS_TIME = datetime.datetime(1999, 12, 2, 21)
DURATION = datetime.timedelta(hours=27)


def _make_wb2_era5(times: pd.DatetimeIndex) -> xr.Dataset:
    """
    Make a lazy dataset in the WeatherBench2 ERA5 layout (global 0.25 deg,
    latitude descending, longitude 0..360), each value the index of its time
    (time-dependent variables) or its longitude (statics).
    """
    lats = np.arange(90.0, -90.25, -0.25)
    lons = np.arange(0.0, 360.0, 0.25)
    levels = [50, 100, 200, 400, 500, 600, 700, 850, 925, 1000]
    i_time = xr.DataArray(np.arange(len(times), dtype="f4"), dims="time")

    def _field(dims: tuple[str, ...]) -> xr.DataArray:
        sizes = dict(
            level=len(levels), latitude=len(lats), longitude=len(lons)
        )
        shape = tuple(sizes[d] for d in dims)
        da_ones = xr.DataArray(
            dask.array.ones(shape, chunks=shape, dtype="f4"), dims=dims
        )
        return i_time * da_ones

    data_vars = {
        v: _field(("latitude", "longitude")) for v in era5_boundary.SL_VARS
    }
    data_vars.update(
        {
            v: _field(("level", "latitude", "longitude"))
            for v in era5_boundary.PL_VARS
        }
    )
    da_lon = xr.DataArray(lats, dims="latitude") * 0 + xr.DataArray(
        lons, dims="longitude"
    )
    data_vars.update({v: da_lon for v in era5_boundary.STATIC_VARS})
    return xr.Dataset(
        data_vars,
        coords=dict(time=times, level=levels, latitude=lats, longitude=lons),
    )


@pytest.fixture(scope="module")
def ds_era5() -> xr.Dataset:
    return _make_wb2_era5(
        pd.date_range("1999-12-01T00:00", "1999-12-05T00:00", freq="6h")
    )


def test_era5_cycle_and_valid_times():
    cycle, valid_times = era5_boundary.get_valid_times(ANALYSIS_TIME, DURATION)
    assert cycle == datetime.datetime(1999, 12, 2, 12)
    # to 6 h after the last boundary time at or before the last prediction
    assert valid_times[0] == cycle
    assert valid_times[-1] == datetime.datetime(1999, 12, 4, 6)
    assert era5_boundary.get_cycle(datetime.datetime(1999, 12, 3, 0)) == (
        datetime.datetime(1999, 12, 3, 0)
    )


def test_era5_as_forecast_contract(ds_era5):
    ds = era5_boundary.select_era5_as_forecast(
        ds_era5, ANALYSIS_TIME, DURATION
    ).compute()
    assert [str(t)[:16] for t in ds.time.values] == ["1999-12-02T12:00"]
    np.testing.assert_array_equal(ds.latitude, BOUNDARY_LATS)
    np.testing.assert_array_equal(ds.longitude, BOUNDARY_LONS)
    assert ds.level.values.tolist() == LEVELS
    assert ds.level.attrs["units"] == "hPa"
    assert ds["geopotential"].dims == (
        "time",
        "prediction_timedelta",
        "level",
        "latitude",
        "longitude",
    )
    # statics without time/level dims, and from 0..360 longitudes
    assert ds["land_sea_mask"].dims == ("latitude", "longitude")
    np.testing.assert_array_equal(
        ds["land_sea_mask"].isel(latitude=0), BOUNDARY_LONS % 360
    )
    # each lead time is the analysis at its valid time
    i_cycle = 6  # 1999-12-02T12 in the synthetic data
    np.testing.assert_array_equal(
        ds["2m_temperature"].isel(time=0, latitude=0, longitude=0),
        i_cycle + np.arange(ds.prediction_timedelta.size),
    )


def test_era5_boundary_to_valid_time(ds_era5):
    ds = era5_boundary.select_era5_as_forecast(
        ds_era5, ANALYSIS_TIME, DURATION
    ).compute()
    ds_valid = boundary_to_valid_time(ds, ANALYSIS_TIME, DURATION)
    assert ds_valid.attrs["boundary_cycle"] == "1999-12-02T12:00:00"
    assert ds_valid.attrs["boundary_times_used"] == (
        "1999-12-02T18:00 .. 1999-12-04T06:00"
    )
    # the used valid times hold ERA5 at those times
    da = ds_valid["2m_temperature"].isel(latitude=0, longitude=0)
    i_valid = {
        f"{t:%Y-%m-%dT%H}": i
        for i, t in enumerate(
            pd.date_range("1999-12-01T00:00", "1999-12-05T00:00", freq="6h")
        )
    }
    for t in ["1999-12-02T18", "1999-12-03T18", "1999-12-04T06"]:
        assert float(da.sel(time=t)) == i_valid[t]


def test_era5_missing_times(ds_era5):
    with pytest.raises(ValueError, match="ERA5 has no valid times"):
        era5_boundary.select_era5_as_forecast(
            ds_era5, datetime.datetime(1999, 12, 4, 0), DURATION
        )


def _make_danra(times: pd.DatetimeIndex) -> tuple[xr.Dataset, xr.Dataset]:
    """Make small datasets in the DANRA v0.5.0 layout."""
    rng = np.random.default_rng(0)
    pressures = [1000, 950, 925, 850, 700, 600, 500, 400, 200, 100]
    coords = dict(
        time=times,
        x=np.arange(4.0),
        y=np.arange(3.0),
        lat=(("y", "x"), rng.normal(size=(3, 4))),
        lon=(("y", "x"), rng.normal(size=(3, 4))),
        danra_projection=0,
    )
    ds_sl = xr.Dataset(
        {
            v: (("time", "y", "x"), rng.normal(size=(len(times), 3, 4)))
            for v in danra_interior.SL_VARS
        },
        coords=coords,
    )
    for v in danra_interior.STATIC_VARS:
        ds_sl[v] = (("y", "x"), rng.normal(size=(3, 4)))
    ds_pl = xr.Dataset(
        {
            v: (
                ("time", "pressure", "y", "x"),
                rng.normal(size=(len(times), len(pressures), 3, 4)),
            )
            for v in danra_interior.PL_VARS + ["extra"]
        },
        coords=dict(coords, pressure=pressures),
    )
    return ds_sl, ds_pl


def test_danra_interior():
    ds_sl, ds_pl = _make_danra(
        pd.date_range("1999-12-01T00:00", "1999-12-05T00:00", freq="3h")
    )
    valid_times = danra_interior.get_valid_times(ANALYSIS_TIME, DURATION)
    # T+0 .. T+27h and the two extra steps
    assert valid_times.size == 27 // 3 + 1 + 2
    assert str(valid_times[0])[:16] == "1999-12-02T21:00"
    assert str(valid_times[-1])[:16] == "1999-12-04T06:00"
    ds_sl_out, ds_pl_out = danra_interior.select_danra_interior(
        ds_sl, ds_pl, valid_times
    )
    assert set(ds_sl_out.data_vars) == set(
        danra_interior.SL_VARS + danra_interior.STATIC_VARS
    )
    assert set(ds_pl_out.data_vars) == set(danra_interior.PL_VARS)
    assert ds_pl_out.pressure.values.tolist() == LEVELS
    assert "danra_projection" not in ds_sl_out.coords
    # copied, not changed
    xr.testing.assert_equal(
        ds_sl_out["t2m"],
        ds_sl["t2m"].sel(time=valid_times).drop_vars("danra_projection"),
    )


def test_danra_interior_missing_times():
    ds_sl, ds_pl = _make_danra(
        pd.date_range("1999-12-03T00:00", "1999-12-05T00:00", freq="3h")
    )
    valid_times = danra_interior.get_valid_times(ANALYSIS_TIME, DURATION)
    with pytest.raises(ValueError, match="DANRA has no valid times"):
        danra_interior.select_danra_interior(ds_sl, ds_pl, valid_times)


@pytest.mark.parametrize(
    "initial, boundary, comment",
    [
        ("dini", "dini", None),
        ("dini", "ifs", None),
        (
            "danra",
            "era5",
            "DANRA/ERA5 hindcast: initial states from DANRA, boundary from "
            "ERA5 (the inputs ANNA was trained with)",
        ),
        (
            "danra",
            "ifs",
            "DANRA/IFS hindcast: initial states from DANRA, boundary from IFS",
        ),
    ],
)
def test_source_attrs(initial, boundary, comment):
    attrs = source_attrs(initial, boundary)
    assert attrs["initial_state_source"] == initial
    assert attrs["boundary_source"] == boundary
    assert attrs.get("comment") == comment
