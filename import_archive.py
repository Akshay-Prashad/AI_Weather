"""Import historical SEVIRI Rapid Scan data from Open Climate Fix's public archive on Google Cloud.

The archive (gs://public-datasets-eumetsat-solar-forecasting, 2008-2022, 5-min, no login)
stores values scaled to 0..1; they are converted back to kelvin, regridded onto the same
Ireland grid as the live data, and appended to config.ARCHIVE_ZARR_PATH.

Example:
    python import_archive.py --start 2021-01-01 --end 2022-01-01 --every 6   # every 6th day of 2021
"""
import argparse
import warnings
from datetime import date, timedelta

import numpy as np
import ocf_blosc2  # noqa: F401  registers the blosc2 codec the archive is compressed with
import pandas as pd
import xarray as xr
from pyresample.area_config import load_area_from_string
from tqdm import tqdm

import config
from preprocess import append_frames

ARCHIVE_URL = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year}_nonhrv.zarr"

# satip's 0..1 scaling constants (satip/constants.py), in the archive's channel order
SCALE_CHANNELS = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                  "VIS006", "VIS008", "WV_062", "WV_073"]
SCALE_MINS = [-2.5118103, -64.83977, 63.404694, 2.844452, 199.10002, -17.254883, -26.29155,
              -1.1009827, -2.4184198, 199.57048, 198.95093]
SCALE_MAXS = [69.60857, 339.15588, 340.26526, 317.86752, 313.2767, 315.99194, 274.82297,
              93.786545, 101.34922, 249.91806, 286.96323]


class Archive:
    def __init__(self, year):
        ds = xr.open_zarr(ARCHIVE_URL.format(year=year), storage_options={"token": "anon"})
        full_area = load_area_from_string(ds.data.attrs["area"])
        xs, ys = full_area.get_area_slices(config.AREA)   # note: (x, y) order
        self.data = ds.data.isel(y_geostationary=ys, x_geostationary=xs).sel(variable=config.CHANNELS)
        # nearest-neighbour lookup from every target pixel to a source pixel, computed once
        src_area = full_area[ys, xs]
        lons, lats = config.AREA.get_lonlats()
        cols, rows = src_area.get_array_indices_from_lonlat(lons, lats)
        self.outside = np.ma.getmaskarray(cols) | np.ma.getmaskarray(rows)
        self.rows, self.cols = np.ma.filled(rows, 0), np.ma.filled(cols, 0)
        idx = [SCALE_CHANNELS.index(c) for c in config.CHANNELS]
        self.mins = np.array(SCALE_MINS, np.float32)[idx]
        self.range = np.array(SCALE_MAXS, np.float32)[idx] - self.mins

    def day(self, d):
        """All frames of one UTC day on the target grid: (times, stack[T, channel, y, x] in K)."""
        start = pd.Timestamp(d)
        chunk = self.data.sel(time=slice(start, start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
        if chunk.sizes["time"] == 0:
            return None, None
        raw = chunk.values.astype(np.float32)                     # (T, y_geos, x_geos, channel), 0..1
        kelvin = self.mins + raw * self.range
        out = kelvin[:, self.rows, self.cols, :]                  # (T, 384, 384, channel)
        out[:, self.outside, :] = np.nan
        # archive timestamps mark the end of each 5-min scan; the live data uses scan start
        times = pd.DatetimeIndex(chunk.time.values) - pd.Timedelta(minutes=5)
        return times, out.transpose(0, 3, 1, 2)


def existing_days():
    if not config.ARCHIVE_ZARR_PATH.exists():
        return set()
    with xr.open_zarr(config.ARCHIVE_ZARR_PATH) as ds:
        # key on the end-of-scan day so a day's first frame (23:55 of the previous day) counts too
        return set((pd.DatetimeIndex(ds.time.values) + pd.Timedelta(minutes=5)).date)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="exclusive")
    parser.add_argument("--every", type=int, default=1, help="take every Nth day (spreads data over seasons)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    days = [args.start + timedelta(days=i) for i in range(0, (args.end - args.start).days, args.every)]
    done = existing_days()
    todo = [d for d in days if d not in done]
    print(f"{len(days)} days requested, {len(days) - len(todo)} already imported, {len(todo)} to fetch")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x, y = config.AREA.get_proj_coords()
    archives = {}
    for d in tqdm(todo, unit="day"):
        if d.year not in archives:
            archives[d.year] = Archive(d.year)
        try:
            times, stack = archives[d.year].day(d)
        except Exception as exc:   # transient network errors: skip, a rerun picks the day up
            tqdm.write(f"  {d}: failed ({exc})")
            continue
        if times is None:
            tqdm.write(f"  {d}: no data in archive")
            continue
        append_frames(times, stack.astype(np.float16), x[0], y[:, 0], path=config.ARCHIVE_ZARR_PATH,
                      source=ARCHIVE_URL.format(year=d.year), dtype="float16")

    with xr.open_zarr(config.ARCHIVE_ZARR_PATH) as ds:
        print(f"{config.ARCHIVE_ZARR_PATH.name}: {ds.sizes['time']} frames, "
              f"{str(ds.time.values[0])[:16]} .. {str(ds.time.values[-1])[:16]}")


if __name__ == "__main__":
    main()
