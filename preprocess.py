import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from satpy import Scene
from tqdm import tqdm

import config


def existing_times():
    if not config.ZARR_PATH.exists():
        return set()
    with xr.open_zarr(config.ZARR_PATH) as ds:
        return set(pd.to_datetime(ds.time.values))


def process_file(path):
    scn = Scene(filenames=[str(path)], reader="seviri_l1b_native")
    scn.load(config.CHANNELS)
    local = scn.resample(config.AREA, resampler="nearest", cache_dir=str(config.CACHE_DIR))
    stack = np.stack([local[ch].values.astype(np.float32) for ch in config.CHANNELS])
    x, y = config.AREA.get_proj_coords()
    return pd.Timestamp(scn.start_time), stack, x[0], y[:, 0]


def append_frames(times, stack, x, y, path=config.ZARR_PATH, source=config.COLLECTION, dtype="float32"):
    """Append frames (T, channel, y, x) to a zarr store, creating it on first use."""
    ds = xr.Dataset(
        {"data": (("time", "channel", "y", "x"), stack)},
        coords={"time": list(times), "channel": config.CHANNELS, "y": y, "x": x},
    )
    if not path.exists():
        ds.attrs.update(area=config.AREA.crs.to_wkt(), collection=source,
                        units="K (IR/WV), % (VIS)")
        ds = ds.chunk({"time": 1, "channel": -1, "y": -1, "x": -1})
        # fixed time units: otherwise xarray picks "days since <first frame>" and appended
        # 5-minute steps get silently mis-encoded
        ds.to_zarr(path, mode="w",
                   encoding={"time": {"units": "seconds since 2000-01-01", "dtype": "int64"},
                             "data": {"dtype": dtype}})
    else:
        # static coords are already stored; only the time axis grows
        ds.drop_vars(["channel", "y", "x"]).chunk({"time": 1}).to_zarr(path, append_dim="time")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete-raw", action="store_true", help="delete .nat files once stored")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaNs outside the scan area
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(config.RAW_DIR.glob("*.nat"))  # filenames sort chronologically
    done = existing_times()
    print(f"{len(files)} raw files, {len(done)} frames already in {config.ZARR_PATH.name}")

    failed = []
    for path in tqdm(files, unit="file"):
        try:
            t, stack, x, y = process_file(path)
        except Exception as exc:
            tqdm.write(f"  failed {path.name}: {exc}")
            failed.append(path.name)
            continue
        if t in done:
            tqdm.write(f"  {t} already stored, skipping")
        else:
            append_frames([t], stack[None], x, y)
            done.add(t)
        if args.delete_raw:
            path.unlink()

    if done:
        times = pd.DatetimeIndex(sorted(done))
        steps = np.diff(times)
        # anything longer than 1.5x the usual cadence (5 min RSS / 15 min full disk) is a gap
        gaps = times[1:][steps > 1.5 * np.median(steps)] if len(steps) else []
        print(f"Stored {len(times)} frames, {times[0]} .. {times[-1]}")
        if len(gaps):
            print(f"Warning: {len(gaps)} gaps in the time series, e.g. before {list(gaps[:5])}")
    if failed:
        print(f"{len(failed)} files failed: {failed[:5]}")


if __name__ == "__main__":
    main()
