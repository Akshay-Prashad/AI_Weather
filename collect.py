
import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import xarray as xr

import config


def last_stored_time():
    if not config.ZARR_PATH.exists():
        return None
    with xr.open_zarr(config.ZARR_PATH) as ds:
        return pd.Timestamp(ds.time.values[-1]).to_pydatetime()


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=config.ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=15, help="minutes between runs")
    args = parser.parse_args()

    while True:
        now = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        # start just after the last stored scan so nothing is missed; a refetched scan is
        # harmless (preprocess skips it). Older than ~55 min is outside the NRT licence.
        last = last_stored_time()
        start = now - timedelta(minutes=55)
        if last is not None:
            start = max(start, last + timedelta(minutes=1))
        print(f"[{now:%Y-%m-%d %H:%M}] fetching {start:%H:%M} .. {now:%H:%M} UTC", flush=True)
        run("download.py", "--start", start.isoformat(" "), "--end", now.isoformat(" "))
        run("preprocess.py", "--delete-raw")
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
