import argparse
import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", default="IR_108")
    parser.add_argument("--n", type=int, default=4, help="number of evenly spaced frames")
    args = parser.parse_args()

    ds = xr.open_zarr(config.ZARR_PATH)
    idx = np.linspace(0, ds.sizes["time"] - 1, min(args.n, ds.sizes["time"])).astype(int)
    crs = config.AREA.to_cartopy_crs()
    config.QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)

    for i in idx:
        frame = ds.data.isel(time=i).sel(channel=args.channel)
        t = str(frame.time.values)[:16]
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection=crs)
        # reversed grey colormap: cold cloud tops appear white, as in standard IR imagery
        im = ax.imshow(frame.values, transform=crs, extent=crs.bounds, origin="upper", cmap="gray_r")
        ax.coastlines(color="yellow", linewidth=0.7)
        ax.set_title(f"{args.channel}  {t} UTC")
        fig.colorbar(im, ax=ax, shrink=0.7, label="K")
        out = config.QUICKLOOK_DIR / f"{args.channel}_{t.replace(':', '').replace('T', '_')}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
