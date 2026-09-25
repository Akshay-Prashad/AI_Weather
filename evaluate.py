"""Presentation figures for the trained cloud model: scores vs persistence, and example forecasts.

Example:
    python evaluate.py                   # held-out archive test period
    python evaluate.py --data live       # today's live data
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import cloudnet as cn
import config
from train import DATA_PATHS, MODEL_DIR, evaluate, split

FIG_DIR = config.ROOT / "figures"


def score_chart(results, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(results["leads"]))
    for ax, metric, label in [(axes[0], "accuracy", "Pixel accuracy"), (axes[1], "iou", "Cloud IoU")]:
        for offset, key, legend in [(-0.2, "persistence", "Persistence (no change)"), (0.2, "model", "U-Net")]:
            vals = [s[metric] for s in results[key]]
            bars = ax.bar(x + offset, vals, 0.4, label=legend)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, results["leads"])
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{name} test: {results['test_samples']} forecasts, {results['first'][:16]} .. {results['last'][:16]} UTC")
    fig.tight_layout()
    out = FIG_DIR / f"scores_{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def example(model, ds, times, k, crs):
    inp, tgt, now = ds[k]
    with torch.no_grad():
        prob = torch.sigmoid(model(inp[None]))[0].numpy()
    t0 = times[ds.indices[k]]
    lead = len(cn.LEADS) - 1   # show the longest lead
    panels = [
        (now.numpy(), f"Now ({t0:%H:%M} UTC)", "Blues_r"),
        (tgt[lead].numpy(), f"Actual {cn.LEAD_NAMES[lead]}", "Blues_r"),
        (now.numpy(), f"Persistence {cn.LEAD_NAMES[lead]}", "Blues_r"),
        (prob[lead], f"U-Net {cn.LEAD_NAMES[lead]}", "Blues_r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), subplot_kw={"projection": crs})
    for ax, (img, title, cmap) in zip(axes, panels):
        im = ax.imshow(img, transform=crs, extent=crs.bounds, origin="upper", cmap=cmap, vmin=0, vmax=1)
        ax.coastlines(color="orange", linewidth=0.7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cloud fraction / probability")
    out = FIG_DIR / f"example_{t0:%Y%m%d_%H%M}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=DATA_PATHS, default="archive",
                        help="archive: the held-out test period; live: every usable sample of the live NRT data")
    parser.add_argument("--examples", type=int, default=3, help="number of example forecasts to plot")
    args = parser.parse_args()
    FIG_DIR.mkdir(exist_ok=True)

    ckpt = torch.load(MODEL_DIR / "cloud_unet.pt", weights_only=False)
    model = cn.UNet(cn.HISTORY * len(config.CHANNELS), len(cn.LEADS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    times, x, y = cn.load_frames(DATA_PATHS[args.data])
    indices = cn.sample_indices(times)
    if args.data == "archive":
        _, indices = split(times, indices, pd.Timestamp(ckpt["args"]["test_from"]))
        results = json.loads((MODEL_DIR / "results.json").read_text())
    else:
        # score the model on today's live data, which it never saw in training
        if not len(indices):
            raise SystemExit("Not enough consecutive live frames yet (need 90 min); let collect.py run longer.")
        loader = torch.utils.data.DataLoader(cn.CloudDataset(x, y, indices), batch_size=16)
        model_scores, persist_scores = evaluate(model, loader, torch.device("cpu"))
        results = {"test_samples": len(indices), "first": str(times[indices[0]]), "last": str(times[indices[-1]]),
                   "leads": cn.LEAD_NAMES, "model": model_scores, "persistence": persist_scores}
    print("saved", score_chart(results, args.data))

    ds = cn.CloudDataset(x, y, indices)
    crs = config.AREA.to_cartopy_crs()
    for k in np.linspace(0, len(ds) - 1, min(args.examples, len(ds))).astype(int):
        print("saved", example(model, ds, times, k, crs))


if __name__ == "__main__":
    main()
