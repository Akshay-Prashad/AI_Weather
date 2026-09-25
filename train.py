"""Train the cloud-cover U-Net and score it against persistence on held-out data.

By default trains on the historical archive before --test-from and tests on the archive after it
(a different year); --data live uses the live NRT store with a chronological split instead.

Example:
    python train.py --epochs 15
    python train.py --data live --epochs 30
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cloudnet as cn
import config

MODEL_DIR = config.ROOT / "models"
DATA_PATHS = {"archive": config.ARCHIVE_ZARR_PATH, "live": config.ZARR_PATH}


def split(times, indices, test_from=None, test_frac=0.2):
    """Test = samples from `test_from` on, or the latest `test_frac`; train never shares a frame with test."""
    if test_from is not None:
        is_test = times[indices] >= test_from
        test = indices[is_test]
        if not len(test):
            raise SystemExit(f"No samples on/after {test_from}; import some test days first.")
    else:
        test = indices[-max(1, int(len(indices) * test_frac)):]
    gap = cn.HISTORY + max(cn.LEADS)
    train = indices[indices <= test[0] - gap] if test_from is None else indices[times[indices] < test_from]
    return train, test


def evaluate(model, loader, dev):
    model.eval()
    preds, persist, truths = [], [], []
    with torch.no_grad():
        for inp, tgt, now in loader:
            preds.append(torch.sigmoid(model(inp.to(dev))).cpu() > 0.5)
            persist.append((now > 0.5).unsqueeze(1).expand_as(tgt))
            truths.append(tgt)
    truth = torch.cat(truths)
    return cn.scores(torch.cat(preds), truth), cn.scores(torch.cat(persist), truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=DATA_PATHS, default="archive")
    parser.add_argument("--test-from", default="2022-01-01", help="archive only: test on data from this date on")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=1, help="use every Nth training sample (neighbours are near-duplicates)")
    args = parser.parse_args()
    torch.manual_seed(0)
    dev = cn.device()

    t0 = time.time()
    times, x, y = cn.load_frames(DATA_PATHS[args.data])
    indices = cn.sample_indices(times)
    if len(indices) < 10:
        raise SystemExit(f"Only {len(indices)} usable samples from {len(times)} frames; collect more data.")
    test_from = pd.Timestamp(args.test_from) if args.data == "archive" else None
    train_idx, test_idx = split(times, indices, test_from)
    train_idx = train_idx[:: args.stride]
    print(f"{len(times)} frames ({times[0]} .. {times[-1]}) loaded in {time.time() - t0:.0f}s; "
          f"{len(train_idx)} train / {len(test_idx)} test samples; device {dev}")

    loader_kw = dict(batch_size=args.batch, num_workers=4, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(cn.CloudDataset(x, y, train_idx), shuffle=True, **loader_kw)
    test_loader = torch.utils.data.DataLoader(cn.CloudDataset(x, y, test_idx), **loader_kw)

    model = cn.UNet(cn.HISTORY * len(config.CHANNELS), len(cn.LEADS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for inp, tgt, _ in train_loader:
            inp, tgt = inp.to(dev), tgt.to(dev)
            valid = ~torch.isnan(tgt)
            # soft targets: each 9 km pixel's cloudy fraction; ignore pixels outside the scan
            loss = F.binary_cross_entropy_with_logits(model(inp)[valid], tgt[valid])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(inp)
        print(f"epoch {epoch:3d}  train loss {total / len(train_idx):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    model_scores, persist_scores = evaluate(model, test_loader, dev)
    print(f"\n{'lead':8} {'model acc':>10} {'persist acc':>12} {'model IoU':>10} {'persist IoU':>12}")
    for name, m, p in zip(cn.LEAD_NAMES, model_scores, persist_scores):
        print(f"{name:8} {m['accuracy']:10.1%} {p['accuracy']:12.1%} {m['iou']:10.3f} {p['iou']:12.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, MODEL_DIR / "cloud_unet.pt")
    results = {"data": args.data, "frames": len(times), "first": str(times[test_idx[0]]), "last": str(times[test_idx[-1]]),
               "train_samples": len(train_idx), "test_samples": len(test_idx),
               "leads": cn.LEAD_NAMES, "model": model_scores, "persistence": persist_scores}
    (MODEL_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {MODEL_DIR / 'cloud_unet.pt'} and results.json")


if __name__ == "__main__":
    main()
