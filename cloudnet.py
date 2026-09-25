"""Cloud-cover nowcasting: data preparation and the U-Net model.

Task: from the last 30 minutes of SEVIRI scans (6 frames x 7 channels), predict the
cloud mask at +15, +30 and +60 minutes.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

import config

HISTORY = 6                 # input frames (5-min steps -> 30 min)
LEADS = [3, 6, 12]          # output steps ahead (+15, +30, +60 min)
LEAD_NAMES = ["+15 min", "+30 min", "+60 min"]
DOWNSAMPLE = 3              # 384 px @ 3 km -> 128 px @ 9 km, keeps CPU training fast
CLOUD_THRESHOLD_K = 270.0   # IR_108 colder than this counts as cloud (simple rule)
STEP = pd.Timedelta(minutes=5)
# fixed normalisation for every channel (all brightness temperatures, ~210-310 K), so live and
# archive data are scaled identically; keeps values near 0 for float16 storage
NORM_MEAN, NORM_STD = 260.0, 25.0


def load_frames(path=config.ZARR_PATH, load_chunk=288):
    """Return times, normalised inputs (T, C, H, W) and cloud fraction per pixel (T, H, W), float16, time-sorted.

    The result is cached next to the resampling cache and reused while the store's frame count is unchanged.
    """
    with xr.open_zarr(path) as ds:
        data = ds.data.isel(time=np.argsort(ds.time.values))
        times = pd.DatetimeIndex(data.time.values)
        cache = config.CACHE_DIR / f"{path.stem}_{len(times)}_{DOWNSAMPLE}_{CLOUD_THRESHOLD_K:g}.npz"
        if cache.exists():
            f = np.load(cache)
            return times, f["x"], f["y"]
        n, c, h, w = data.shape
        x = np.empty((n, c, h // DOWNSAMPLE, w // DOWNSAMPLE), np.float16)
        y = np.empty((n, h // DOWNSAMPLE, w // DOWNSAMPLE), np.float16)
        coarse = dict(y=DOWNSAMPLE, x=DOWNSAMPLE)
        for s in range(0, n, load_chunk):   # chunked so a months-long archive fits in memory
            part = data.isel(time=slice(s, s + load_chunk)).load().astype(np.float32)
            ir108 = part.sel(channel="IR_108")
            # cloud label at full resolution, then averaged -> fraction of each 9 km pixel that is cloudy
            cloud = (ir108 < CLOUD_THRESHOLD_K).where(ir108.notnull())
            x[s: s + load_chunk] = (part.coarsen(**coarse).mean().values - NORM_MEAN) / NORM_STD
            y[s: s + load_chunk] = cloud.coarsen(**coarse).mean().values
    for old in config.CACHE_DIR.glob(f"{path.stem}_*.npz"):
        old.unlink()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return times, x, y


def sample_indices(times):
    """Indices t where frames t-HISTORY+1 .. t and every lead target are exactly 5 min apart."""
    pos = {t: i for i, t in enumerate(times)}
    out = []
    for i, t in enumerate(times):
        needed = [t - k * STEP for k in range(HISTORY)] + [t + k * STEP for k in LEADS]
        if all(n in pos for n in needed):
            out.append(i)
    return np.array(out, dtype=int)


class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, indices):
        self.x, self.y, self.indices = x, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        inp = np.nan_to_num(self.x[i - HISTORY + 1: i + 1].reshape(-1, *self.x.shape[-2:]))   # (HISTORY*C, H, W)
        tgt = np.stack([self.y[i + lead] for lead in LEADS])                                  # (len(LEADS), H, W)
        now = self.y[i]                                                                      # for persistence baseline
        return (torch.from_numpy(inp.astype(np.float32)), torch.from_numpy(tgt.astype(np.float32)),
                torch.from_numpy(now.astype(np.float32)))


def device():
    """ROCm GPUs also appear as 'cuda' in PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small 3-level U-Net; outputs one cloud-probability logit map per lead time."""

    def __init__(self, in_ch, out_ch, base=32):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = _block(in_ch, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.mid = _block(base * 4, base * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), _block(base * 8, base * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), _block(base * 4, base * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base * 2, base, 2, 2), _block(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def scores(pred_cloud, true_frac):
    """Pixel accuracy and cloud IoU per lead. pred_cloud: bool (N, L, H, W); true_frac: (N, L, H, W) with NaN outside scan."""
    valid = ~torch.isnan(true_frac)
    truth = true_frac > 0.5
    out = []
    for lead in range(truth.shape[1]):
        v, p, t = valid[:, lead], pred_cloud[:, lead], truth[:, lead]
        acc = (p == t)[v].float().mean().item()
        inter = (p & t)[v].sum().item()
        union = (p | t)[v].sum().item()
        out.append({"accuracy": acc, "iou": inter / union if union else float("nan")})
    return out
