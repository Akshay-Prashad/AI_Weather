"""Shared settings for the SEVIRI-over-Ireland data pipeline."""
from pathlib import Path

from pyresample import create_area_def

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"          # resampling lookup tables
QUICKLOOK_DIR = ROOT / "data" / "quicklook"
ZARR_PATH = PROCESSED_DIR / "ireland_seviri.zarr"            # live NRT data (collect.py)
ARCHIVE_ZARR_PATH = PROCESSED_DIR / "archive_seviri.zarr"    # historical data (import_archive.py)

# Rapid Scan Service: 5-minute scans of Europe (fallback: "EO:EUM:DAT:MSG:HRSEVIRI", 15-min full disk)
COLLECTION = "EO:EUM:DAT:MSG:MSG15-RSS"

# IR + water-vapour channels work day and night. Add "VIS006", "VIS008", "IR_016"
# for daytime-only visible reflectances (NaN/dark at night).
CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_108", "IR_120", "IR_134"]

# Fixed target grid: 384 x 384 pixels at 3 km in a Lambert azimuthal equal-area
# projection centred west of Ireland, so Atlantic weather is visible before it
# arrives. Covers roughly lon -18.6..-1.4, lat 47.8..58.2.
GRID_SIZE = 384
RESOLUTION_M = 3000
_half = GRID_SIZE * RESOLUTION_M / 2
AREA = create_area_def(
    "ireland_laea_3km",
    {"proj": "laea", "lat_0": 53.0, "lon_0": -10.0, "ellps": "WGS84", "units": "m"},
    width=GRID_SIZE,
    height=GRID_SIZE,
    area_extent=(-_half, -_half, _half, _half),
)
