"""Download SEVIRI Level 1.5 native (.nat) files from the EUMETSAT Data Store.

Example:
    python download.py --start 2025-06-01 --end 2025-06-02
    python download.py --start "2025-06-01 12:00" --end "2025-06-01 13:00" --dry-run
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import eumdac
from dotenv import load_dotenv
from tqdm import tqdm

import config


def get_datastore():
    load_dotenv(config.ROOT / ".env")
    key = os.getenv("EUMETSAT_CONSUMER_KEY")
    secret = os.getenv("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        sys.exit("Set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET in .env (see .env.example).")
    token = eumdac.AccessToken((key, secret))
    return eumdac.DataStore(token)


def download_product(product, out_dir, retries=3):
    """Download only the .nat entry of a product. Returns the path, or None if skipped."""
    nat_entries = [e for e in product.entries if e.endswith(".nat")]
    if not nat_entries:
        tqdm.write(f"  no .nat file in {product}, skipping")
        return None
    entry = nat_entries[0]
    target = out_dir / os.path.basename(entry)
    if target.exists():
        return target

    tmp = target.with_suffix(".nat.part")
    for attempt in range(1, retries + 1):
        try:
            with product.open(entry=entry) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.rename(target)
            return target
        except Exception as exc:  # network errors, token expiry, etc.
            tqdm.write(f"  {product}: attempt {attempt} failed ({exc})")
            tmp.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    tqdm.write(f"  giving up on {product}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=datetime.fromisoformat, help="UTC start, e.g. 2025-06-01 or '2025-06-01 12:00'")
    parser.add_argument("--end", required=True, type=datetime.fromisoformat, help="UTC end (exclusive)")
    parser.add_argument("--collection", default=config.COLLECTION)
    parser.add_argument("--dry-run", action="store_true", help="only list matching products")
    args = parser.parse_args()

    datastore = get_datastore()
    collection = datastore.get_collection(args.collection)
    products = list(collection.search(dtstart=args.start, dtend=args.end))
    products.sort(key=lambda p: str(p))
    print(f"Found {len(products)} products in {args.collection} between {args.start} and {args.end}")

    if args.dry_run:
        for p in products[:10]:
            print(" ", p)
        if len(products) > 10:
            print(f"  ... and {len(products) - 10} more")
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for product in tqdm(products, unit="file"):
        if download_product(product, config.RAW_DIR) is None:
            failed.append(str(product))

    print(f"Done: {len(products) - len(failed)} ok, {len(failed)} failed/skipped")
    for name in failed:
        print("  failed:", name)


if __name__ == "__main__":
    main()
