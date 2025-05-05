"""base_analyzer.py
Automates Google Earth web views of military bases listed in a CSV, captures
screenshots, rescales them to 1024‑px width, and stores them as JPEGs.

Usage
-----
    # (inside an activated venv)
    pip install pandas pillow selenium webdriver-manager
    python base_analyzer.py military_bases.csv

The script expects Google Chrome to be installed locally. webdriver‑manager
will fetch the correct ChromeDriver automatically. Headless mode is **off** so
you can watch what happens while debugging.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS_TO_PROCESS  = 5      # how many CSV rows to visit
ALTITUDE_M       = 10.0   # camera altitude ("a")
DISTANCE_M       = 1800.0 # camera distance / range ("d")
TILT_DEG         = 30.0   # camera tilt ("y")
HEADING_DEG      = 0.0    # compass heading ("h")
LOAD_WAIT_SEC    = 12      # wait for Earth tiles to load (seconds)
TARGET_WIDTH_PX  = 1024   # resize screenshot width
SCREENSHOTS_DIR  = Path("screenshots")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_coords(row: pd.Series) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) if found in *row*, else None."""
    lat = lon = None

    # 1️⃣  Dedicated latitude/longitude columns?
    for col in row.index:
        v = row[col]
        if not isinstance(v, (int, float)):
            continue
        lower = col.lower()
        if "lat" in lower:
            lat = float(v)
        elif "lon" in lower or "lng" in lower:
            lon = float(v)
    if lat is not None and lon is not None:
        return lat, lon

    # 2️⃣  Try to parse from any URL-like field containing "@lat,lon"
    regex = re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)")
    for col in row.index:
        m = regex.search(str(row[col]))
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def build_earth_url(lat: float, lon: float) -> str:
    return (
        "https://earth.google.com/web/@"
        f"{lat:.8f},{lon:.8f},{ALTITUDE_M:.2f}a,"
        f"{DISTANCE_M:.2f}d,{TILT_DEG:.2f}y,{HEADING_DEG:.0f}h,0t,0r"
    )


def ensure_dir() -> None:
    SCREENSHOTS_DIR.mkdir(exist_ok=True)


def resize_to_width(img_path: Path, width: int) -> Path:
    """Downscale *img_path* to *width* px and convert to JPEG. Returns new path."""
    with Image.open(img_path) as im:
        ratio = width / im.width
        new_size = (width, int(im.height * ratio))
        im_resized = im.resize(new_size, Image.LANCZOS)
        out_path = img_path.with_suffix(".jpg")
        im_resized.save(out_path, format="JPEG", quality=90)
    img_path.unlink(missing_ok=True)
    return out_path


def safe_print_saved(path: Path) -> None:
    try:
        rel = path.relative_to(Path.cwd())
        print(f"[✓] Saved {rel}")
    except ValueError:
        print(f"[✓] Saved {path}")

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    ensure_dir()

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")         # visible window
    # options.add_argument("--headless=new")          # uncomment ONLY if needed

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        for idx, row in df.head(ROWS_TO_PROCESS).iterrows():
            coords = parse_coords(row)
            if coords is None:
                print(f"[WARN] Row {idx}: Coordinates not found. Skipping…", file=sys.stderr)
                continue

            lat, lon = coords
            url = build_earth_url(lat, lon)
            driver.get(url)
            time.sleep(LOAD_WAIT_SEC)

            base_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(row.get("name", f"base_{idx}")))
            png_path = SCREENSHOTS_DIR / f"{idx:02d}_{base_name}.png"
            driver.save_screenshot(str(png_path))
            jpg_path = resize_to_width(png_path, TARGET_WIDTH_PX)
            safe_print_saved(jpg_path)
    finally:
        driver.quit()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Capture Google Earth screenshots for bases in a CSV file.")
    ap.add_argument("csv", type=Path, help="CSV file containing military bases")
    args = ap.parse_args()
    process_csv(args.csv)


if __name__ == "__main__":
    main()
