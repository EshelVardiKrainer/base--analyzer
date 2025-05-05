from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import os
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai
import json

load_dotenv()     
if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY not set - aborting.")                               
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.0-flash-exp-image-generation"  
model = genai.GenerativeModel(MODEL_ID)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS_TO_PROCESS  = 1
ALTITUDE_M       = 10.0
DISTANCE_M       = 1800.0
TILT_DEG         = 30.0
HEADING_DEG      = 0.0
LOAD_WAIT_SEC    = 12
TARGET_WIDTH_PX  = 1024
SCREENSHOTS_DIR  = Path("screenshots")

# ---------------------------------------------------------------------------
# Dynamic camera state
# ---------------------------------------------------------------------------
current_lat: float | None = None
current_lon: float | None = None
current_range_m           = DISTANCE_M   # starting zoom range
PAN_DEG    = 0.0008
ZOOM_FACTOR = 0.5
MIN_RANGE_M = 450
LLM_RETRY_ON_FORBIDDEN_ZOOMIN = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_coords(row: pd.Series) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) if found in *row*, else None."""
    lat = lon = None

    # Dedicated latitude/longitude columns?
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

    # Try to parse from any URL-like field containing "@lat,lon"
    regex = re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)")
    for col in row.index:
        m = regex.search(str(row[col]))
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def build_earth_url() -> str:
    """Build a Google-Earth Web URL from the current_lat/lon/range state."""
    return (
        "https://earth.google.com/web/@"
        f"{current_lat:.8f},{current_lon:.8f},{ALTITUDE_M:.2f}a,"
        f"{current_range_m:.2f}d,{TILT_DEG:.2f}y,{HEADING_DEG:.0f}h,0t,0r"
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
        
def gemini_analyze(image_path: Path, country: str, forbid_zoomin: bool = False) -> str:
    note = ""
    if forbid_zoomin:
        note = (
            "NOTE: The current image is already at the closest zoom level. "
            "You are NOT allowed to respond with 'zoom-in'. Choose one of: "
            "'zoom-out', 'move-north', 'move-south', 'move-east', 'move-west', or 'finish'.\n\n"
        )

    prompt = (
        note +
        "You are an expert in satellite-imagery interpretation working for the US Army.\n\n"
        "TASK:\n"
        f"  • You receive a satellite image of a suspected {country} military facility.\n"
        "  • Respond WITH NOTHING BUT a single, minified JSON object that matches the schema below.\n\n"
        "SCHEMA:\n"
        '{'
        '"findings":[string,...],'
        '"analysis":string,'
        '"things_to_continue_analyzing":[string,...],'
        '"action":"zoom-in"|"zoom-out"|"move-north"|"move-south"|"move-east"|"move-west"|"finish"'
        '}\n\n'
        "ACTION GUIDANCE:\n"
        '  • "zoom-in"  – need finer details;\n'
        '  • "zoom-out" – need broader context;\n'
        '  • "move-north/south/east/west" – likely features just outside the current frame;\n'
        '  • "finish"   – analysis complete.\n\n'
        "OUTPUT RULES: valid JSON only, double quotes, no markdown or commentary outside the object.\n"
        "If you believe we are ALREADY at the closest useful zoom, DO NOT ask for 'zoom-in'; choose a different action instead.\n"
        "DO NOT wrap the JSON in back-ticks or code fences."
    )
    
    with open(image_path, "rb") as img:
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": img.read()}]
        )
    return response.text.strip()


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
            global current_lat, current_lon, current_range_m
            if current_lat is None:              # first row only
                current_lat, current_lon = lat, lon

            # ----- 8 independent analysts loop -----
            for analyst_idx in range(8):
                # 1) Navigate to current camera position
                driver.get(build_earth_url())
                time.sleep(LOAD_WAIT_SEC)

                # 2) Screenshot
                shot_name = f"{analyst_idx:02d}_zoom{int(current_range_m)}.png"
                png_path  = SCREENSHOTS_DIR / shot_name
                driver.save_screenshot(str(png_path))
                jpg_path  = resize_to_width(png_path, TARGET_WIDTH_PX)
                safe_print_saved(jpg_path)

                # 3) Gemini call
                country = str(row.get("country", "unknown"))
                print(f"[→] Analyst {analyst_idx+1} reviewing image …")
                report = gemini_analyze(jpg_path, country)
                print(report, "\n")

                # 4) Parse JSON + update camera
                clean = report.strip()

                # ── Remove ``` fences, e.g. ```json ... ``` or plain ```
                if clean.startswith("```"):
                    #   ^```       optional language tag   whitespace/newline   …   closing ```
                    clean = re.sub(r"^```[\w]*\s*|```$", "", clean).strip()

                for attempt in range(2):  # allow 1 retry
                    try:
                        data = json.loads(clean)
                        action = data.get("action", "finish")
                    except json.JSONDecodeError:
                        print("[WARN] Invalid JSON – defaulting to 'finish'")
                        data = {}
                        action = "finish"

                    # If it requested zoom-in but we can't, ask Gemini again with hint
                    if (
                        LLM_RETRY_ON_FORBIDDEN_ZOOMIN and 
                        action == "zoom-in" and 
                        current_range_m <= MIN_RANGE_M and 
                        attempt == 0
                    ):
                        print("[INFO] LLM requested zoom-in but already at min zoom → suggesting alternatives")
                        clean = gemini_analyze(jpg_path, country, forbid_zoomin=True).strip()
                        # Remove ``` if needed
                        if clean.startswith("```"):
                            clean = re.sub(r"^```[\w]*\s*|```$", "", clean).strip()
                        continue  # retry loop
                    break  # exit loop


                # ---- auto-override if the image is clearly unusable -----------------
                analysis_txt = data.get("analysis", "").lower()
                too_blurry   = ("blurry" in analysis_txt) or ("black" in analysis_txt) or ("low resolution" in analysis_txt)
                no_findings  = not data.get("findings")

                if action == "zoom-in" and (too_blurry or no_findings or current_range_m <= MIN_RANGE_M):
                    print("[INFO] Image unusable at this zoom → switching to zoom-out")
                    action = "zoom-out"


                if action == "zoom-in":
                    if current_range_m > MIN_RANGE_M:
                        current_range_m *= ZOOM_FACTOR
                    else:
                        print("[INFO] Reached min zoom; ignoring further zoom-in requests")
                elif action == "zoom-out":
                    current_range_m /= ZOOM_FACTOR
                elif action == "move-north":
                    current_lat += PAN_DEG
                elif action == "move-south":
                    current_lat -= PAN_DEG
                elif action == "move-east":
                    current_lon += PAN_DEG
                elif action == "move-west":
                    current_lon -= PAN_DEG
                # 'finish' → leave camera unchanged

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
