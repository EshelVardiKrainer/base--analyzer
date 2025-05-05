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
from openai import OpenAI

load_dotenv()     
if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY not set - aborting.")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.0-flash-exp-image-generation"  
model = genai.GenerativeModel(MODEL_ID)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("OPENROUTER_API_KEY not set!")
router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# choose the Microsoft reasoning model
REASONING_MODEL = "qwen/qwen3-235b-a22b:free"                           


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS_TO_PROCESS  = 8
ALTITUDE_M       = 10.0
DISTANCE_M       = 1800.0
TILT_DEG         = 30.0
HEADING_DEG      = 0.0
LOAD_WAIT_SEC    = 12
TARGET_WIDTH_PX  = 1024
SCREENSHOTS_DIR  = Path("screenshots")
DATA_FILE        = Path("data.json")

# ---------------------------------------------------------------------------
# Dynamic camera state
# ---------------------------------------------------------------------------
current_lat: float | None = None
current_lon: float | None = None
current_range_m           = DISTANCE_M   # starting zoom range
PAN_DEG    = 0.0006
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
        
def gemini_analyze(image_path: Path, country: str, forbid_zoomin: bool = False, history: list[str] = []) -> str:
    note = ""
    if forbid_zoomin:
        note = (
            "NOTE: The current image is already at the closest zoom level. "
            "You are NOT allowed to respond with 'zoom-in'. Choose one of: "
            "'zoom-out', 'move-north', 'move-south', 'move-east', 'move-west', or 'finish'.\n\n"
        )
    prompt = note + (
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
        "If you believe we are ALREADY at the closest useful zoom, DO NOT ask for \"zoom-in\"; choose a different action instead.\n"
        "DO NOT wrap the JSON in back-ticks or code fences."
    )

    if history:
        # join each previous report with blank lines
        hist_block = "\n\n".join(history)
        prompt += (
            "\n\nHere is what previous analysts said (don’t take it as gospel, think for yourself):\n"
            f"{hist_block}\n"
        )

    with open(image_path, "rb") as img:
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": img.read()}]
        )
    return response.text.strip()


def process_csv(csv_path: Path) -> None:
    # --- Load existing data ---
    if DATA_FILE.exists():
        try:
            all_results = json.loads(DATA_FILE.read_text())
            print(f"[INFO] Loaded {len(all_results)} existing records from {DATA_FILE}")
        except json.JSONDecodeError:
            print(f"[WARN] Failed to decode existing {DATA_FILE}. Starting fresh.", file=sys.stderr)
            all_results: list[dict] = [] # Ensure type hint if file is invalid
    else:
        all_results: list[dict] = []
        print(f"[INFO] {DATA_FILE} not found. Starting fresh.")
    # Create a set of names for faster lookup
    processed_base_names = {r.get("name") for r in all_results if r.get("name")}
    # --- End Load existing data ---
    df = pd.read_csv(csv_path)
    ensure_dir()

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")         # visible window
    # options.add_argument("--headless=new")          # uncomment ONLY if needed

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    global current_lat, current_lon, current_range_m

    try:
        for idx, row in df.head(ROWS_TO_PROCESS).iterrows():
            original_coords = parse_coords(row)
            if original_coords is None: # Check if coords were successfully parsed
                print(f"[WARN] Row {idx+1}: Coordinates not found. Skipping…", file=sys.stderr)
                continue
            
            # --- Create unique base name ---
            lat_str = f"{original_coords[0]:.5f}".replace('.', 'p')
            lon_str = f"{original_coords[1]:.5f}".replace('.', 'p')
            raw_name = str(row.get("name", f"base_lat{lat_str}_lon{lon_str}"))
            base_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name) # Sanitize name
            # --- Skip if already processed ---
            if base_name in processed_base_names:
                print(f"[SKIP] Analysis for '{base_name}' found in {DATA_FILE}. Skipping.")
                continue
            current_range_m = DISTANCE_M      # Reset zoom to the default start distance
            current_lat, current_lon = original_coords  # Use the coords parsed from the CURRENT row
            print(f"[INFO] Processing Row {idx+1}: Set location to Lat: {current_lat:.6f}, Lon: {current_lon:.6f}, Range: {current_range_m}m") # Optional: Log the update
            
        
            analyst_report_strings: list[str] = []
            parsed_analyst_reports: list[dict] = []
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
                # pass the raw list so gemini_analyze can join it itself
                report_str = gemini_analyze(jpg_path, country, forbid_zoomin=False, history=analyst_report_strings)
                analyst_report_strings.append(report_str)
                print(report_str, "\n")

                # 4) Parse JSON + update camera
                clean = report_str.strip()

                # ── Remove ``` fences, e.g. ```json ... ``` or plain ```
                if clean.startswith("```"):
                    #   ^```       optional language tag   whitespace/newline   …   closing ```
                    clean = re.sub(r"^```[\w]*\s*|```$", "", clean).strip()

                parsed_data = None
                for attempt in range(2):  # allow 1 retry
                    try:
                        data = json.loads(clean)
                        action = data.get("action", "finish")
                        parsed_data = data
                    except json.JSONDecodeError:
                        print(f"[WARN] Analyst {analyst_idx+1} for '{base_name}' returned invalid JSON. Defaulting action to 'finish'.")
                        parsed_data = {"error": "Invalid JSON response", "raw_response": report_str}

                    # If it requested zoom-in but we can't, ask Gemini again with hint
                    if (
                        LLM_RETRY_ON_FORBIDDEN_ZOOMIN and 
                        action == "zoom-in" and 
                        current_range_m <= MIN_RANGE_M and 
                        attempt == 0
                    ):
                        print("[INFO] LLM requested zoom-in but already at min zoom → suggesting alternatives")
                        clean = gemini_analyze(jpg_path, country, forbid_zoomin=True).strip()
                        if clean.startswith("```"):
                            clean = re.sub(r"^```[\w]*\s*|```$", "", clean).strip()
                        continue  # retry loop
                    break  # exit loop
                
                # Store the parsed data (or error dict) from the analyst report
                if parsed_data is not None:
                    parsed_analyst_reports.append(parsed_data)


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
                
            try:
                final_report = generate_commander_report(analyst_report_strings)
                print(f"\n=== FINAL COMMANDER REPORT ('{base_name}') ===") # Modified print
                print(final_report)
            except Exception as e:
                print(f"[ERROR] Failed to generate commander report for '{base_name}': {e}", file=sys.stderr)
                final_report = f"Error generating commander report: {e}" # Store error message
            # persist this base’s analysis
            record = {
                "name":             base_name,
                "country":          str(row.get("country", "unknown")),
                "original_coords":  original_coords,
                "analysts":         parsed_analyst_reports,
                "commander_report": final_report
            }
            all_results = [r for r in all_results if r.get("name") != base_name]
            all_results.append(record)
            processed_base_names.add(base_name)
            try:
                DATA_FILE.write_text(json.dumps(all_results, indent=2))
                print(f"[✓] Saved analysis of '{base_name}' to {DATA_FILE}")
            except Exception as e:
                print(f"[ERROR] Failed to write results to {DATA_FILE}: {e}", file=sys.stderr)

    finally:
        driver.quit()
        
def generate_commander_report(analyst_report_strings: list[str]) -> str:
    """
    Given the 8 analysts' JSON strings, synthesize a final commander report via OpenRouter.
    """
    # 1) Build the combined history block
    history_block = "\n\n".join(analyst_report_strings)

    # 2) Commander‐style system prompt
    commander_prompt = f"""
You are a commander of military analysts and you are investigating a suspected enemy facility.
Here is the history of what eight different analysts reported (each written independently):

{history_block}


YOUR TASK:
1. Write a concise **executive summary** of the site.
2. Identify the **three most critical military assets or patterns** observed.
3. Provide a **threat assessment** describing possible enemy capabilities.
4. Recommend **next steps** (e.g., reconnaissance actions, target priorities, security measures).
5. Conclude with a **final recommendation**: choose one of [attack, monitor, recon, dismiss].

Present your answer as a **professional briefing** in **plain text**, using paragraphs and bullet points—not JSON. 
"""

    # 3) Call the OpenRouter chat endpoint
    resp = router_client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": commander_prompt},
            {"role": "user",   "content": "Please draft a professional, narrative briefing in plain text—no JSON."}
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    #print(f"\n[DEBUG] OpenRouter Raw Response for Commander:\n{resp}\n")
    choices = resp.choices
    if not choices or not choices[0].message or not choices[0].message.content:
        raise RuntimeError("Commander LLM returned no content")
    return choices[0].message.content.strip()



    


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
