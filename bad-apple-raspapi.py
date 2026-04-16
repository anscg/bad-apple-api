import requests
import time
import sys
import argparse
import io
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_BASE = "http://localhost:8000"
CANVAS_URL = "https://raspapi.hackclub.com/api/canvas"
FRAME_DELAY = 6  # seconds per frame (measured from frame start)
MAX_WORKERS = 128
MAX_RETRIES = 3

username = os.environ["PROXY_USERNAME"]
password = os.environ["PROXY_PASSWORD"]
host = os.environ["PROXY_HOST"]
port = os.environ["PROXY_PORT"]
proxy = f"http://{username}:{password}@{host}:{port}"
proxies = {"http": proxy, "https": proxy}


def get_frame(frame_number):
    resp = requests.get(
        f"{API_BASE}/frame/{frame_number}",
        params={"width": 32, "height": 32, "img_format": "json"},
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def try_send_pixel(x, y, color):
    """Returns (x, y, color) on failure, None on success."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                CANVAS_URL,
                json={"x": x, "y": y, "color": color},
                proxies=proxies,
                timeout=8,
            )
            r.raise_for_status()
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.3)
    return (x, y, color)


def frame_to_pixels(data):
    pixels = {}
    for y, row in enumerate(data["pixels"]):
        for x, white in enumerate(row):
            pixels[(x, y)] = "#ffffff" if white else "#000000"
    return pixels


def image_url_to_pixels(url):
    """Download image from URL, resize to 32x32, return {(x,y): '#rrggbb'}."""
    print(f"Downloading image from {url}...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img)
    pixels = {}
    for y in range(32):
        for x in range(32):
            r, g, b = arr[y, x]
            pixels[(x, y)] = f"#{r:02x}{g:02x}{b:02x}"
    return pixels


def send_batch(to_send, desc=""):
    """Send a dict of {(x,y): color} in parallel. Returns list of failed (x,y,color)."""
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(try_send_pixel, x, y, color): (x, y, color)
            for (x, y), color in to_send.items()
        }
        with tqdm(total=len(futures), desc=desc, unit="px", leave=False) as bar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    failed.append(result)
                bar.update(1)
    return failed


def play(start_frame=0):
    prev_pixels = {}
    carry_over = []
    frame_number = start_frame

    while True:
        t0 = time.time()
        print(f"\nFrame {frame_number}: fetching...", end=" ", flush=True)
        data = get_frame(frame_number)
        if data is None:
            print("No more frames. Done.")
            break

        curr_pixels = frame_to_pixels(data)

        changed = {
            (x, y): color
            for (x, y), color in curr_pixels.items()
            if prev_pixels.get((x, y)) != color
        }

        to_send = {(x, y): color for x, y, color in carry_over}
        to_send.update(changed)

        prev_pixels = curr_pixels

        if not to_send:
            print("no changes, skipping")
            frame_number += 1
            continue

        if carry_over:
            print(f"{len(changed)} changed + {len(carry_over)} retried = {len(to_send)} total")
        else:
            print(f"{len(to_send)} pixels to send")

        failed = send_batch(to_send, desc=f"Frame {frame_number}")

        carry_over = failed
        if failed:
            tqdm.write(f"  {len(failed)} pixels failed, will retry next frame")

        elapsed = time.time() - t0
        wait = max(0, FRAME_DELAY - elapsed)
        tqdm.write(f"  Frame {frame_number} done in {elapsed:.1f}s, waiting {wait:.1f}s")
        time.sleep(wait)
        frame_number += 1


def display_custom(url):
    carry_over = []

    # Initial send
    t0 = time.time()
    try:
        curr_pixels = image_url_to_pixels(url)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    to_send = curr_pixels
    print(f"Sending {len(to_send)} pixels...")
    failed = send_batch(to_send, desc="Custom image")
    carry_over = failed

    if failed:
        tqdm.write(f"  {len(failed)} pixels failed, retrying...")

    # Retry loop until all pixels are sent
    while carry_over:
        t0 = time.time()
        to_send = {(x, y): color for x, y, color in carry_over}
        print(f"Retrying {len(to_send)} failed pixels...")
        failed = send_batch(to_send, desc="Retry")
        carry_over = failed
        elapsed = time.time() - t0
        wait = max(0, FRAME_DELAY - elapsed)
        if carry_over:
            tqdm.write(f"  {len(carry_over)} still failing, waiting {wait:.1f}s before retry")
            time.sleep(wait)

    tqdm.write("Done! Image displayed on canvas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bad Apple on raspapi canvas")
    parser.add_argument("start_frame", nargs="?", type=int, default=0,
                        help="Frame to start from (default: 0)")
    parser.add_argument("--custom", metavar="URL",
                        help="Display a custom image from a CDN URL instead of playing Bad Apple")
    args = parser.parse_args()

    if args.custom:
        display_custom(args.custom)
    else:
        play(args.start_frame)
