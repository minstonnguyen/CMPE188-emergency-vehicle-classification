"""Download police car images from Wikimedia Commons.

Uses the public MediaWiki API (no auth, no Cloudflare). Police cars are an
emergency vehicle, so images land in `data/incoming/emergency_vehicle/`.

Usage (from project root):
    python scripts/scrape_policecar_images.py --target 100
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

API_URL = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's User-Agent policy requires a descriptive UA, but
# `upload.wikimedia.org` (the static media host) tends to 403 anything that
# looks like a script. We send a realistic browser UA plus a `From` header so
# we still satisfy the spirit of the policy.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CONTACT_EMAIL = "local-dev@example.com"

# Search terms - kept fairly broad to hit 100 unique images across many forces.
SEARCH_TERMS = [
    "police car",
    "police cruiser",
    "police vehicle",
    "patrol car",
    "police interceptor",
    "Ford Crown Victoria police",
    "Ford Police Interceptor",
    "Chevrolet Caprice police",
    "Dodge Charger police",
    "NYPD car",
    "LAPD car",
    "CHP patrol car",
    "Metropolitan Police car London",
    "Polizei car Germany",
    "Carabinieri car",
    "Gendarmerie car",
    "Guardia Civil car",
    "Royal Canadian Mounted Police vehicle",
    "Australian police car",
]

# Skip drawings, diagrams, and very small icons.
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
MIN_WIDTH = 400
MIN_BYTES = 15 * 1024

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "incoming" / "emergency_vehicle"


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:80]


def api_get(session: requests.Session, params: dict, max_retries: int = 5) -> dict:
    """GET the MediaWiki API with exponential backoff on 429/5xx."""
    delay = 2.0
    for attempt in range(max_retries):
        resp = session.get(API_URL, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            print(f"  ! API {resp.status_code}; backing off {delay:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"API kept failing for params={params}")


def search_files(session: requests.Session, term: str, limit: int = 50) -> list[str]:
    """Return Commons file titles (e.g. 'File:Foo.jpg') matching `term`."""
    data = api_get(
        session,
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": term,
            "srnamespace": 6,
            "srlimit": limit,
            "srwhat": "text",
        },
    )
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def file_info(session: requests.Session, titles: list[str]) -> list[dict]:
    """Look up imageinfo (URL, size, mime) for up to 50 file titles per call."""
    out: list[dict] = []
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        data = api_get(
            session,
            {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "titles": "|".join(chunk),
            },
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            info["_title"] = page.get("title", "")
            out.append(info)
        time.sleep(1.0)
    return out


def download_bytes(session: requests.Session, url: str, max_retries: int = 4) -> bytes | None:
    """Download `url` honoring `Retry-After` on 429/5xx. Returns bytes or None."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! request error {url}: {exc}")
            return None
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after_raw = r.headers.get("Retry-After", "")
            try:
                # Cap at 90s so we don't sit forever; if upstream wants more
                # than that we'll just stop the run.
                wait_s = min(int(retry_after_raw), 90) if retry_after_raw else 15
            except ValueError:
                wait_s = 15
            print(f"  ! HTTP {r.status_code}; Retry-After={retry_after_raw or 'n/a'}; "
                  f"sleeping {wait_s}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_s)
            if retry_after_raw and int(retry_after_raw) > 90:
                # Server wants a long break - bubble up so caller can stop early.
                return "RATE_LIMITED"  # type: ignore[return-value]
            continue
        print(f"  ! HTTP {r.status_code} {url}")
        return None
    print(f"  ! gave up on {url}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100, help="How many images to download.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/incoming/emergency_vehicle).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_hashes: set[str] = set()
    for p in out_dir.iterdir():
        if p.is_file():
            try:
                existing_hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
            except Exception:
                pass
    print(f"Output: {out_dir} (already has {len(existing_hashes)} files)")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "From": CONTACT_EMAIL,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://commons.wikimedia.org/",
        }
    )

    # Phase 1: gather candidate file titles.
    candidate_titles: list[str] = []
    seen_titles: set[str] = set()
    for term in SEARCH_TERMS:
        try:
            titles = search_files(session, term, limit=50)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! search '{term}' failed: {exc}")
            continue
        added = 0
        for t in titles:
            if t in seen_titles:
                continue
            seen_titles.add(t)
            candidate_titles.append(t)
            added += 1
        print(f"search '{term}': +{added} new (total {len(candidate_titles)})")
        time.sleep(1.0)
        # 3x target is plenty - many will be filtered (SVGs, tiny, near-dupes).
        if len(candidate_titles) >= args.target * 3:
            break

    # Cap candidates so we don't hammer the API when we already have plenty.
    candidate_titles = candidate_titles[: max(args.target * 3, 150)]
    print(f"\n{len(candidate_titles)} candidate file titles. Fetching imageinfo...")
    infos = file_info(session, candidate_titles)
    print(f"got imageinfo for {len(infos)} files\n")

    saved = 0
    for info in infos:
        if saved >= args.target:
            break
        url = info.get("url") or ""
        mime = info.get("mime") or ""
        width = int(info.get("width") or 0)
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        if not mime.startswith("image/"):
            continue
        if width < MIN_WIDTH:
            continue
        title = info.get("_title", "")
        slug = safe_slug(title.replace("File:", "").rsplit(".", 1)[0])
        name = f"wm_{slug}{ext}"
        dest = out_dir / name
        if dest.exists():
            continue
        body = download_bytes(session, url)
        if body == "RATE_LIMITED":
            print("  ! upstream demanded a long cooldown; stopping run.")
            break
        if body is None:
            time.sleep(2.0)
            continue
        if len(body) < MIN_BYTES:
            print(f"  ! too small ({len(body)} B) {title}")
            continue
        digest = hashlib.md5(body).hexdigest()
        if digest in existing_hashes:
            continue
        existing_hashes.add(digest)
        dest.write_bytes(body)
        saved += 1
        print(f"  [{saved}/{args.target}] {dest.name} ({len(body) // 1024} KB)")
        # Be polite to upload.wikimedia.org so we don't trip rate limits.
        time.sleep(3.0)

    print(f"\nDone. Saved {saved} new image(s) to {out_dir}.")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
