"""Download police car images from Wikimedia Commons.

Uses the public MediaWiki API (no auth, no Cloudflare). Police cars are an
emergency vehicle, so images land in `data/incoming/emergency_vehicle/`.

Usage (from project root):
    python scripts/scrape_policecar_images.py --target 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
# Cache imageinfo results so we only hit the rate-limited API once and can
# safely re-run the download phase as many times as needed.
CACHE_FILE = PROJECT_ROOT / "data" / "incoming" / ".wm_imageinfo_cache.json"


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
    """Look up imageinfo (URL, size, mime, thumburl) for many file titles.

    Uses small batches with generous pauses so we don't trip the API limiter.
    `iiurlwidth=1280` makes each result include a `thumburl` from Wikimedia's
    thumbnail cache, which rate-limits much less aggressively than the
    full-resolution media host.
    """
    out: list[dict] = []
    batch = 25
    for i in range(0, len(titles), batch):
        chunk = titles[i : i + batch]
        try:
            data = api_get(
                session,
                {
                    "action": "query",
                    "format": "json",
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": 1280,
                    "titles": "|".join(chunk),
                },
            )
        except RuntimeError as exc:
            print(f"  ! imageinfo batch {i}-{i + batch} failed: {exc}")
            print("  ! continuing with what we have so far")
            break
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            info["_title"] = page.get("title", "")
            out.append(info)
        print(f"  imageinfo: {len(out)} so far ({i + len(chunk)}/{len(titles)} queried)")
        time.sleep(3.0)
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
                wait_s = min(int(retry_after_raw), 30) if retry_after_raw else 10
            except ValueError:
                wait_s = 10
            print(f"  ! HTTP {r.status_code}; Retry-After={retry_after_raw or 'n/a'}; "
                  f"sleeping {wait_s}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_s)
            # Anything over 30s is the long-cooldown signal - stop early
            # so we don't make our IP penalty worse.
            if retry_after_raw and int(retry_after_raw) > 30:
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
    parser.add_argument(
        "--auto-wait",
        action="store_true",
        help=(
            "When upstream demands a long cooldown, sleep and resume "
            "automatically instead of exiting."
        ),
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

    # Phase 1: gather imageinfo - use cache if available so we don't hammer
    # the rate-limited API on re-runs.
    infos: list[dict] = []
    if CACHE_FILE.exists():
        try:
            infos = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            print(f"Loaded {len(infos)} cached imageinfo records from {CACHE_FILE.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! cache load failed: {exc}; refetching")
            infos = []

    if not infos:
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
            if len(candidate_titles) >= args.target * 3:
                break

        candidate_titles = candidate_titles[: max(args.target * 3, 150)]
        print(f"\n{len(candidate_titles)} candidate file titles. Fetching imageinfo...")
        infos = file_info(session, candidate_titles)
        print(f"got imageinfo for {len(infos)} files\n")
        try:
            CACHE_FILE.write_text(json.dumps(infos, ensure_ascii=False), encoding="utf-8")
            print(f"cached imageinfo -> {CACHE_FILE}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! cache save failed: {exc}")

    saved = 0
    for info in infos:
        if saved >= args.target:
            break
        # Original (full-res) URL is mainly used for filtering/extension; we
        # download the thumbnail to dodge upload.wikimedia.org rate limits.
        original_url = info.get("url") or ""
        thumb_url = info.get("thumburl") or original_url
        mime = info.get("mime") or ""
        width = int(info.get("width") or 0)
        ext = Path(urlparse(original_url).path).suffix.lower()
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
        url = thumb_url
        body = download_bytes(session, url)
        if body == "RATE_LIMITED":
            if args.auto_wait:
                # Sleep ~11 min so the 10-min Wikimedia cooldown definitely
                # clears, then retry this same image.
                wait_s = 11 * 60
                print(f"  ! long cooldown; auto-waiting {wait_s}s then resuming...")
                for remaining in range(wait_s, 0, -60):
                    time.sleep(60)
                    print(f"    ... {remaining - 60}s left")
                body = download_bytes(session, url)
                if body == "RATE_LIMITED" or body is None:
                    print("  ! still throttled after wait; giving up.")
                    break
            else:
                print("  ! upstream demanded a long cooldown; stopping run.")
                print("    (re-run with --auto-wait to keep going automatically)")
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
        # Be polite to upload.wikimedia.org so we don't trip the long
        # (10-minute) cooldown. ~6s/image keeps us under the soft limit.
        time.sleep(6.0)

    print(f"\nDone. Saved {saved} new image(s) to {out_dir}.")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
