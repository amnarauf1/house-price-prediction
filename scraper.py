"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  FILE: scraper.py  |  Part 1 — Web Scraper                                  ║
║  Scrapes property listings from Zameen.com                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  USAGE (standalone):
    python scraper.py

  OUTPUT:
    dataset/real_estate_dataset.csv
"""

import os
import re
import time
import random
import logging

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.zameen.com"
SEARCH_URL      = "https://www.zameen.com/Homes/Islamabad-3-1.html"
TARGET_LISTINGS = 350
MAX_PAGES       = 25
DELAY_MIN       = 2.0
DELAY_MAX       = 4.0
RAW_CSV         = "dataset/real_estate_dataset.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

os.makedirs("dataset", exist_ok=True)

log = logging.getLogger(__name__)

# Patterns that are NEVER valid property URLs
_INVALID_URL_PATTERNS = [
    "mailto:", "javascript:", "facebook.com", "twitter.com",
    "linkedin.com", "instagram.com", "youtube.com", "whatsapp.com",
    "pinterest.com", "tiktok.com", "sharer.php", "share?", "tel:",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _is_valid_property_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    if not u.startswith("http"):
        return False
    if "zameen.com" not in u:
        return False
    if any(p in u for p in _INVALID_URL_PATTERNS):
        return False
    if "/property/" not in u:
        return False
    return True


def _polite_get(session, url, retries=3):
    u_lower = url.lower()
    if any(p in u_lower for p in _INVALID_URL_PATTERNS):
        log.warning(f"[Guard] Blocked invalid URL: {url[:80]}")
        return None
    if not u_lower.startswith("http"):
        log.warning(f"[Guard] Skipping non-http URL: {url[:80]}")
        return None

    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                log.warning("Rate-limited. Waiting 30 s …")
                time.sleep(30)
            elif r.status_code in (403, 503):
                log.warning(f"Blocked ({r.status_code}). Stopping gracefully.")
                return None
            else:
                log.warning(f"HTTP {r.status_code} on attempt {attempt}")
        except requests.RequestException as e:
            log.error(f"Request error: {e}")
        time.sleep(5 * attempt)
    return None


def _listing_urls_from_page(soup):
    urls = set()
    for a in soup.find_all("a", href=True):
        try:
            h = a["href"].strip()
        except Exception:
            continue
        if not h:
            continue
        h_lower = h.lower()
        if any(p in h_lower for p in _INVALID_URL_PATTERNS):
            continue
        if not h_lower.startswith("http") and not h_lower.startswith("/"):
            continue
        if "/property/" not in h_lower:
            continue
        full = h if h.startswith("http") else BASE_URL + h
        full = full.split("#")[0]
        if "zameen.com" not in full.lower():
            continue
        full = re.sub(r"(?<!:)/{2,}", "/", full)
        if not _is_valid_property_url(full):
            continue
        urls.add(full)
    return list(urls)


def _next_page(soup, current):
    for sel in ['a[aria-label="Next"]', 'a[rel="next"]', 'a[class*="next"]']:
        tag = soup.select_one(sel)
        if tag and tag.get("href"):
            h = tag["href"]
            return h if h.startswith("http") else BASE_URL + h
    m = re.search(r"-(\d+)\.html$", current)
    if m:
        nxt = int(m.group(1)) + 1
        return re.sub(r"-\d+\.html$", f"-{nxt}.html", current)
    return None


def _parse_detail(session, url):
    rec = {k: None for k in [
        "URL", "Price", "Area", "City", "Bedrooms", "Bathrooms", "Location",
        "Property_Type", "Built_in_Year", "Parking_Spaces", "Servant_Quarters",
        "Store_Rooms", "Kitchens", "Drawing_Rooms"
    ]}
    rec["URL"]  = url
    rec["City"] = "Islamabad"

    if not _is_valid_property_url(url):
        log.warning(f"[Skip] Invalid URL: {url[:80]}")
        return rec

    resp = _polite_get(session, url)
    if resp is None:
        return rec

    soup = BeautifulSoup(resp.text, "html.parser")

    for sel in ['[aria-label="Price"]', 'span[class*="price"]', 'div[class*="price"]']:
        t = soup.select_one(sel)
        if t:
            rec["Price"] = t.get_text(strip=True); break

    for sel in ['[aria-label="Area"]', 'span[class*="area"]']:
        t = soup.select_one(sel)
        if t:
            rec["Area"] = t.get_text(strip=True); break

    for sel in ['[aria-label="Location"]', 'div[class*="location"]',
                'span[class*="location"]', 'nav[aria-label="breadcrumb"]']:
        t = soup.select_one(sel)
        if t:
            rec["Location"] = t.get_text(separator=" › ", strip=True); break

    for sel in ['[aria-label="Type"]', 'div[class*="type"]']:
        t = soup.select_one(sel)
        if t:
            rec["Property_Type"] = t.get_text(strip=True); break

    fmap = {
        "bedrooms":         "Bedrooms",
        "bathrooms":        "Bathrooms",
        "built in year":    "Built_in_Year",
        "parking spaces":   "Parking_Spaces",
        "servant quarters": "Servant_Quarters",
        "store rooms":      "Store_Rooms",
        "kitchens":         "Kitchens",
        "drawing rooms":    "Drawing_Rooms",
    }
    for item in soup.find_all(["li", "tr"]):
        text = item.get_text(separator="|", strip=True).lower()
        for key, col in fmap.items():
            if key in text:
                parts = [p.strip() for p in item.get_text(separator="|", strip=True).split("|") if p.strip()]
                if len(parts) >= 2:
                    rec[col] = parts[-1]
    for dt in soup.find_all("dt"):
        label = dt.get_text(strip=True).lower()
        dd = dt.find_next_sibling("dd")
        if dd:
            for key, col in fmap.items():
                if key in label:
                    rec[col] = dd.get_text(strip=True)
    return rec


# ── MAIN SCRAPER ──────────────────────────────────────────────────────────────

def run_scraper(target=TARGET_LISTINGS):
    print("\n╔" + "═"*64 + "╗")
    print("║  PART 1 — WEB SCRAPER" + " "*42 + "║")
    print("╚" + "═"*64 + "╝")

    session = requests.Session()
    session.headers.update(HEADERS)

    all_urls = []
    page_url = SEARCH_URL
    page_num = 0

    while len(all_urls) < target and page_num < MAX_PAGES:
        page_num += 1
        log.info(f"[Scraper] Page {page_num}: {page_url}")
        r = _polite_get(session, page_url)
        if r is None:
            break
        soup     = BeautifulSoup(r.text, "html.parser")
        new_urls = [u for u in _listing_urls_from_page(soup) if u not in all_urls]
        all_urls.extend(new_urls)
        log.info(f"          +{len(new_urls)} URLs  (total {len(all_urls)})")
        nxt = _next_page(soup, page_url)
        if not nxt or nxt == page_url:
            break
        page_url = nxt

    log.info(f"[Scraper] Collected {len(all_urls)} URLs. Now visiting detail pages …")

    records, seen = [], set()
    for i, url in enumerate(all_urls[:target], 1):
        if url in seen:
            continue
        seen.add(url)
        log.info(f"[Scraper] [{i}/{min(len(all_urls), target)}] {url[:80]}")
        records.append(_parse_detail(session, url))
        if i % 50 == 0:
            pd.DataFrame(records).to_csv(RAW_CSV, index=False)
            log.info(f"          ↳ Checkpoint saved ({i} rows)")

    df = pd.DataFrame(records)
    df.to_csv(RAW_CSV, index=False)
    log.info(f"[Scraper] Done — {len(df)} listings saved → {RAW_CSV}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    run_scraper()
