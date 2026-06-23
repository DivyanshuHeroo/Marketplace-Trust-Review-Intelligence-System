"""Capture dashboard screenshots for the README.

Drives the running Streamlit app (http://localhost:8501) with a headless browser,
clicks through the key tabs, and saves PNGs into reports/screenshots/.

Usage
-----
    # 1. start the app in another terminal:
    streamlit run app/dashboard.py
    # 2. then:
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

# (tab label substring, output filename)
TABS = [
    ("Overview", "01_overview.png"),
    ("Delivery Insights", "02_delivery.png"),
    ("Seller Trust", "03_trust.png"),
    ("Risk Predictor", "04_risk.png"),
]


def _settle(page, ms: int = 2500) -> None:
    """Give Streamlit time to finish rendering charts after an interaction."""
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        _settle(page, 4000)

        for label, fname in TABS:
            tab = page.get_by_role("tab", name=label)
            tab.click()
            _settle(page)
            page.screenshot(path=str(OUT / fname), full_page=True)
            print(f"  saved {fname}")

        browser.close()
    print(f"\nScreenshots written to {OUT}")


if __name__ == "__main__":
    main()
