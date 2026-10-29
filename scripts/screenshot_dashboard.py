"""Render the three dashboard pages headlessly and save PNGs to reports/figures/ (smoke test of the app)."""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
port = 8599
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
                         "--server.headless", "true", "--server.port", str(port)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(8)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 1500})
        pg.goto(f"http://localhost:{port}", wait_until="networkidle")
        for i, name in enumerate(["Portfolio monitor", "Loan detail", "Model health"]):
            if i:
                pg.get_by_text(name, exact=True).click()
            pg.wait_for_timeout(9000)
            errs = pg.locator("[data-testid='stException']").count()
            pg.screenshot(path=str(ROOT / "reports" / "figures" / f"dashboard_{i + 1}_{name.split()[0].lower()}.png"),
                          full_page=True)
            print(name, "exceptions on page:", errs)
        b.close()
finally:
    proc.terminate()
