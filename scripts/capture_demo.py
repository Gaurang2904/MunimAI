"""Capture screenshots of the running agent console for the deck.

    .venv\\Scripts\\streamlit run streamlit_app.py --server.headless true   (in another terminal)
    .venv\\Scripts\\python scripts\\capture_demo.py

Each shot types a real merchant message into the console, waits for the agent to finish, opens the
tool-call timeline and saves a PNG to docs/screens/.
"""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.scenarios import S  # noqa: E402

SHOTS = [("demo_hi_refund_due", "hi", "P06"), ("demo_mr_escalation", "mr", "E01")]


def main():
    out = os.path.join(ROOT, "docs", "screens")
    os.makedirs(out, exist_ok=True)
    msgs = {sid: m for sid, _, _, _, _, m in S}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
        for name, lang, sid in SHOTS:
            page.goto("http://localhost:8501/?lang=%s&state=%s" % (lang, sid))
            box = page.get_by_placeholder("Type as the merchant")
            box.wait_for(timeout=60000)
            box.fill(msgs[sid][lang])
            box.press("Enter")
            page.get_by_text("outcome:").first.wait_for(timeout=180000)
            page.get_by_text("outcome:").first.click()
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(out, name + ".png"), full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
