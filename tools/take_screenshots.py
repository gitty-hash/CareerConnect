"""Capture the report's screenshots by driving the running app in headless Chromium.

Usage:
    python tools/take_screenshots.py live  http://localhost:8502   # app started WITH OPENAI_API_KEY
    python tools/take_screenshots.py demo  http://localhost:8501   # app in demo (cached) mode
Writes PNGs to report/figures/.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def wait_idle(page, settle=2.0, timeout=240):
    """Wait until Streamlit has finished re-running the script."""
    time.sleep(1.5)
    end = time.time() + timeout
    while time.time() < end:
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if not running:
            break
        time.sleep(0.5)
    time.sleep(settle)


def open_app(p, url):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
    page.goto(url)
    page.wait_for_selector(".cc-topbar", timeout=300_000)
    wait_idle(page, settle=3)
    return browser, page


def tab(page, name):
    page.get_by_role("tab").filter(has_text=name).first.click()
    time.sleep(1.5)


def live(url):
    with sync_playwright() as p:
        browser, page = open_app(p, url)

        # L1 - Jobs: a new free-text query, never recorded -> live few-shot filter extraction.
        box = page.locator(".st-key-card_search input").first
        box.fill("paid remote finance internship for an international student")
        box.press("Enter")
        page.locator(".st-key-card_detail").filter(has_text="Finance").wait_for(timeout=180_000)
        wait_idle(page, settle=3)
        top_at(page, page.get_by_text("AI understood your search as").first, -20)
        page.screenshot(path=str(OUT / "live_1_jobs.png"))

        # L2 - Coach: a new question, answered live with citations and a fact-check badge.
        tab(page, "Career Coach")
        q = page.locator(".st-key-coach_input input").first
        q.fill("How should I follow up with a recruiter after the career fair?")
        page.get_by_role("button", name="Ask", exact=True).click()
        page.wait_for_selector(".badge", timeout=180_000)
        wait_idle(page, settle=3)
        top_at(page, page.locator('[data-testid="stChatMessage"]').nth(1), -10)
        page.screenshot(path=str(OUT / "live_2_coach.png"))

        # L3 - Employers: a new pasted draft -> live extraction, flags, judge, rewrite.
        tab(page, "For Employers")
        draft = page.locator(".st-key-draft_0 textarea").first
        draft.fill("Marketing Intern - Sunrise Coffee Co. (Phoenix, AZ)\nWe want a young, energetic social "
                   "media guru to run our Instagram and TikTok. Must have 2+ years of agency experience and "
                   "be a U.S. citizen. Unpaid, 30 hours/week, onsite in Phoenix.")
        draft.press("Control+Enter")
        # Done only when the constraints show the NEW draft and the rewrite exists.
        page.locator(".st-key-card_emp_cons").filter(has_text="Marketing").wait_for(timeout=240_000)
        page.wait_for_function(
            "() => [...document.querySelectorAll('textarea')].filter(t => t.value.includes('Sunrise')).length >= 2",
            timeout=240_000)
        wait_idle(page, settle=4)
        top_at(page, page.locator(".kpi").first, -30)
        page.screenshot(path=str(OUT / "live_3_employer.png"))
        browser.close()


def top_at(page, locator, offset):
    """Scroll so the element sits at the top of the viewport (plus an offset)."""
    locator.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.wheel(0, offset)
    time.sleep(1.2)


def demo(url):
    with sync_playwright() as p:
        browser, page = open_app(p, url)
        # Semantic map for the default query (Part A view).
        page.get_by_role("tab").filter(has_text="Job map").first.click()
        time.sleep(3)
        page.locator(".st-key-card_map").screenshot(path=str(OUT / "fig_semantic_map.png"))
        # Insights: KPIs + baseline-vs-upgrades chart.
        tab(page, "Insights")
        time.sleep(2)
        page.locator(".st-key-card_eval_chart").screenshot(path=str(OUT / "fig_eval_chart.png"))
        browser.close()


if __name__ == "__main__":
    {"live": live, "demo": demo}[sys.argv[1]](sys.argv[2])
    print("saved to", OUT)
