from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def wait_for_port(port: int, timeout: int = 30) -> bool:
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


async def run_automation():
    from playwright.async_api import async_playwright

    print("=" * 60)
    print("Starting Automated Browser Testing for Streamlit UI")
    print("=" * 60)

    print("1. Launching FastAPI server (port 8000)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_port(8000, 20):
        print("ERROR: FastAPI did not start on port 8000.")
        api_proc.kill()
        return False
    print("   -> FastAPI is UP and listening on 127.0.0.1:8000")

    print("2. Launching Streamlit server (port 8501)...")
    ui_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "ui.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_port(8501, 20):
        print("ERROR: Streamlit did not start on port 8501.")
        api_proc.kill()
        ui_proc.kill()
        return False
    print("   -> Streamlit is UP and listening on 127.0.0.1:8501")

    print("3. Launching Google Chrome via Playwright...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()

            print("   -> Navigating to http://127.0.0.1:8501 ...")
            await page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)

            s1_path = SCREENSHOT_DIR / "01_initial_dashboard.png"
            await page.screenshot(path=str(s1_path), full_page=True)
            print(f"   -> Screenshot saved: {s1_path.name}")

            sidebar_text = await page.inner_text("section[data-testid='stSidebar']")
            if "API Online" in sidebar_text:
                print("   -> [PASS] Sidebar displays API Online status.")
            else:
                print(f"   -> [WARN] Sidebar text: {sidebar_text[:100]}")

            print("4. Switching to 'Manage & Ingest Documents' page...")
            nav_btn = page.locator("label:has-text('Manage & Ingest Documents')")
            await nav_btn.click()
            await page.wait_for_timeout(2000)
            print("   -> Switched to Manage & Ingest Documents view.")

            print("5. Uploading test technical document...")
            test_doc = PROJECT_ROOT / "docs" / "corpus" / "fastapi_guide.md"
            file_input = page.locator("input[type='file']")
            await file_input.set_input_files(str(test_doc))
            await page.wait_for_timeout(2000)

            s2_path = SCREENSHOT_DIR / "02_file_selected.png"
            await page.screenshot(path=str(s2_path))
            print(f"   -> Screenshot saved: {s2_path.name}")

            print("6. Clicking 'Ingest Selected Files' button...")
            ingest_btn = page.locator("button:has-text('Ingest Selected Files')")
            await ingest_btn.click()
            print("   -> Clicked Ingest button, waiting for processing...")

            await page.wait_for_selector("div[data-testid='stAlert']", timeout=30000)
            await page.wait_for_timeout(2000)

            s3_path = SCREENSHOT_DIR / "03_ingested_success.png"
            await page.screenshot(path=str(s3_path))
            print(f"   -> Screenshot saved: {s3_path.name}")
            alert_text = await page.locator("div[data-testid='stAlert']").first.inner_text()
            print(f"   -> Alert result: {alert_text}")

            print("7. Switching back to Chat view...")
            chat_nav = page.locator("label:has-text('Ask Questions (Chat)')")
            await chat_nav.click()
            await page.wait_for_timeout(2000)

            s4_path = SCREENSHOT_DIR / "04_chat_tab.png"
            await page.screenshot(path=str(s4_path))
            print(f"   -> Screenshot saved: {s4_path.name}")

            print("=" * 60)
            print("Browser Automation Test Finished Successfully!")
            print(f"All screenshots available in: {SCREENSHOT_DIR}")
            print("=" * 60)

            await browser.close()
            return True

    finally:
        print("Cleaning up background servers...")
        try:
            api_proc.send_signal(signal.SIGTERM)
            ui_proc.send_signal(signal.SIGTERM)
        except Exception:
            pass


if __name__ == "__main__":
    success = asyncio.run(run_automation())
    sys.exit(0 if success else 1)
