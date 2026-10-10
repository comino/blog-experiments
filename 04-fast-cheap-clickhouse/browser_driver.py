#!/usr/bin/env python3
"""Local-only Playwright workbench; commands on stdin are trusted Python code.

Keep authentication state under ignored private/. Do not put this on a network.
Final recordings use a fresh browser context after login, with recorded wall time.
"""
import json
from pathlib import Path
import sys
import traceback
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1, color_scheme='dark')
    page = context.new_page()
    page.goto('http://localhost:3080/login')
    credentials = json.loads((BASE / 'private/login.json').read_text())['credentials']
    page.locator('#email').fill(credentials['email'])
    page.locator('#password').fill(credentials['login'])
    page.locator('button[type=submit]').click()
    page.wait_for_url('**/c/**', timeout=30000)
    context.storage_state(path=str(BASE / 'private/browser-state.json'))
    del credentials
    print('Browser ready at', page.url, flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            exec(command['code'])
            print('COMMAND COMPLETE', flush=True)
        except Exception:
            traceback.print_exc()
    context.close()
    browser.close()
