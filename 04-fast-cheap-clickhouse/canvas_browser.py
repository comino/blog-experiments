#!/usr/bin/env python3
"""Trusted local stdin workbench for the real public canvas. No replayed results."""
import json,sys,traceback,time
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=Path(__file__).resolve().parent
OUT=BASE/'performance'/'canvas-development'
OUT.mkdir(parents=True,exist_ok=True)
URL='https://'+json.loads((BASE/'public-hosting.json').read_text())['hostname']
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 context=browser.new_context(viewport={'width':1600,'height':1050},color_scheme='light',device_scale_factor=1)
 page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(URL+'/canvas/');page.locator('#login-screen').wait_for(state='visible')
 c=json.loads((BASE/'private/login.json').read_text())['credentials']
 page.locator('#email').fill(c['email']);page.locator('#password').fill(c['login']);page.locator('#login-form button[type=submit]').click();del c
 page.locator('#login-screen').wait_for(state='hidden',timeout=20000)
 page.screenshot(path=str(OUT/'empty.png'))
 print('READY '+page.url,flush=True)
 for line in sys.stdin:
  try:
   exec(json.loads(line)['code'])
   print('COMMAND COMPLETE',flush=True)
  except Exception:traceback.print_exc()
 context.close();browser.close()
