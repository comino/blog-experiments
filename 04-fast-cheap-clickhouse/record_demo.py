#!/usr/bin/env python3
"""Record a continuous LibreChat browser trial and retain all timing events."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
assert args.name.replace('-', '').isalnum()
OUT = BASE / 'evidence' / args.name
OUT.mkdir(exist_ok=False)
cases = json.loads((BASE / 'demo-cases.json').read_text())['cases']
record = {'started_at': datetime.now(timezone.utc).isoformat(), 'cases': [], 'errors': [],
          'browser': 'Playwright Chromium headless, SwiftShader software WebGL',
          'capture': '1440x1000, wall-clock speed, automated typing and clicks',
          'cache': 'fresh browser context; prefetch chart library before first prompt; no SQL result cache; warm DB/OS and model prefix cache possible'}
script = """(() => {
  window.demoEvents=[];window.activeCase=null;let busy=false;
  const add=d=>window.demoEvents.push({...d,case_id:window.activeCase,at:performance.now(),epoch:Date.now()});
  document.addEventListener('click',e=>{if(e.target.closest('button[aria-label="Send message"]'))add({type:'submit'})},true);
  window.addEventListener('message',e=>{if(['taxi-chart-ready','taxi-table-ready','taxi-local-action'].includes(e.data?.type))add(e.data)});
  new MutationObserver(()=>{const now=!!document.querySelector('[data-testid="stop-generation-button"]');if(now!==busy){busy=now;add({type:now?'generation-started':'generation-complete'})}}).observe(document,{childList:true,subtree:true});
})()"""
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    setup = browser.new_context()
    login_page = setup.new_page()
    login_page.goto('http://localhost:3080/login')
    credentials = json.loads((BASE/'private/login.json').read_text())['credentials']
    login_page.locator('#email').fill(credentials['email'])
    login_page.locator('#password').fill(credentials['login'])
    login_page.locator('button[type=submit]').click()
    login_page.wait_for_url('**/c/**')
    state = setup.storage_state()
    setup.close()
    del credentials
    record['recording_context_created_at'] = datetime.now(timezone.utc).isoformat()
    context = browser.new_context(storage_state=state,
        viewport={'width':1440,'height':1000}, device_scale_factor=1, color_scheme='dark',
        record_video_dir=str(OUT), record_video_size={'width':1440,'height':1000})
    context.add_init_script(script)
    page = context.new_page()
    page.on('pageerror', lambda error: record['errors'].append({'message':str(error),'stack':error.stack,'at':datetime.now(timezone.utc).isoformat()}))
    try:
        page.goto('http://localhost:3080/c/new')
        page.locator('#prompt-textarea').wait_for()
        # Model APIs and SQL are never mocked or replayed. Only static JS is prefetched.
        page.evaluate("async()=>{const r=await fetch('http://localhost:3081/assets/plotly-3.1.0.min.js');if(!r.ok)throw Error('Chart library unavailable');await r.arrayBuffer()}")
        # MCP is assigned to the model spec in librechat.yaml, not a transient picker.
        page.get_by_role('button', name='Select a model', exact=True).wait_for()
        if page.get_by_role('button',name='Close sidebar',exact=True).count():
            page.get_by_role('button',name='Close sidebar',exact=True).click()
        page.wait_for_timeout(1200)
        record['sequence_start_epoch'] = page.evaluate('Date.now()')
        for case in cases:
            page.evaluate('(id)=>window.activeCase=id', case['id'])
            page.locator('#prompt-textarea').press_sequentially(case['prompt'], delay=22)
            page.get_by_role('button', name='Send message', exact=True).click()
            page.wait_for_function("id=>window.demoEvents.some(e=>e.case_id===id&&e.type==='taxi-chart-ready')", arg=case['id'], timeout=45000)
            page.locator('[data-testid="stop-generation-button"]').wait_for(state='hidden',timeout=15000)
            frame_element = page.locator('iframe[title="MCP HTML Resource (Embedded Content)"]').last
            frame_element.scroll_into_view_if_needed()
            frame = frame_element.content_frame
            page.wait_for_timeout(1800)
            row = {'id':case['id'],'prompt':case['prompt'],'conversation_url':page.url,
                   'frame_box':frame_element.bounding_box(), 'body_text':page.locator('body').inner_text(),
                   'tool_id':frame.locator('body').get_attribute('data-chart-ready')}
            record['cases'].append(row)
            print('Rendered',case['id'],json.dumps(page.evaluate('window.demoEvents').copy()[-3:]),flush=True)
            if case['id']=='monthly':
                page.evaluate("()=>window.demoEvents.push({type:'local-table-click',case_id:window.activeCase,at:performance.now(),epoch:Date.now()})")
                frame.get_by_role('button',name='Table',exact=True).click()
                page.wait_for_timeout(1300)
                frame.locator('th').last.click()
                page.wait_for_timeout(900)
                frame.get_by_role('button',name='Chart',exact=True).click()
                page.wait_for_timeout(1000)
            if case['id']=='jfk':
                page.evaluate("()=>window.demoEvents.push({type:'local-3d-click',case_id:window.activeCase,at:performance.now(),epoch:Date.now()})")
                frame.get_by_role('button',name='3D view',exact=True).click()
                page.wait_for_function("()=>window.demoEvents.some(e=>e.type==='taxi-chart-ready'&&e.mode==='3d')",timeout=30000)
                page.wait_for_timeout(1200)
                box=frame.locator('#plot').bounding_box()
                page.mouse.move(box['x']+box['width']*.55,box['y']+box['height']*.5)
                page.mouse.down()
                page.mouse.move(box['x']+box['width']*.72,box['y']+box['height']*.6,steps=36)
                page.mouse.up()
                page.wait_for_timeout(1800)
        page.wait_for_timeout(1500)
        record['completed']=True
    except Exception as exc:
        record['completed']=False
        record['errors'].append(type(exc).__name__+': '+str(exc))
        page.screenshot(path=str(OUT/'failure.png'))
        print('Trial failed:',type(exc).__name__,str(exc)[:500],flush=True)
    finally:
        record['events']=page.evaluate('window.demoEvents || []')
        record['ended_at']=datetime.now(timezone.utc).isoformat()
        record['video']=Path(page.video.path()).name
        context.storage_state(path=str(BASE/'private/browser-state.json'))
        context.close()
        browser.close()
        (OUT/'recording.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Saved',str(OUT),flush=True)
if not record.get('completed'):
    raise SystemExit(1)
