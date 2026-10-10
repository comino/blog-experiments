#!/usr/bin/env python3
"""Normal-speed recording of live wildlife queries in the public canvas."""
import json,time,argparse,re
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--name',default='canvas-recording-02');args=parser.parse_args();assert re.fullmatch(r'canvas-recording-[0-9]{2}',args.name)
OUT=BASE/'performance'/args.name;OUT.mkdir(exist_ok=False)
URL='https://'+json.loads((BASE/'public-hosting.json').read_text())['hostname']
CASES=[
 ('map-2025','Map monarch butterfly observations worldwide in 2025.'),
 ('map-2024','Only 2024, keep the same map.'),
 ('seasons','Add a heatmap of monarch observations in 2024 by month and 10-degree latitude band.'),
 ('photos','Add photos of monarch butterflies observed in 2024 alongside these views.')]
report={'started_at':datetime.now(timezone.utc).isoformat(),'url':URL+'/canvas/','cases':[],'errors':[], 'method':'Live OpenRouter and ClickHouse, normal-speed browser capture. Login and workspace reset before recording. Browser assets may be cached; no model or SQL result replay. Dates and species are prepared demonstration prompts, not a blind benchmark.'}
AUTH="""async()=>{const r=await fetch('/api/auth/refresh',{method:'POST'});if(!r.ok)throw Error('Not signed in');const {token}=await r.json();const headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'};return {headers};}"""
def workspace(page):
 return page.evaluate("""async()=>{const {token}=await(await fetch('/api/auth/refresh',{method:'POST'})).json();return await(await fetch('/canvas/api/workspace',{headers:{'Authorization':'Bearer '+token}})).json();}""")
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 setup=browser.new_context(storage_state=str(BASE/'private/canvas-browser-state.json'))
 page=setup.new_page();page.goto(URL+'/canvas/')
 page.wait_for_function("()=>!document.querySelector('#login-screen').hidden||document.querySelector('#save-status').textContent==='Saved'",timeout=20000)
 if page.locator('#login-screen').is_visible():
  credentials=json.loads((BASE/'private/login.json').read_text())['credentials']
  page.locator('#email').fill(credentials['email']);page.locator('#password').fill(credentials['login']);page.locator('#login-form button[type=submit]').click();del credentials
  page.locator('#login-screen').wait_for(state='hidden',timeout=20000)

 # This is the experiment account, separate from the user's demo account.
 page.evaluate("""async()=>{const {token}=await(await fetch('/api/auth/refresh',{method:'POST'})).json();const headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'};const user=await(await fetch('/api/user',{headers})).json();if(user.email!=='explorer@example.invalid')throw Error('Not the test account');const s=await(await fetch('/canvas/api/workspace',{headers})).json();const r=await fetch('/canvas/api/workspace',{method:'PUT',headers,body:JSON.stringify({cards:[],messages:[],undo:[],redo:[],version:s.version,name:'Life, all around us'})});if(!r.ok)throw Error('Reset failed');}""")
 storage=setup.storage_state();setup.close()
 context=browser.new_context(storage_state=storage,viewport={'width':1600,'height':1050},color_scheme='light',record_video_dir=str(OUT),record_video_size={'width':1600,'height':1050})
 page=context.new_page();page.on('pageerror',lambda e:report['errors'].append(str(e)))
 page.goto(URL+'/canvas/');page.locator('#save-status').filter(has_text='Saved').wait_for(timeout=20000);page.wait_for_timeout(800)
 for name,prompt in CASES:
  n=page.evaluate("window.canvasTelemetry.filter(e=>e.type==='paint').length")
  page.locator('#prompt').press_sequentially(prompt,delay=22);page.locator('#send').click()
  page.wait_for_function("n=>window.canvasTelemetry.filter(e=>e.type==='paint').length>n || window.canvasTelemetry.some(e=>e.type==='request-error')",arg=n,timeout=60000)
  events=page.evaluate('window.canvasTelemetry');assert not any(e['type']=='request-error' for e in events),events
  current=workspace(page);row={'id':name,'prompt':prompt,'event':[e for e in events if e['type']=='paint'][-1],'workspace':current}
  report['cases'].append(row);(OUT/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'id':name,'elapsed_ms':row['event']['elapsed_ms'],'model_ms':row['event']['model_ms'],'query_ms':row['event']['query_ms']}),flush=True)
  page.wait_for_timeout(1100);page.screenshot(path=str(OUT/(name+'.png')))
 # Select the heatmap. Representation and color edits need no inference or SQL.
 heatmap=page.locator('.data-card').filter(has=page.locator('select option:checked[value="heatmap"]'))
 assert heatmap.count()==1
 heatmap=page.locator('.data-card[data-id="'+heatmap.get_attribute('data-id')+'"]')
 heatmap.locator('.card-title').click()
 page.locator('#prompt').press_sequentially('make it 3d',delay=55);page.locator('#send').click()
 page.wait_for_function("()=>window.canvasTelemetry.some(e=>e.type==='local-request')",timeout=10000);page.wait_for_timeout(400)
 canvas=heatmap.locator('.card-body canvas');box=canvas.bounding_box();page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.down();page.mouse.move(box['x']+box['width']/2+80,box['y']+box['height']/2+15,steps=24);page.mouse.up();page.wait_for_timeout(500)
 page.screenshot(path=str(OUT/'canvas-3d.png'))
 page.locator('#undo').click();page.wait_for_timeout(500)
 page.locator('#prompt').press_sequentially('make it blue',delay=55);page.locator('#send').click();page.wait_for_function("()=>window.canvasTelemetry.filter(e=>e.type==='local-request').length===2",timeout=10000);page.wait_for_timeout(600)
 page.locator('.photo-open').first.click();page.locator('.photo-dialog').wait_for();page.wait_for_timeout(1400);page.screenshot(path=str(OUT/'photo-detail.png'));page.locator('.photo-close').click();page.wait_for_timeout(700)
 page.screenshot(path=str(OUT/'final-canvas.png'))
 report['events']=page.evaluate('window.canvasTelemetry');report['final_workspace']=workspace(page);report['ended_at']=datetime.now(timezone.utc).isoformat()
 (OUT/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
 video=page.video;context.close();video.save_as(str(OUT/'wildlife-canvas.webm'));browser.close()
 assert not report['errors'],report['errors']
 print('Recording complete',flush=True)
