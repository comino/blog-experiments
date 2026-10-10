#!/usr/bin/env python3
"""Rehearse sequential demo prompts against the real deployed canvas.

Creates isolated QA chats, records each story at normal speed, keeps failed
attempts, and restores the originally active QA session. No response replay.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import time

from playwright.sync_api import sync_playwright

B = Path(__file__).resolve().parent


def interrupted(signum, frame):
    raise KeyboardInterrupt('Rehearsal interrupted by signal '+str(signum))


def api(page, path, body=None):
    return page.evaluate('''async ({path,body})=>{
      const r=await fetch('/api/auth/refresh',{method:'POST'});
      if(!r.ok)throw Error('QA sign-in expired');
      const {token}=await r.json(),headers={Authorization:'Bearer '+token,'Content-Type':'application/json'};
      const user=await(await fetch('/api/user',{headers})).json();
      if(user.email!=='explorer@example.invalid')throw Error('Only the separate QA account is permitted');
      const response=await fetch('/canvas/api/'+path,{headers,...(body?{method:'POST',body:JSON.stringify(body)}:{})});
      if(!response.ok)throw Error(path+' '+response.status+' '+await response.text());
      return response.json();
    }''', {'path':path,'body':body})


def idle(page):
    page.wait_for_function('''()=>document.querySelector('#login-screen').hidden
      &&document.querySelector('#save-status').textContent==='Saved'
      &&!document.querySelector('#send').disabled&&document.querySelector('#board').dataset.framing!=='true'
      &&!document.querySelector('.blueprint,.materializing')
      &&![...document.querySelectorAll('.data-card')].some(n=>n.getAnimations().some(a=>a.playState==='running'))''', timeout=90000)
    page.wait_for_timeout(250)


def select(page, card, focus=False):
    if focus:
        page.get_by_role('button', name='Focus '+card['title'], exact=True).last.click()
        idle(page)
    else:
        page.locator(f'[data-id="{card["id"]}"] .card-title').click()


def mark_peak(page, card, spec):
    """Find a real heatmap/map cell, click it, and verify the marked row."""
    view = page.locator(f'[data-id="{card["id"]}"]')
    select(page, card, focus=True)
    columns, rows = card['data']['columns'], card['data']['rows']
    xi, yi, zi = (columns.index(card[k]) for k in ('x','y','z'))
    candidates = [(i,row) for i,row in enumerate(rows)
                  if all(row[columns.index(key)] == value for key,value in spec.get('where',{}).items())]
    if spec.get('x') is not None:
        candidates = [(i,row) for i,row in candidates if row[xi] == spec['x']]
    if spec.get('y') is not None:
        candidates = [(i,row) for i,row in candidates if row[yi] == spec['y']]
    assert candidates, 'No real cell matches the requested mark'
    index, row = max(candidates, key=lambda pair:pair[1][zi])
    marker=view.get_by_role('button',name='Mark data to ask a follow-up',exact=True)
    if card['chart']=='density':
        if marker.get_attribute('aria-pressed')=='true':marker.click()
        # Locate the cell through the same hover tooltip a person uses. This
        # does not access private Leaflet state or substitute any query data.
        point=view.locator('.explore-map').evaluate('''(map,axes)=>{
          const r=map.getBoundingClientRect(),tip=map.parentElement.querySelector('.explore-tooltip');
          const wanted=Number(axes[1]).toFixed(3)+'°, '+Number(axes[0]).toFixed(3)+'°';
          for(let y=48;y<map.clientHeight-55;y+=12)for(let x=28;x<map.clientWidth-28;x+=12){
            const clientX=r.left+x*r.width/map.clientWidth,clientY=r.top+y*r.height/map.clientHeight;
            map.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX,clientY}));
            if(!tip.hidden&&tip.textContent.includes(wanted))return {x:clientX,y:clientY};
          }
          throw Error('Requested density cell was not found by hover');
        }''',[row[xi],row[yi]])
        page.mouse.move(point['x'],point['y']);page.wait_for_timeout(80)
        assert f'{row[yi]:.3f}°, {row[xi]:.3f}°' in view.locator('.explore-tooltip').text_content()
    else:
        point = view.locator('.plot').evaluate('''(p,axes)=>{
          const r=p.getBoundingClientRect(),l=p._fullLayout;
          return {x:r.left+(l.xaxis._offset+l.xaxis.d2p(axes[0]))*r.width/p.clientWidth,
                  y:r.top+(l.yaxis._offset+l.yaxis.d2p(axes[1]))*r.height/p.clientHeight};
        }''', [row[xi],row[yi]])
    count=page.evaluate('window.canvasTelemetry.length')
    if card['chart']!='density' and marker.get_attribute('aria-pressed')!='true':marker.click()
    page.mouse.click(point['x'],point['y'])
    assert page.locator('#selection-chip').is_visible()
    marks=page.evaluate("n=>window.canvasTelemetry.slice(n).filter(e=>e.type==='mark')",count)
    assert marks and marks[-1]['selection']=={'card_id':card['id'],'revision':card['revision'],'kind':'points','row_indices':[index]},marks
    return {'index':index,'columns':columns,'row':row}


def validate_card(card, step):
    expected = step.get('expect',{})
    if expected.get('chart'):
        assert card['chart'] == expected['chart'], (card['chart'],expected['chart'])
    rows, columns = card['data']['rows'], card['data']['columns']
    assert len(rows) >= expected.get('min_rows',1), ('Too few rows',len(rows))
    assert len(rows) <= expected.get('max_rows',1000), ('Too many rows',len(rows))
    for column in expected.get('columns',[]):
        assert column in columns, (column,columns)
    if expected.get('exact_rows') is not None:
        assert len(rows) == expected['exact_rows'], (len(rows),expected['exact_rows'])
    for column, values in expected.get('sets',{}).items():
        assert set(row[columns.index(column)] for row in rows) == set(values)
    if expected.get('numeric_z'):
        assert card['z'] in columns
        assert all(isinstance(row[columns.index(card['z'])],(int,float)) for row in rows)
    return {'chart':card['chart'],'rows':len(rows),'columns':columns}


def action(page, card, operation):
    kind = operation['action']
    view = page.locator(f'[data-id="{card["id"]}"]')
    if kind == 'play':
        slider = view.get_by_role('slider',name='Track time in UTC')
        slider.wait_for(state='visible',timeout=30000)
        if 'position' in operation:
            slider.evaluate('(n,v)=>{n.value=String(v);n.dispatchEvent(new Event("input",{bubbles:true}));}',operation['position'])
        before = view.locator('.explore-clock').text_content()
        view.get_by_role('button',name='Play measured tracks',exact=True).click()
        page.wait_for_timeout(operation.get('ms',2500))
        after = view.locator('.explore-clock').text_content()
        pause=view.get_by_role('button',name='Pause measured tracks',exact=True)
        if pause.count():pause.click()
        assert before != after
        return {'action':kind,'from':before,'to':after}
    if kind == 'satellite':
        view.get_by_role('button',name='Toggle daily NASA Aqua / MODIS imagery',exact=True).click()
        day=view.locator('.explore-playback').evaluate("n=>new Date(Number(n.dataset.time)).toISOString().slice(0,10)")
        page.wait_for_function('''({id,day})=>[...document.querySelectorAll('[data-id="'+id+'"] .leaflet-tile')].some(n=>n.src.includes('gibs.earthdata.nasa.gov')&&n.src.includes('/'+day+'/')&&n.complete&&n.naturalWidth>0)''',arg={'id':card['id'],'day':day},timeout=30000)
        page.wait_for_timeout(operation.get('ms',1200))
        return {'action':kind,'dated_tiles_loaded':True,'utc_date':day}
    if kind == 'scrub':
        slider=view.get_by_role('slider',name='Track time in UTC')
        slider.evaluate('(n,v)=>{n.value=String(v);n.dispatchEvent(new Event("input",{bubbles:true}));}',operation['position'])
        page.wait_for_timeout(450)
        moment=view.locator('.explore-playback').evaluate('n=>Number(n.dataset.time)')
        cursors=page.locator('.plot').evaluate_all('''plots=>plots.filter(p=>p._fullLayout?.xaxis?.type==='date').map(p=>({card_id:p.closest('[data-id]').dataset.id,time:Date.parse(p.layout.shapes?.[0]?.x0)}))''')
        assert cursors and all(c['time'] is not None and abs(c['time']-moment)<2 for c in cursors),cursors
        page.wait_for_timeout(operation.get('ms',1200))
        return {'action':kind,'utc_epoch_ms':moment,'linked_cursors':cursors}
    if kind == 'orbit':
        button = view.get_by_role('button',name='Play or pause 3D orbit',exact=True)
        surface=view.locator('.card-body canvas').first
        before=surface.evaluate('n=>n.toDataURL()')
        button.click();assert button.get_attribute('aria-pressed')=='true'
        page.wait_for_timeout(operation.get('ms',2200))
        after=surface.evaluate('n=>n.toDataURL()')
        button.click();assert button.get_attribute('aria-pressed')=='false'
        assert before!=after,'Orbit did not change the rendered pixels'
        return {'action':kind,'pixels_changed':True,'stopped':True}
    if kind == 'photo':
        view.locator('.photo-tile.loaded .photo-open').nth(operation.get('index',0)).click()
        page.locator('.photo-dialog').wait_for(state='visible')
        page.wait_for_function("()=>{const n=document.querySelector('.photo-dialog>img');return n?.complete&&n.naturalWidth>0}",timeout=30000)
        page.wait_for_timeout(operation.get('ms',2400))
        page.locator('.photo-close').click()
        return {'action':kind,'index':operation.get('index',0)}
    if kind == 'season_group':
        view.get_by_label('Seasonal group').select_option(label=operation['label'])
        page.wait_for_timeout(900)
        return {'action':kind,'label':operation['label']}
    if kind == 'focus':
        select(page,card,focus=True)
        return {'action':kind}
    if kind == 'fit':
        page.locator('#fit').click(); idle(page)
        return {'action':kind}
    if kind == 'clear_mark':
        if page.locator('#selection-chip').is_visible():page.locator('#clear-mark').click()
        return {'action':kind}
    raise ValueError('Unknown story action: '+kind)


def main():
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--spec',default='demos/stories.json')
    parser.add_argument('--stories',nargs='*')
    args=parser.parse_args()
    assert re.fullmatch(r'story-rehearsal-\d{2}',args.name)
    output=B/'performance'/args.name;output.mkdir(exist_ok=False)
    spec=json.loads((B/args.spec).read_text())
    stories=[s for s in spec['stories'] if not args.stories or s['id'] in args.stories]
    assert stories
    base='https://'+json.loads((B/'public-hosting.json').read_text())['hostname']
    report={'started_at':datetime.now(timezone.utc).isoformat(),'stories':[],'errors':[],
            'method':'Exact prepared prompt sequences in new QA chats on the deployed app. Live pinned OpenRouter model and ClickHouse; no mocked, intercepted, replayed or manually substituted model results. Videos run at original speed. Read-only reference checks are recorded separately.'}
    (output/'stories-tested.json').write_text(json.dumps(spec,indent=2)+'\n')
    original=None
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        setup=browser.new_context(viewport={'width':1600,'height':1000})
        home=setup.new_page();home.goto(base+'/canvas/')
        home.locator('#login-screen').wait_for(state='visible')
        creds=json.loads((B/'private/login.json').read_text())['credentials'];assert creds['email']=='explorer@example.invalid'
        home.locator('#email').fill(creds['email']);home.locator('#password').fill(creds['login']);del creds
        home.locator('#login-form button').click();idle(home)
        original=api(home,'workspace')
        (output/'original-workspace.json').write_text(json.dumps(original,indent=2)+'\n')
        report['original_session_id']=original['session_id']
        context=browser.new_context(storage_state=setup.storage_state(),viewport={'width':1600,'height':1000},
                                    record_video_dir=str(output/'raw-video'),record_video_size={'width':1600,'height':1000})
        setup.close()
        control=context.new_page();control.goto(base+'/canvas/');idle(control)
        page=None
        try:
            for story in stories:
                item={'id':story['id'],'title':story['title'],'dataset_id':story['dataset_id'],'steps':[],'errors':[]}
                report['stories'].append(item)
                current=api(control,'workspace')
                fresh=api(control,'sessions',{'version':current['version'],'dataset_id':story['dataset_id']})
                item['session_id']=fresh['session_id']
                page=context.new_page();page.on('pageerror',lambda error,item=item:item['errors'].append(str(error)))
                def capture_playback(response,item=item):
                    if response.url.endswith('/canvas/api/playback') and response.status==200:
                        item.setdefault('playback_responses',[]).append(response.json())
                page.on('response',capture_playback)
                page.goto(base+'/canvas/');idle(page)
                assert page.locator('#motion-select').input_value()=='on','Recording requires Flowing motion'
                cards={}
                try:
                    for number,step in enumerate(story['steps'],1):
                        result={'id':step['id'],'number':number}
                        item['steps'].append(result)
                        if step.get('target'):
                            current=api(page,'workspace');target=next(c for c in current['cards'] if c['id']==cards[step['target']]['id'])
                            select(page,target,focus=step.get('focus',False))
                        else:target=None
                        if step.get('clear_mark') and page.locator('#selection-chip').is_visible():page.locator('#clear-mark').click()
                        if step.get('mark_peak'):
                            assert target and target['chart'] in ('heatmap','density')
                            result['mark']=mark_peak(page,target,step['mark_peak'])
                        if step.get('prompt'):
                            before=api(page,'workspace');count=page.evaluate('window.canvasTelemetry.length')
                            page.locator('#prompt').press_sequentially(step['prompt'],delay=6)
                            page.wait_for_timeout(250);page.locator('#send').click()
                            page.wait_for_function('''count=>window.canvasTelemetry.slice(count).some(e=>['paint','local-request','request-error'].includes(e.type))''',arg=count,timeout=90000)
                            idle(page)
                            events=page.evaluate('count=>window.canvasTelemetry.slice(count)',count)
                            result['events']=events;result['prompt']=step['prompt']
                            assert not [e for e in events if e['type'] in ('request-error','render-error')],events
                            current=api(page,'workspace');result['workspace']=current
                            if step.get('local'):
                                assert any(e['type']=='local-request' for e in events)
                                assert not any(e['type']=='paint' for e in events)
                                assert len(current['cards'])==len(before['cards'])
                                card=next(c for c in current['cards'] if c['id']==target['id'])
                                assert card['data']==target['data'] and card['sql']==target['sql']
                            else:
                                paint=next(e for e in reversed(events) if e['type']=='paint')
                                assert len(paint['changes'])==1,paint
                                changed=paint['changes'][0]
                                assert changed['action']==step.get('intent','add'),changed
                                card=next(c for c in current['cards'] if c['id']==changed['id'])
                                if changed['action']=='add':assert len(current['cards'])==len(before['cards'])+1
                                else:assert target and changed['id']==target['id'] and len(current['cards'])==len(before['cards'])
                                result['timing']={k:paint[k] for k in ('request_id','elapsed_ms','model_ms','query_ms')}
                            cards[step['id']]=card
                            if step.get('target') and step.get('local'):cards[step['target']]=card
                            result['card']=card;result['validation']=validate_card(card,step)
                            if card['chart']=='tracks':
                                view=page.locator(f'[data-id="{card["id"]}"]')
                                view.get_by_role('slider',name='Track time in UTC').wait_for(state='visible',timeout=30000)
                                assert not view.get_by_text('Tracks unavailable',exact=True).count()
                            if card['chart']=='gallery':
                                minimum=min(6,len(card['data']['rows']))
                                page.wait_for_function('''({id,n})=>[...document.querySelectorAll('[data-id="'+id+'"] img')].filter(i=>i.complete&&i.naturalWidth>0).length>=n''',arg={'id':card['id'],'n':minimum},timeout=30000)
                            print(json.dumps({'story':story['id'],'step':step['id'],'chart':card['chart'],'rows':len(card['data']['rows']),'elapsed_ms':result.get('timing',{}).get('elapsed_ms'),'local':step.get('local',False)}),flush=True)
                        else:card=target
                        result['actions']=[action(page,card,operation) for operation in step.get('actions',[])]
                        page.screenshot(path=str(output/f'{story["id"]}-{number:02d}.png'))
                        result['passed']=True
                        (output/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
                    page.locator('#fit').click();idle(page);page.wait_for_timeout(900)
                    page.screenshot(path=str(output/f'{story["id"]}-final.png'))
                    assert not item['errors'],item['errors']
                    item['passed']=True
                except Exception as error:
                    item['failure']=repr(error);item['telemetry']=page.evaluate('window.canvasTelemetry||[]')
                    page.screenshot(path=str(output/f'{story["id"]}-failure.png'))
                    print(json.dumps({'story':story['id'],'failure':repr(error)}),flush=True)
                finally:
                    video=page.video;page.close();video.save_as(str(output/f'{story["id"]}.webm'));page=None
                    (output/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:
            try:
                if original:
                    current=api(control,'workspace')
                    restored=api(control,'sessions/'+original['session_id']+'/activate',{'version':current['version']})
                    assert {k:v for k,v in restored.items() if k!='version'}=={k:v for k,v in original.items() if k!='version'}
                    report['original_session_restored']=True
            except Exception as error:
                report['restoration_error']=repr(error)
            finally:
                report['passed']=report.get('original_session_restored',False) and len(report['stories'])==len(stories) and all(s.get('passed') for s in report['stories'])
                report['ended_at']=datetime.now(timezone.utc).isoformat()
                (output/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
                context.close();browser.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
