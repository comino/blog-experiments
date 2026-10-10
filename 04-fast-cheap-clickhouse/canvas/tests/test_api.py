"""HTTP boundary tests use fake model/MCP calls and isolated on-disk workspaces."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import uuid

TEMP = tempfile.TemporaryDirectory()
os.environ['CANVAS_STORE'] = TEMP.name + '/workspaces.sqlite'
os.environ.setdefault('MCP_AUTH_TOKEN','test-only')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as server
import httpx
from test_datasets import track_card, extent


class ApiBoundaries(unittest.IsolatedAsyncioTestCase):
    async def maritime_state(self, ids=('123456789',)):
        previous = (await self.client.get('/canvas/api/workspace', headers=self.h)).json()
        state = (await self.client.post('/canvas/api/sessions', headers=self.h,
                                      json={'version': previous['version'], 'dataset_id': 'maritime'})).json()
        example = track_card(ids)
        op = {key: example[key] for key in ('title', 'sql', 'chart', 'x', 'y', 'z', 'color') if key in example}
        op.update(action='add', title='Selected cargo ships')
        state, _ = server.apply_operations(state, [op], [{'data': example['data'], 'query_ms': 1}], 'Replay ships')
        return (await self.client.put('/canvas/api/workspace', headers=self.h, json=state)).json()

    def playback_task(self, state):
        card = state['cards'][0]
        return {'card_id': card['id'], 'revision': card['revision'], 'session_id': state['session_id'], 'version': state['version']}

    async def test_dataset_switch_creates_separate_session_and_new_chat_retains_dataset(self):
        wildlife = (await self.client.get('/canvas/api/workspace', headers=self.h)).json()
        maritime = (await self.client.post('/canvas/api/sessions', headers=self.h,
                                         json={'version': wildlife['version'], 'dataset_id': 'maritime'})).json()
        self.assertEqual(maritime['dataset_id'], 'maritime')
        fresh = (await self.client.post('/canvas/api/sessions', headers=self.h, json={'version': maritime['version']})).json()
        self.assertEqual(fresh['dataset_id'], 'maritime')
        resumed = (await self.client.post('/canvas/api/sessions/'+wildlife['session_id']+'/activate',
                                         headers=self.h, json={'version': fresh['version']})).json()
        self.assertEqual(resumed['dataset_id'], 'wildlife')
        sessions = (await self.client.get('/canvas/api/sessions', headers=self.h)).json()['sessions']
        self.assertEqual({s['dataset_id'] for s in sessions}, {'wildlife', 'maritime'})
        catalogue = (await self.client.get('/canvas/api/datasets', headers=self.h)).json()
        self.assertEqual(catalogue['active_id'], 'wildlife')
        self.assertEqual({d['id'] for d in catalogue['datasets']}, {'wildlife', 'maritime'})
        self.assertEqual((await self.client.get('/canvas/api/datasets')).status_code, 401)

    async def test_true_legacy_session_migrates_dataset_without_resetting_data_or_version(self):
        legacy = server.blank() | {'session_id': uuid.uuid4().hex, 'version': 23,
                                  'messages': [{'role': 'user', 'text': 'Old wildlife'}], 'undo': [{'cards': []}]}
        del legacy['dataset_id']
        with server.DB:
            server.DB.execute('INSERT INTO workspaces VALUES (?,?,?)', (self.user, 23, json.dumps(legacy)))
            server.remember_session(self.user, legacy)
        migrated = (await self.client.get('/canvas/api/workspace', headers=self.h)).json()
        self.assertEqual(migrated, legacy | {'dataset_id': 'wildlife'})
        raw = server.DB.execute('SELECT body FROM canvas_sessions WHERE id=?', (legacy['session_id'],)).fetchone()[0]
        self.assertEqual(json.loads(raw)['dataset_id'], 'wildlife')

    async def test_dataset_cannot_be_changed_by_workspace_save_or_stale_paint(self):
        state = (await self.client.get('/canvas/api/workspace', headers=self.h)).json()
        denied = await self.client.put('/canvas/api/workspace', headers=self.h, json=state | {'dataset_id': 'maritime'})
        self.assertEqual(denied.status_code, 409)
        with patch.object(server, 'plan_canvas', AsyncMock()) as planner:
            denied = await self.client.post('/canvas/api/paint', headers=self.h, json={'prompt': 'Ships', 'version': state['version'],
                        'request_id': uuid.uuid4().hex, 'dataset_id': 'maritime'})
        self.assertEqual(denied.status_code, 409); planner.assert_not_awaited()
        self.assertEqual((await self.client.get('/canvas/api/workspace', headers=self.h)).json(), state)

    async def test_model_cannot_query_other_dataset_or_external_functions(self):
        state = (await self.client.get('/canvas/api/workspace', headers=self.h)).json()
        for sql in ['SELECT count() AS n FROM exploration.ais_positions', "SELECT * FROM url('https://example.org')"]:
            op = {'action': 'add', 'title': 'Forbidden', 'sql': sql, 'chart': 'table', 'x': '', 'y': '', 'z': '', 'color': 'teal'}
            with patch.object(server, 'plan_canvas', AsyncMock(return_value=({'operations': [op]}, 1, {}))), patch.object(server, 'query', AsyncMock()) as query:
                response = await self.client.post('/canvas/api/paint', headers=self.h, json={'prompt': 'Show data', 'version': state['version'], 'request_id': uuid.uuid4().hex})
            self.assertIn('event: error', response.text); query.assert_not_awaited()
            self.assertEqual((await self.client.get('/canvas/api/workspace', headers=self.h)).json(), state)

    async def test_playback_uses_owned_saved_filters_and_does_not_mutate_workspace(self):
        state = await self.maritime_state()
        task = self.playback_task(state)
        positions = {'columns': ['vessel_id', 'observed_at', 'longitude', 'latitude', 'speed_knots'],
                     'rows': [['123456789', '2024-10-09 00:00:00', -84, 27, 9]]}
        storms = {'columns': ['storm_id', 'storm_name', 'observed_at', 'longitude', 'latitude', 'wind_knots'],
                  'rows': [['milton', 'MILTON', '2024-10-09 00:00:00', -85, 25, 145]]}
        query = AsyncMock(side_effect=[{'data': extent(), 'query_ms': 1}, {'data': positions, 'query_ms': 2}, {'data': storms, 'query_ms': 3}])
        with patch.object(server, 'query', query), patch.object(server, 'plan_canvas', AsyncMock()) as planner:
            response = await self.client.post('/canvas/api/playback', headers=self.h, json=task)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['rows'], positions['rows'])
        self.assertEqual(response.json()['storm_overlay']['names'], ['MILTON'])
        self.assertEqual(response.json()['query_ms'], 6)
        self.assertIn("lower(vessel_type) = 'cargo'", query.await_args_list[1].args[0])
        self.assertIn("vessel_id IN ('123456789')", query.await_args_list[1].args[0])
        self.assertEqual(query.await_args_list[1].kwargs['max_rows'], 6000)
        planner.assert_not_awaited()
        self.assertEqual((await self.client.get('/canvas/api/workspace', headers=self.h)).json(), state)

    async def test_playback_auth_foreign_cards_and_stale_versions_never_query(self):
        state = await self.maritime_state()
        task = self.playback_task(state)
        with patch.object(server, 'query', AsyncMock()) as query:
            self.assertEqual((await self.client.post('/canvas/api/playback', json=task)).status_code, 401)
            for change, status in [({'card_id': 'other-card'}, 404), ({'revision': 'old'}, 409), ({'version': -1}, 409), ({'session_id': 'other-session'}, 409)]:
                denied = await self.client.post('/canvas/api/playback', headers=self.h, json=task | change)
                self.assertEqual(denied.status_code, status)
            query.assert_not_awaited()

    async def test_playback_rejects_overlong_windows_and_too_many_tracks(self):
        state = await self.maritime_state()
        query = AsyncMock(return_value={'data': extent('2024-11-01 00:00:00'), 'query_ms': 1})
        with patch.object(server, 'query', query):
            response = await self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state))
        self.assertEqual(response.status_code, 400); self.assertEqual(query.await_count, 1)
        state = await self.maritime_state(tuple(str(i) for i in range(25)))
        with patch.object(server, 'query', AsyncMock()) as query:
            response = await self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state))
        self.assertEqual(response.status_code, 400); query.assert_not_awaited()

    async def test_playback_result_is_fenced_after_session_switch(self):
        state = await self.maritime_state()
        started = asyncio.Event(); release = asyncio.Event()
        async def delayed(sql, **kwargs):
            if 'min(observed_at) AS start' in sql:
                started.set(); await release.wait()
                return {'data': extent(), 'query_ms': 1}
            return {'data': {'columns': [], 'rows': []}, 'query_ms': 1}
        with patch.object(server, 'query', delayed):
            pending = asyncio.create_task(self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state)))
            await asyncio.wait_for(started.wait(), 1)
            fresh = await self.client.post('/canvas/api/sessions', headers=self.h, json={'version': state['version'], 'dataset_id': 'wildlife'})
            self.assertEqual(fresh.status_code, 200)
            release.set(); response = await pending
        self.assertEqual(response.status_code, 409)

    async def test_concurrent_playback_reads_queue_without_a_workspace_conflict(self):
        state = await self.maritime_state()
        started = asyncio.Event(); release = asyncio.Event()
        active = 0; maximum = 0; extent_calls = 0
        async def delayed(sql, **kwargs):
            nonlocal active, maximum, extent_calls
            active += 1; maximum = max(maximum, active)
            try:
                if 'min(observed_at) AS start' in sql:
                    extent_calls += 1
                    if extent_calls == 1:
                        started.set(); await release.wait()
                    return {'data': extent(), 'query_ms': 1}
                return {'data': {'columns': [], 'rows': []}, 'query_ms': 1}
            finally: active -= 1
        with patch.object(server, 'query', delayed):
            first = asyncio.create_task(self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state)))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state)))
            await asyncio.sleep(0)
            self.assertFalse(second.done())
            release.set()
            responses = await asyncio.gather(first, second)
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(maximum, 1)
        self.assertNotIn(self.user, server.PLAYBACK_WAITERS)
        self.assertEqual((await self.client.get('/canvas/api/workspace', headers=self.h)).json(), state)

    async def test_playback_queue_overload_is_retryable_and_does_not_query(self):
        state = await self.maritime_state()
        server.PLAYBACK_WAITERS[self.user] = 12
        try:
            with patch.object(server, 'query', AsyncMock()) as query:
                response = await self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state))
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers['retry-after'], '2')
            query.assert_not_awaited()
        finally: server.PLAYBACK_WAITERS.pop(self.user, None)

    async def test_new_card_does_not_interrupt_unchanged_track_playback(self):
        state = await self.maritime_state()
        started = asyncio.Event(); release = asyncio.Event()
        async def delayed(sql, **kwargs):
            if 'min(observed_at) AS start' in sql:
                started.set(); await release.wait()
                return {'data': extent(), 'query_ms': 1}
            return {'data': {'columns': [], 'rows': []}, 'query_ms': 1}
        with patch.object(server, 'query', delayed):
            pending = asyncio.create_task(self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state)))
            await asyncio.wait_for(started.wait(), 1)
            op = {'action': 'add', 'title': 'Another view', 'sql': 'SELECT 1', 'chart': 'bar',
                  'x': 'type', 'y': 'vessels', 'z': '', 'color': 'teal'}
            newer, _ = server.apply_operations(state, [op], [{'data': {'columns': ['type', 'vessels'], 'rows': [['Cargo', 7]]}, 'query_ms': 1}], 'Add a breakdown')
            saved = await self.client.put('/canvas/api/workspace', headers=self.h, json=newer)
            self.assertEqual(saved.status_code, 200)
            release.set(); response = await pending
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((await self.client.get('/canvas/api/workspace', headers=self.h)).json(), saved.json())

    async def test_same_revision_cannot_hide_a_changed_playback_query(self):
        state = await self.maritime_state()
        started = asyncio.Event(); release = asyncio.Event()
        async def delayed(sql, **kwargs):
            if 'min(observed_at) AS start' in sql:
                started.set(); await release.wait()
                return {'data': extent(), 'query_ms': 1}
            return {'data': {'columns': [], 'rows': []}, 'query_ms': 1}
        with patch.object(server, 'query', delayed):
            pending = asyncio.create_task(self.client.post('/canvas/api/playback', headers=self.h, json=self.playback_task(state)))
            await asyncio.wait_for(started.wait(), 1)
            changed = copy.deepcopy(state)
            changed['cards'][0]['sql'] = changed['cards'][0]['sql'].replace("'cargo'", "'tanker'")
            saved = await self.client.put('/canvas/api/workspace', headers=self.h, json=changed)
            self.assertEqual(saved.status_code, 200)
            release.set(); response = await pending
        self.assertEqual(response.status_code, 409)

    async def test_mark_filter_is_executed_and_stale_marks_never_call_model(self):
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'Monarch months','sql':"SELECT toMonth(observed_on) AS month,count() AS n FROM exploration.wildlife WHERE taxon_id=48662 GROUP BY month",'chart':'bar','x':'month','y':'n','z':'','color':'teal'}
        data={'data':{'columns':['month','n'],'rows':[[1,10],[2,20]]},'query_ms':1}
        state,_=server.apply_operations(state,[op],[data],'Start')
        state=(await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).json()
        card=state['cards'][0]
        mark={'kind':'points','card_id':card['id'],'revision':card['revision'],'row_indices':[1]}
        payload={'prompt':'Break this down','selected_id':card['id'],'version':state['version'],'request_id':uuid.uuid4().hex,'selection':mark}
        planner=AsyncMock(return_value=({'operations':[op|{'action':'update','target_id':card['id'],'sql':'SELECT toMonth(observed_on) AS month,count() AS n FROM exploration.wildlife GROUP BY month'}]},100,{}))
        query=AsyncMock(return_value=data|{'data':{'columns':['month','n'],'rows':[[2,20]]}})
        with patch.object(server,'plan_canvas',planner),patch.object(server,'query',query):
            bad=await self.client.post('/canvas/api/paint',headers=self.h,json=payload|{'selection':mark|{'revision':'old'}})
            self.assertEqual(bad.status_code,400);planner.assert_not_awaited()
            response=await self.client.post('/canvas/api/paint',headers=self.h,json=payload)
        self.assertIn('event: result',response.text)
        executed=query.await_args.args[0]
        self.assertIn('taxon_id = 48662',executed);self.assertIn('(toMonth(observed_on)) IN ((2))',executed)
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(latest['cards'][0]['sql'],executed)
        self.assertEqual(latest['cards'][0]['title'],'Breakdown · 1 marked point')

    async def asyncSetUp(self):
        # Local test runs do not have the container's /evidence volume.
        def test_path(value):
            if str(value) == '/evidence/canvas-calls.jsonl':
                return Path(TEMP.name) / 'canvas-calls.jsonl'
            return Path(value)
        evidence_path = patch.object(server, 'Path', side_effect=test_path)
        evidence_path.start()
        self.addCleanup(evidence_path.stop)
        self.user = uuid.uuid4().hex
        self.other = uuid.uuid4().hex
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://test')
        async def get(url,headers):
            value=headers['Authorization']
            if value=='Bearer user-a': return httpx.Response(200,json={'_id':self.user})
            if value=='Bearer user-b': return httpx.Response(200,json={'_id':self.other})
            return httpx.Response(401,json={})
        self.auth=patch.object(server.HTTP,'get',side_effect=get);self.auth.start()
        self.h={'Authorization':'Bearer user-a'}

    async def asyncTearDown(self):
        self.auth.stop();await self.client.aclose()

    async def test_legacy_canvas_is_adopted_without_losing_history(self):
        original=server.blank()|{'version':37,'messages':[{'role':'user','text':'Original exploration'}],
            'undo':[{'label':'Earlier view','cards':[]}],'name':'My existing work'}
        with server.DB:
            server.DB.execute('INSERT INTO workspaces VALUES (?,?,?)',(self.user,37,json.dumps(original)))
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual({k:v for k,v in state.items() if k!='session_id'},original)
        sessions=(await self.client.get('/canvas/api/sessions',headers=self.h)).json()
        self.assertEqual(sessions['active_id'],state['session_id'])
        self.assertEqual(len(sessions['sessions']),1)
        self.assertEqual(sessions['sessions'][0]['title'],'Original exploration')
        self.assertEqual((await self.client.get('/canvas/api/workspace',headers=self.h)).json(),state)

    async def test_new_chat_is_empty_and_previous_chat_can_be_resumed(self):
        original=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'Original','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        original,_=server.apply_operations(original,[op],[result],'First question')
        original['messages']=[{'role':'user','text':'First question'}]
        original=(await self.client.put('/canvas/api/workspace',headers=self.h,json=original)).json()
        created=await self.client.post('/canvas/api/sessions',headers=self.h,json={'version':original['version']})
        self.assertEqual(created.status_code,200)
        fresh=created.json()
        for key in ['cards','messages','undo','redo']:self.assertEqual(fresh[key],[])
        self.assertNotEqual(fresh['session_id'],original['session_id'])
        sessions=(await self.client.get('/canvas/api/sessions',headers=self.h)).json()
        self.assertEqual(len(sessions['sessions']),2)
        resumed=await self.client.post('/canvas/api/sessions/'+original['session_id']+'/activate',headers=self.h,json={'version':fresh['version']})
        self.assertEqual(resumed.status_code,200)
        self.assertEqual({k:v for k,v in resumed.json().items() if k!='version'}, {k:v for k,v in original.items() if k!='version'})
        self.assertGreater(resumed.json()['version'],fresh['version'])

    async def test_sessions_are_private_and_forged_targets_cannot_be_saved(self):
        mine=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        other_headers={'Authorization':'Bearer user-b'}
        other=(await self.client.get('/canvas/api/workspace',headers=other_headers)).json()
        sessions=(await self.client.get('/canvas/api/sessions',headers=self.h)).json()['sessions']
        self.assertNotIn(other['session_id'],[s['id'] for s in sessions])
        denied=await self.client.post('/canvas/api/sessions/'+other['session_id']+'/activate',headers=self.h,json={'version':mine['version']})
        self.assertEqual(denied.status_code,404)
        forged=await self.client.put('/canvas/api/workspace',headers=self.h,json=mine|{'session_id':other['session_id']})
        self.assertEqual(forged.status_code,409)
        self.assertEqual((await self.client.get('/canvas/api/workspace',headers=self.h)).json(),mine)
        self.assertEqual((await self.client.get('/canvas/api/sessions')).status_code,401)
        self.assertEqual((await self.client.post('/canvas/api/sessions',json={'version':0})).status_code,401)

    async def test_stale_tab_cannot_change_the_new_chat_or_create_orphan_sessions(self):
        old=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        fresh=(await self.client.post('/canvas/api/sessions',headers=self.h,json={'version':old['version']})).json()
        self.assertEqual((await self.client.put('/canvas/api/workspace',headers=self.h,json=old)).status_code,409)
        self.assertEqual((await self.client.post('/canvas/api/sessions',headers=self.h,json={'version':old['version']})).status_code,409)
        with patch.object(server,'plan_canvas',AsyncMock()) as planner:
            response=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'An old request','version':old['version'],'request_id':uuid.uuid4().hex})
            self.assertEqual(response.status_code,409);planner.assert_not_awaited()
        self.assertEqual((await self.client.get('/canvas/api/workspace',headers=self.h)).json(),fresh)
        self.assertEqual(len((await self.client.get('/canvas/api/sessions',headers=self.h)).json()['sessions']),2)

    async def test_inflight_result_cannot_leak_into_a_new_session(self):
        old=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        started=asyncio.Event();release=asyncio.Event()
        async def delayed_plan(*args):
            started.set();await release.wait()
            return {'operations':[{'action':'add','title':'Late result','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}]},100,{}
        with patch.object(server,'plan_canvas',delayed_plan),patch.object(server,'query',AsyncMock()) as query:
            pending=asyncio.create_task(self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Slow request','version':old['version'],'request_id':uuid.uuid4().hex}))
            await asyncio.wait_for(started.wait(),1)
            fresh=(await self.client.post('/canvas/api/sessions',headers=self.h,json={'version':old['version']})).json()
            release.set();response=await pending
            self.assertIn('event: error',response.text);query.assert_not_awaited()
        self.assertEqual((await self.client.get('/canvas/api/workspace',headers=self.h)).json(),fresh)

    async def test_authentication_and_user_isolation(self):
        self.assertEqual((await self.client.get('/canvas/api/workspace')).status_code,401)
        self.assertEqual((await self.client.get('/canvas/api/workspace',headers={'Authorization':'Bearer forged'})).status_code,401)
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        state['name']='Owner A workspace'
        self.assertEqual((await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).status_code,200)
        other=(await self.client.get('/canvas/api/workspace',headers={'Authorization':'Bearer user-b'})).json()
        self.assertNotEqual(other['name'],state['name'])

    async def test_stale_save_cannot_overwrite_newer_work(self):
        original=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        changed=original|{'name':'New value'}
        self.assertEqual((await self.client.put('/canvas/api/workspace',headers=self.h,json=changed)).status_code,200)
        self.assertEqual((await self.client.put('/canvas/api/workspace',headers=self.h,json=original)).status_code,409)
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(latest['name'],'New value');self.assertEqual(latest['version'],1)

    async def test_query_failure_does_not_commit_any_card_or_history(self):
        before=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'A view','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        with patch.object(server,'plan_canvas',AsyncMock(return_value=({'operations':[op,op.copy()]},100,{}))),patch.object(server,'query',AsyncMock(side_effect=[result,ValueError('Query failed')])):
            r=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Add two views','version':0,'request_id':uuid.uuid4().hex})
        self.assertIn('event: error',r.text)
        after=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(after,before)

    async def test_visual_edit_uses_no_database_call(self):
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'A view','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        state,_=server.apply_operations(state,[op],[result],'Start')
        state=(await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).json()
        cid=state['cards'][0]['id']
        operation=op|{'action':'update','target_id':cid,'source_id':cid,'sql':None,'chart':'bar'}
        query=AsyncMock()
        with patch.object(server,'plan_canvas',AsyncMock(return_value=({'operations':[operation]},100,{}))),patch.object(server,'query',query):
            r=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Make bars','selected_id':cid,'version':state['version'],'request_id':uuid.uuid4().hex})
        self.assertIn('event: result',r.text);query.assert_not_awaited()
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(latest['cards'][0]['id'],cid)
        self.assertEqual(latest['cards'][0]['chart'],'bar')
        self.assertEqual(latest['cards'][0]['data'],result['data'])

    async def test_explicit_add_never_overwrites_a_selected_card(self):
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'Original','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        state,_=server.apply_operations(state,[op],[result],'Start')
        state=(await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).json()
        cid=state['cards'][0]['id']
        # Even an incorrect model update must respect the explicit Add intent.
        operation=op|{'action':'update','target_id':cid,'title':'Comparison','sql':None}
        with patch.object(server,'plan_canvas',AsyncMock(return_value=({'operations':[operation]},100,{}))):
            r=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Compare','intent':'add','selected_id':cid,'version':state['version'],'request_id':uuid.uuid4().hex})
        self.assertIn('event: result',r.text)
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(latest['cards'][0],state['cards'][0])
        self.assertEqual(len(latest['cards']),2)
        self.assertEqual(latest['cards'][1]['derived_from'],cid)

    async def test_new_question_adds_even_if_model_updates_the_selected_card(self):
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'Original','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        state,_=server.apply_operations(state,[op],[result],'Start')
        state=(await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).json()
        cid=state['cards'][0]['id']
        operation=op|{'action':'update','target_id':cid,'title':'Another breakdown','sql':None}
        planner=AsyncMock(return_value=({'operations':[operation]},100,{}))
        with patch.object(server,'plan_canvas',planner):
            response=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Show monthly counts','selected_id':cid,'version':state['version'],'request_id':uuid.uuid4().hex})
        self.assertIn('event: result',response.text)
        first_event=json.loads(response.text.split('data: ',1)[1].split('\n',1)[0])
        self.assertEqual(first_event['intent'],'add')
        self.assertEqual(planner.await_args.args[1].intent,'add')
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(latest['cards'][0],state['cards'][0])
        self.assertEqual(len(latest['cards']),2)
        self.assertEqual(latest['cards'][1]['derived_from'],cid)

    async def test_explicit_comparison_edit_cannot_add_a_card_when_model_chooses_add(self):
        state=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        op={'action':'add','title':'Original','sql':'SELECT 1','chart':'line','x':'x','y':'y','z':'','color':'teal'}
        result={'data':{'columns':['x','y'],'rows':[[1,2]]},'query_ms':4}
        state,_=server.apply_operations(state,[op],[result],'Start')
        state=(await self.client.put('/canvas/api/workspace',headers=self.h,json=state)).json()
        cid=state['cards'][0]['id']
        operation=op|{'action':'add','target_id':None,'source_id':cid,'title':'Refined','sql':None,'chart':'bar'}
        with patch.object(server,'plan_canvas',AsyncMock(return_value=({'operations':[operation]},100,{}))):
            r=await self.client.post('/canvas/api/paint',headers=self.h,json={'prompt':'Change this comparison chart to bars','selected_id':cid,'version':state['version'],'request_id':uuid.uuid4().hex})
        self.assertIn('event: result',r.text)
        latest=(await self.client.get('/canvas/api/workspace',headers=self.h)).json()
        self.assertEqual(len(latest['cards']),1)
        self.assertEqual(latest['cards'][0]['id'],cid)
        self.assertEqual(latest['cards'][0]['chart'],'bar')
        self.assertEqual(latest['cards'][0]['history'][-1]['title'],'Original')

if __name__=='__main__':unittest.main()
