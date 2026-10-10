#!/usr/bin/env python3
"""Materialize an observation cohort with at least one CC0/CC-BY photograph."""
from pathlib import Path
from datetime import datetime, timezone
import sys,json,time
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from load_data import query

started=time.perf_counter()
query('CREATE TABLE IF NOT EXISTS exploration.wildlife_photo_index (observation_uuid UUID) ENGINE=MergeTree ORDER BY observation_uuid')
query('CREATE TABLE IF NOT EXISTS exploration.wildlife_gallery AS exploration.wildlife')
assert int(query('SELECT count() FROM exploration.wildlife_photo_index'))==0,'Inspect existing index before retrying'
assert int(query('SELECT count() FROM exploration.wildlife_gallery'))==0,'Inspect partial gallery before retrying'
query('''INSERT INTO exploration.wildlife_photo_index
 SELECT observation_uuid FROM exploration.wildlife_photos GROUP BY observation_uuid
 SETTINGS optimize_aggregation_in_order=1, max_bytes_before_external_group_by=700000000, max_bytes_before_external_sort=700000000''','wildlife_photo_index',timeout=1800)
count=int(query('SELECT count() FROM exploration.wildlife_photo_index'))
print('Photo-eligible observation IDs:',count,flush=True)
query('''INSERT INTO exploration.wildlife_gallery SELECT w.* FROM exploration.wildlife w
 INNER JOIN exploration.wildlife_photo_index p ON w.observation_uuid=p.observation_uuid
 SETTINGS join_algorithm='full_sorting_merge', max_bytes_before_external_sort=700000000,
 max_insert_block_size=500000, max_bytes_before_external_group_by=700000000''','wildlife_gallery_materialize',timeout=3600)
rows=int(query('SELECT count() FROM exploration.wildlife_gallery'))
assert 0 < rows <= count,(rows,count)
record={'rows':rows,'orphan_photo_observation_ids':count-rows,'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.perf_counter()-started,'cohort':'Observations with at least one CC0 or CC-BY photo metadata record in the same snapshot; one row per observation.'}
(BASE/'gallery.json').write_text(json.dumps(record,indent=2)+'\n')
dataset=json.loads((BASE/'dataset.json').read_text());dataset['photo_eligible_observations']=rows
dataset['tables']=json.loads(query("SELECT table,sum(rows) AS rows,sum(bytes_on_disk) AS bytes_on_disk FROM system.parts WHERE database='exploration' AND table LIKE 'wildlife%' AND active GROUP BY table FORMAT JSON"))['data']
(BASE/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
print(json.dumps(record),flush=True)
