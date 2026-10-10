#!/usr/bin/env python3
"""Post-hoc deadline view of the frozen SQL benchmark files; no source mutation."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'evidence/subsecond-analysis'
OUT.mkdir(exist_ok=True)
rows=[]
sources={}
for study in ['llm-clickhouse-budget','llm-clickhouse-deepseek-quant']:
    study_root=BASE/'benchmarks'/study
    if not study_root.is_dir():
        study_root=BASE.parent/study
    path=study_root/'results/full.jsonl'
    raw=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
    sources[study+'/results/full.jsonl']=hashlib.sha256(raw).hexdigest()
    groups=collections.defaultdict(list)
    for line in raw.decode().splitlines():
        row=json.loads(line)
        groups[row['route_id']].append(row)
    for route,attempts in groups.items():
        for population,subset in [('all',attempts),('tiers_1_to_3',[r for r in attempts if r['tier']<=3])]:
            for deadline in [250,500,750,1000,1500,2000]:
                count=sum(bool(r['correct']) and r.get('result_ready_ms') is not None and r['result_ready_ms']<=deadline for r in subset)
                rows.append(dict(study=study,route_id=route,population=population,deadline_ms=deadline,
                                 attempts=len(subset),correct_on_time=count,fraction=count/len(subset)))
(OUT/'analysis.json').write_text(json.dumps({'status':'Post-hoc descriptive analysis requested after initial results. Does not replace the predeclared five-second winner rule.','source_sha256':sources,'rows':rows},indent=2)+'\n')
with (OUT/'deadlines.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,ax=plt.subplots(figsize=(8,4.2))
for route,label,color in [('inference-net/fp8','V4.1 · InferenceNet FP8','#168879'),('sail-research/fp4','V4.1 · Sail Research FP4','#ec9435')]:
    subset=[r for r in rows if r['study'].endswith('quant') and route in r['route_id'] and r['population']=='all']
    ax.plot([r['deadline_ms']/1000 for r in subset],[100*r['fraction'] for r in subset],marker='o',label=label,color=color,lw=2)
ax.set(xlabel='Deadline from request start (seconds)',ylabel='Correct results within deadline (%)',
       ylim=(0,100),xlim=(0,2.08),title='Half a second is still a demanding deadline')
ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,loc='lower right');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/('correct-within-2s.'+ext),dpi=160)
print('Saved deadline counts and figure from frozen attempts')
