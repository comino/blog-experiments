#!/usr/bin/env python3
"""Offline reference audit for live Data Canvas rehearsal reports.

No network, SQL execution, browser access or application mutations. Requires the
project's locked sqlglot dependency. Unknown semantics fail closed for review.
Consolidated reference shape: {wildlife: {...}, maritime: {...}, supplemental: [...]}.
Reads plain JSON and gzip-compressed JSON. Defaults live beside this script under
demos/, so it can run from any working directory.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import gzip
import math
from pathlib import Path
import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope, Scope

UNKNOWN = object()
MAPPINGS = {
 'monarch-seasonal-footprint': {'map':'monarch_density_5deg','wheel':'monarch_seasonal_wheel','latitude':'monarch_month_latitude','surface':'monarch_month_latitude','photos':'monarch_august_cell_gallery'},
 'fungi-opposite-seasons': {'map':'fungi_two_boxes_density_2deg','months':'fungi_normalized_seasons','wheel':'fungi_normalized_seasons','australia':'fungi_australia_may_density_1deg','photos':'fungi_sydney_may_gallery_diverse'},
 'elsa-and-milton': {'replay':'elsa_replay','longitude':'elsa_longitude','ports':'raw_ports','index':'normalized_ports','activity':'tampa_matrix','surface':'tampa_matrix'},
 'mississippi-motion-and-stillness': {'replay':'river_replay','speeds':'river_speeds','slow':'river_slow_spells','activity':'river_moving_matrix','surface':'river_moving_matrix'},
}
LOCAL_TARGETS = {('monarch-seasonal-footprint','surface'):'latitude',('fungi-opposite-seasons','wheel'):'months',('elsa-and-milton','surface'):'activity',('mississippi-motion-and-stillness','surface'):'activity'}
VESSELS = {'538006273':'ELSA','538090649':'AQUAVITA TRADER','538004628':'CHEM RANGER','563105500':'FAIRCHEM FORTITUDE','311001216':'AFRICAN TEAL','477615900':'CSSC TAI YUAN','538009752':'GENCO MARY'}
ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCES = ROOT / 'demos' / 'reference-data.json.gz'
DEFAULT_REPORT = ROOT / 'demos' / 'rehearsal-evidence.json.gz'


def read(path):
    path=Path(path)
    opener=gzip.open if path.suffix.lower()=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as source:
        return json.load(source)
def records(value):
    if isinstance(value.get('result'),list): return value['result']
    data=value.get('data',value)
    return [dict(zip(data['columns'],row)) for row in data.get('rows',[])]
def textkey(value): return re.sub(r'[^a-z0-9]+','',str(value).lower())
def fname(node):
    return (str(node.this) if isinstance(node,(exp.Anonymous,exp.AnonymousAggFunc)) else node.sql_name()).lower().replace('_','') if isinstance(node,exp.Func) else ''
def funcargs(node):
    if isinstance(node,(exp.Anonymous,exp.AnonymousAggFunc)): return node.expressions
    return [v for k,v in node.args.items() if k not in {'alias'} for v in (v if isinstance(v,list) else [v]) if isinstance(v,exp.Expression)]
def parse(sql):
    trees=sqlglot.parse(sql,read='clickhouse')
    if len(trees)!=1 or not isinstance(trees[0],exp.Select): raise ValueError('Expected one SELECT query')
    return trees[0]
def alias_map(tree):
    out={}
    for select in tree.find_all(exp.Select):
        for item in select.expressions:
            if isinstance(item,exp.Alias): out[item.alias]=item.this
    return out

def resolve(node,aliases,seen=frozenset()):
    if isinstance(node,exp.Column) and not node.table and node.name in aliases and node.name not in seen:
        return resolve(aliases[node.name],aliases,seen|{node.name})
    return node.copy().transform(lambda x:resolve(aliases[x.name],aliases,seen|{x.name}) if isinstance(x,exp.Column) and not x.table and x.name in aliases and x.name not in seen else x,copy=False)

def utc(value):
    if isinstance(value,datetime): dt=value
    elif isinstance(value,date): dt=datetime.combine(value,datetime.min.time())
    elif isinstance(value,str):
        raw=value.strip().replace('Z','+00:00')
        if re.fullmatch(r'\d{4}-\d{2}',raw):raw+='-01'
        dt=datetime.fromisoformat(raw)
    else: raise ValueError('Not a date/time: '+repr(value))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
def comparable(a,b):
    if isinstance(a,(date,datetime)) and isinstance(b,str):b=utc(b)
    if isinstance(b,(date,datetime)) and isinstance(a,str):a=utc(a)
    if isinstance(a,date) and not isinstance(a,datetime):a=utc(a)
    if isinstance(b,date) and not isinstance(b,datetime):b=utc(b)
    return a,b

def evaluate(node,row,aliases=None,seen=frozenset()):
    """Small explicit evaluator for cohort/gap predicates, never arbitrary SQL."""
    aliases=aliases or {}
    if node is None:return None
    if isinstance(node,exp.Paren):return evaluate(node.this,row,aliases,seen)
    if isinstance(node,exp.Null):return None
    if isinstance(node,exp.Boolean):return node.this
    if isinstance(node,exp.Literal):return node.this if node.is_string else float(node.this)
    if isinstance(node,exp.Column):
        if node.name in row:return row[node.name]
        if node.name in aliases and node.name not in seen:return evaluate(aliases[node.name],row,aliases,seen|{node.name})
        return UNKNOWN
    if isinstance(node,exp.Window) and fname(node.this) in {'lag','laginframe'}:
        args=funcargs(node.this)
        previous=row.get('__previous__')
        return evaluate(args[0],previous,aliases,seen) if args and previous is not None else UNKNOWN
    if isinstance(node,exp.Tuple):return tuple(evaluate(e,row,aliases,seen) for e in node.expressions)
    if isinstance(node,(exp.And,exp.Or)):
        a,b=evaluate(node.this,row,aliases,seen),evaluate(node.expression,row,aliases,seen)
        if isinstance(node,exp.And):return False if a is False or b is False else (UNKNOWN if UNKNOWN in (a,b) else bool(a and b))
        return True if a is True or b is True else (UNKNOWN if UNKNOWN in (a,b) else bool(a or b))
    if isinstance(node,exp.Not):
        a=evaluate(node.this,row,aliases,seen);return UNKNOWN if a is UNKNOWN else not a
    if isinstance(node,exp.Is):
        a,b=evaluate(node.this,row,aliases,seen),evaluate(node.expression,row,aliases,seen)
        return UNKNOWN if UNKNOWN in (a,b) else a is b
    if isinstance(node,exp.Between):
        value=evaluate(node.this,row,aliases,seen);lo=evaluate(node.args['low'],row,aliases,seen);hi=evaluate(node.args['high'],row,aliases,seen)
        if UNKNOWN in (value,lo,hi):return UNKNOWN
        if None in (value,lo,hi):return False
        value,lo=comparable(value,lo);value,hi=comparable(value,hi);return lo<=value<=hi
    if isinstance(node,exp.In):
        value=evaluate(node.this,row,aliases,seen)
        if node.args.get('query'):return UNKNOWN
        values=[evaluate(e,row,aliases,seen) for e in node.expressions]
        if value is UNKNOWN or any(v is UNKNOWN for v in values):return UNKNOWN
        return value in values
    if isinstance(node,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Add,exp.Sub,exp.Mul,exp.Div,exp.Mod)):
        a,b=evaluate(node.this,row,aliases,seen),evaluate(node.expression,row,aliases,seen)
        if UNKNOWN in (a,b):return UNKNOWN
        if a is None or b is None:return False if isinstance(node,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE)) else None
        a,b=comparable(a,b)
        if isinstance(node,exp.EQ):return a==b
        if isinstance(node,exp.NEQ):return a!=b
        if isinstance(node,exp.GT):return a>b
        if isinstance(node,exp.GTE):return a>=b
        if isinstance(node,exp.LT):return a<b
        if isinstance(node,exp.LTE):return a<=b
        if isinstance(node,exp.Add):return a+b
        if isinstance(node,exp.Sub):return (a-b).total_seconds() if isinstance(a,datetime) else a-b
        if isinstance(node,exp.Mul):return a*b
        if isinstance(node,exp.Div):return a/b
        return a%b
    if isinstance(node,exp.Neg):
        a=evaluate(node.this,row,aliases,seen);return UNKNOWN if a is UNKNOWN else -a
    if isinstance(node,(exp.Cast,exp.TryCast)):return evaluate(node.this,row,aliases,seen)
    if isinstance(node,exp.If):
        c=evaluate(node.this,row,aliases,seen)
        return UNKNOWN if c is UNKNOWN else evaluate(node.args.get('true') if c else node.args.get('false'),row,aliases,seen)
    if isinstance(node,exp.Case):
        for branch in node.args.get('ifs',[]):
            c=evaluate(branch.this,row,aliases,seen)
            if c is UNKNOWN:return UNKNOWN
            if c:return evaluate(branch.args['true'],row,aliases,seen)
        return evaluate(node.args.get('default'),row,aliases,seen)
    if isinstance(node,exp.Coalesce):
        for child in [node.this]+node.expressions:
            value=evaluate(child,row,aliases,seen)
            if value is UNKNOWN:return UNKNOWN
            if value is not None:return value
        return None
    if isinstance(node,exp.DateDiff):
        a,b=evaluate(node.this,row,aliases,seen),evaluate(node.expression,row,aliases,seen)
        if UNKNOWN in (a,b):return UNKNOWN
        if a is None or b is None:return None
        unit=str(node.args['unit']).lower().strip("'")
        scale={'second':1,'minute':60,'hour':3600,'day':86400}.get(unit)
        if not scale:return UNKNOWN
        return math.floor(utc(a).timestamp()/scale)-math.floor(utc(b).timestamp()/scale)
    if isinstance(node,exp.Func):
        name=fname(node);args=[evaluate(e,row,aliases,seen) for e in funcargs(node)]
        if any(a is UNKNOWN for a in args):return UNKNOWN
        if name=='multiif':
            for i in range(0,len(args)-1,2):
                if args[i]:return args[i+1]
            return args[-1]
        if name in {'isnull','isnotnull','notempty','empty'}:
            return {'isnull':args[0] is None,'isnotnull':args[0] is not None,'notempty':bool(args[0]),'empty':not bool(args[0])}[name]
        if name in {'ifnull','coalesce'}:return next((a for a in args if a is not None),None)
        if not args or args[0] is None:return None
        if name in {'lower','lowerutf8'}:return str(args[0]).lower()
        if name in {'upper','upperutf8'}:return str(args[0]).upper()
        if name in {'month','tomonth'}:return utc(args[0]).month
        if name in {'year','toyear'}:return utc(args[0]).year
        if name in {'todate','todate32','date'}:return utc(args[0]).replace(hour=0,minute=0,second=0,microsecond=0)
        if name in {'todatetime','todatetime64','parsedatetimebesteffort'}:return utc(args[0])
        if name in {'floor','intdiv'}:return math.floor(args[0] if name=='floor' else args[0]/args[1])
        if name in {'round'}:return round(args[0],int(args[1]) if len(args)>1 else 0)
        if name in {'abs'}:return abs(args[0])
        if name in {'tostring'}:return str(args[0])
    return UNKNOWN


def base_paths(tree):
    scopes=list(traverse_scope(tree));consumers=defaultdict(list)
    for scope in scopes:
        for _,source in scope.selected_sources.values():
            if isinstance(source,Scope):consumers[id(source)].append(scope)
    paths=[]
    def barrier(scope):return any(e.find(exp.AggFunc) or e.find(exp.Window) for e in scope.expression.expressions)
    def visit(scope,conditions,trail):
        if id(scope) in trail:return
        own=[scope.expression.args[k].this for k in ('prewhere','where') if scope.expression.args.get(k)]
        conditions=conditions+own
        following=[] if barrier(scope) else consumers[id(scope)]
        if not following:paths.append(conditions)
        for parent in following:visit(parent,conditions,trail|{id(scope)})
    tables=[]
    for scope in scopes:
        for _,source in scope.selected_sources.values():
            if isinstance(source,exp.Table):
                if not isinstance(source.this,exp.Identifier) or source.db!='exploration':raise ValueError('Unexpected external/table-function source')
                tables.append(source.name);visit(scope,[],set())
    return set(tables),paths

def accepted(paths,row,aliases):
    unknown=False
    for conditions in paths:
        values=[evaluate(e,row,aliases) for e in conditions]
        if any(v is False or v is None for v in values):continue
        if any(v is UNKNOWN for v in values):unknown=True
        else:return True
    return UNKNOWN if unknown else False

def cohort_probes(reference):
    t=utc('2025-08-15') if reference.startswith('monarch') else utc('2025-05-15') if reference.startswith('fungi') else utc('2024-10-09 12:00:00')
    base={'observed_at':t,'observed_on':t,'source_date':t.replace(hour=0), 'vessel_id':'538006273' if reference.startswith('elsa') else '538090649',
          'vessel_name':'ELSA','vessel_type':'Cargo','speed_knots':0.5,'latitude':42.5 if reference.startswith('monarch') else -33.5 if reference.startswith('fungi') else 29.5,'longitude':-85 if reference.startswith('monarch') else 151.5 if reference.startswith('fungi') else -89.5,
          'scientific_name':'Danaus plexippus' if reference.startswith('monarch') else 'Fungi','taxon_id':48662 if reference.startswith('monarch') else 47170,'group_name':'Insects' if reference.startswith('monarch') else 'Fungi'}
    if reference in {'raw_ports','normalized_ports','tampa_matrix'}:base.update(latitude=27.8,longitude=-82.6)
    if reference.startswith('elsa'):base.update(latitude=27.5,longitude=-85)
    seeds=[base]
    if reference=='fungi_normalized_seasons' or reference=='fungi_two_boxes_density_2deg':seeds.append(base|{'latitude':52,'longitude':10})
    if reference in {'raw_ports','normalized_ports'}:seeds.append(base|{'latitude':26.1,'longitude':-80})
    probes=[]
    for seed in seeds:
        probes.append(seed)
        for raw in ['2024-09-30 23:59:59','2024-10-01','2024-10-05 23:59:59','2024-10-06','2024-10-06 23:59:59','2024-10-07','2024-10-08 23:59:59','2024-10-09','2024-10-09 23:59:59','2024-10-10','2024-10-12 23:59:59','2024-10-13','2024-10-15 23:59:59','2024-10-16','2024-12-31 23:59:59','2025-01-01','2025-04-30 23:59:59','2025-05-01','2025-05-31 23:59:59','2025-06-01','2025-08-01','2025-08-31 23:59:59','2025-09-01','2025-12-31 23:59:59','2026-01-01']:
            stamp=utc(raw);probes.append(seed|{'observed_at':stamp,'observed_on':stamp,'source_date':stamp.replace(hour=0,minute=0,second=0)})
        # Half-open vs inclusive geographic edges are tolerated if complete real
        # aggregates agree. Probe 0.001 degrees on either side, not exact edges.
        for field,bounds in [('latitude',[-40,-34,-33,-30,0,25.95,26.25,27.5,28.1,28.5,30.2,40,45,48,60]),('longitude',[-130,-90.5,-88.6,-82.9,-82.35,-80.2,-79.8,-60,-10,30,140,151,152,154])]:
            for value in bounds:
                for delta in (-.001,.001):probes.append(seed|{field:value+delta})
        for value in list(VESSELS)+['000000000']:probes.append(seed|{'vessel_id':value})
        for speed in [None,0,.999,1,2.999,3,12]:probes.append(seed|{'speed_knots':speed})
        for label in ['Cargo','Tanker','Fishing','Passenger']:probes.append(seed|{'vessel_type':label})
        # Taxonomy fields must change together: an observation cannot simultaneously
        # be the verified monarch taxon and a mammal. Omitted redundant group filters
        # are harmless when taxon/species filters and every real count agree.
        probes.append(seed|{'scientific_name':'Vulpes vulpes','taxon_id':42069,'group_name':'Mammals'})
    return probes

def scope_check(actual,reference,ref_id):
    a,ap=base_paths(actual);r,rp=base_paths(reference)
    if a!=r:return False,{'expected_tables':sorted(r),'actual_tables':sorted(a)}
    aa,ra=alias_map(actual),alias_map(reference);different=[];unknown=0
    for row in cohort_probes(ref_id):
        av,rv=accepted(ap,row,aa),accepted(rp,row,ra)
        if av is UNKNOWN or rv is UNKNOWN:unknown+=1;continue
        if av!=rv:different.append({k:(v.isoformat() if isinstance(v,datetime) else v) for k,v in row.items() if k in {'observed_at','latitude','longitude','vessel_id','vessel_type','speed_knots','scientific_name','group_name'}})
    return not different and not unknown,{'probes':len(cohort_probes(ref_id)),'different_probes':len(different),'unknown_probes':unknown,'examples':different[:3]}


def canonical_region(value):
    s=textkey(value)
    if 'everglades' in s:return 'everglades'
    if 'tampa' in s:return 'tampa'
    if 'australia' in s or s in {'south','southernhemisphere'}:return 'australia'
    if 'europe' in s or s in {'north','northernhemisphere'}:return 'europe'
    return s

def vessel(value):
    raw=str(value).strip()
    if raw in VESSELS:return raw
    matches=[i for i,n in VESSELS.items() if textkey(n)==textkey(raw) or i in raw]
    return matches[0] if len(matches)==1 else raw

def key_value(role,value):
    if role=='month':
        try:return int(value)
        except (ValueError,TypeError):return utc(value).month
    if role=='day':return utc(value).date().isoformat()
    if role in {'hour','started_at','ended_at','start_time','end_time'}:return utc(value).isoformat()
    if role=='region':return canonical_region(value)
    if role=='vessel_id':return vessel(value)
    if role in {'lon_bin','lat_bin','latitude_band','hour_utc'}:return round(float(value),7)
    return value

def projection(card):
    tree=parse(card['sql']);aliases=alias_map(tree)
    out={}
    for node in tree.expressions:
        name=node.alias_or_name
        if name:out[name]=resolve(node.unalias(),aliases)
    return tree,aliases,out

def role_from_expr(node):
    s=textkey(node.sql(dialect='clickhouse'));name=fname(node)
    cols={c.name for c in node.find_all(exp.Column)}
    if isinstance(node,exp.Column):return {'vessel_name':'name_label','vessel_type':'type_label'}.get(node.name,node.name)
    if name in {'any','anyvalue','arbitrary'}:
        if 'vessel_name' in cols:return 'name_label'
        if 'vessel_type' in cols:return 'type_label'
    if isinstance(node,exp.Avg):
        if 'longitude' in cols:return 'lon_center'
        if 'latitude' in cols:return 'lat_center'
        if 'speed_knots' in cols:return 'mean_speed'
    if isinstance(node,(exp.Min,exp.Max)):
        prefix='min' if isinstance(node,exp.Min) else 'max'
        if 'observed_at' in cols:return 'start_time' if prefix=='min' else 'end_time'
        for field in ('speed_knots','longitude','latitude'):
            if field in cols:return prefix+'_'+('speed' if field=='speed_knots' else field)
    if name=='argmax':
        if 'longitude' in cols:return 'last_longitude'
        if 'latitude' in cols:return 'last_latitude'
    if (isinstance(node,exp.Count) and isinstance(node.this,exp.Distinct)) or name=='uniqexact':return 'vessel_ids'
    if isinstance(node,exp.Count):return 'reports'
    if 'avgif' in s:return 'baseline'
    if 'datediff' in s and ('3600' in s or 'hour' in s):return 'duration_hours'
    return None

def role_columns(card,ref_id):
    tree,aliases,exprs=projection(card);out={};keys=[]
    x,y,z=card.get('x'),card.get('y'),card.get('z')
    def assign(col,role):
        if col and col in card['data']['columns']:out[col]=role
    if 'density' in ref_id:
        keys=['lon_bin','lat_bin'];assign(x,'lon_bin');assign(y,'lat_bin');assign(z,'observations')
    elif ref_id=='monarch_seasonal_wheel':
        keys=['month'];assign(x,'month');assign(z if z else y,'observations')
    elif ref_id=='monarch_month_latitude':
        keys=['month','latitude_band'];assign(x,'month');assign(y,'latitude_band');assign(z,'observations')
    elif ref_id=='fungi_normalized_seasons':
        keys=['month','region'];assign(x,'month')
        assign(y,'region' if card['chart']=='seasons' else 'annual_share_pct');assign(z,'annual_share_pct' if card['chart']=='seasons' else 'region')
    elif ref_id in {'raw_ports','normalized_ports'}:
        keys=['day','region'];assign(x,'day');assign(z,'region');assign(y,'vessel_ids' if ref_id=='raw_ports' else 'activity_index')
    elif ref_id in {'tampa_matrix','river_moving_matrix'}:
        keys=['day','hour_utc'];assign(x,'day');assign(y,'hour_utc');assign(z,'vessel_ids' if ref_id=='tampa_matrix' else 'moving_vessel_ids')
    elif ref_id=='elsa_longitude':
        keys=['hour'];assign(x,'hour');assign(y,'reported_longitude')
    elif ref_id=='river_speeds':
        keys=['hour','vessel_id'];assign(x,'hour');assign(y,'mean_speed');assign(z,'vessel_id')
    elif ref_id.endswith('_replay'):keys=['vessel_id']
    elif ref_id=='river_slow_spells':keys=['vessel_id','started_at','ended_at']
    aliases_known={'observationcount':'observations','count':'observations','n':'observations','annualsharepct':'annual_share_pct','monthlysharepct':'annual_share_pct','sharepct':'annual_share_pct','baseline':'baseline','baselineavg':'baseline','vesselcount':'vessel_ids','vessels':'vessel_ids','vesselids':'vessel_ids','activityindex':'activity_index','durationhours':'duration_hours','durationh':'duration_hours','hours':'duration_hours','reportcount':'reports','reports':'reports','name':'name_label','namelabel':'name_label','vesselname':'name_label','vesseltype':'type_label','typelabel':'type_label','start':'start_time','end':'end_time','starttime':'start_time','endtime':'end_time','startedat':'started_at','endedat':'ended_at','runstart':'started_at','runend':'ended_at','meanspeed':'mean_speed','avgspeed':'mean_speed','averagespeed':'mean_speed'}
    for column in card['data']['columns']:
        if column in out:continue
        role=role_from_expr(exprs[column]) if column in exprs else None
        if role in {'reports','observations'} and ref_id not in {'river_slow_spells','elsa_replay','river_replay'}:role='observations'
        if role is None:role=aliases_known.get(textkey(column),column)
        if ref_id=='river_slow_spells':role={'start_time':'started_at','end_time':'ended_at'}.get(role,role)
        out[column]=role
    return keys,out,exprs

def tolerance(role,expr=None):
    if role in {'observations','vessel_ids','moving_vessel_ids','reports','vessel_id','cell_size_deg'}:return 0
    precision=2 if role in {'duration_hours','annual_share_pct'} else 1 if role=='activity_index' else None
    if expr:
        for node in expr.find_all(exp.Round):
            decimals=node.args.get('decimals')
            if isinstance(decimals,exp.Literal) and not decimals.is_string:
                value=int(decimals.this);precision=value if precision is None else min(precision,value)
    return .5*10**(-precision)+1e-7 if precision is not None else 1e-6

def equal(a,b,role,expr=None):
    if a is None or b is None:return a is b
    if role in {'month','day','hour','region','vessel_id','started_at','ended_at','start_time','end_time','latitude_band','lon_bin','lat_bin','hour_utc'}:
        try:return key_value(role,a)==key_value(role,b)
        except (ValueError,TypeError):return False
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=0,abs_tol=tolerance(role,expr))
    return str(a).strip()==str(b).strip()

def table_check(card,ref_id,reference,supplement=None):
    expected=records(reference);keys,mapping,exprs=role_columns(card,ref_id)
    if not keys:raise ValueError('No comparison contract for '+ref_id)
    if ref_id.endswith('_replay') and supplement:expected=records(supplement)
    expected=[dict(r, name_label=r.get('name_label',r.get('vessel_name')), type_label=r.get('type_label',r.get('vessel_type'))) for r in expected]
    def rk(row):return tuple(key_value(k,row[k]) for k in keys)
    expected_map={rk(row):row for row in expected};actual_map={};errors=[];unverified=set();compared=0
    for raw in records(card):
        row={}
        for column,value in raw.items():
            role=mapping[column]
            if role=='vessel_id':value=vessel(value)
            if role in row and not equal(row[role],value,role):errors.append({'duplicate_role':role,'values':[row[role],value]})
            row[role]=value
        try:key=rk(row)
        except (KeyError,ValueError,TypeError) as exc:errors.append({'key_error':str(exc)});continue
        if key in actual_map:errors.append({'duplicate_key':key})
        actual_map[key]=row
        baseline=expected_map.get(key)
        if baseline is None:continue
        for column,value in raw.items():
            role=mapping[column]
            if role in baseline:
                compared+=1
                if not equal(value,baseline[role],role,exprs.get(column)):
                    errors.append({'key':key,'column':column,'role':role,'actual':value,'expected':baseline[role]})
            elif role=='scientific_name' and ref_id.startswith('monarch') and value=='Danaus plexippus':continue
            elif role=='cell_size_deg':
                required=5 if ref_id.startswith('monarch') else 2 if 'two_boxes' in ref_id else 1
                if value!=required:errors.append({'cell_size_deg':value,'expected':required})
            elif column in exprs and isinstance(exprs[column],exp.Literal):
                if evaluate(exprs[column],{})!=value:errors.append({'constant':column,'actual':value})
            else:unverified.add(column)
    missing=set(expected_map)-set(actual_map);extra=set(actual_map)-set(expected_map)
    return not errors and not missing and not extra and not unverified,{'expected_rows':len(expected),'actual_rows':len(records(card)),'compared_values':compared,'missing_keys':len(missing),'extra_keys':len(extra),'unverified_columns':sorted(unverified),'examples':errors[:5],'column_roles':mapping}


def segmentation_checks(tree):
    aliases=alias_map(tree);out=[]
    named={n.this.name:n for n in tree.find_all(exp.Window) if isinstance(n.this,exp.Identifier)}
    lagged={};cumulative=[]
    for i,window in enumerate(tree.find_all(exp.Window)):
        if fname(window.this) in {'lag','laginframe'}:lagged['lag_'+str(i)]=window
    for window in tree.find_all(exp.Window):
        if fname(window.this) in {'sum','sumif','countif'}:cumulative.append(window)
    out.append(('lag_and_cumulative_segments',bool(lagged and cumulative),{}))
    for label,window in list(lagged.items())+[(f'cumulative_{i}',w) for i,w in enumerate(cumulative)]:
        definition=named.get(window.args.get('alias').name) if window.args.get('alias') else window
        partition={c.name for n in definition.args.get('partition_by',[]) for c in n.find_all(exp.Column)}
        order=definition.args.get('order');ordered=list(order.expressions) if order else []
        valid='vessel_id' in partition and any('observed_at' in {c.name for c in n.find_all(exp.Column)} and not n.args.get('desc') for n in ordered)
        out.append(('chronological_partition_'+label,valid,{}))
    if cumulative:
        candidate=cumulative[-1].this
        formula=candidate.this if isinstance(candidate,exp.Sum) else (candidate.expressions[-1] if candidate.expressions else candidate.this)
        # Both valid forms occur in these demos: increment at a fast/unknown row
        # before filtering it out, or increment at the next slow row. Test actual
        # membership of the surrounding slow reports, not the break's placement.
        trials=[]
        scenarios=[('600-second gap',[(.2,0),(.2,600)],False),('601-second gap',[(.2,0),(.2,601)],True),('fast report between slow fixes',[(.2,0),(1,60),(.2,120)],True),('unknown speed between slow fixes',[(.2,0),(None,60),(.2,120)],True)]
        for label,sequence,expected in scenarios:
            stamp=utc('2024-10-09 12:00:37');previous={'speed_knots':.2,'observed_at':stamp-timedelta(seconds=60),'vessel_id':'test'}
            ids=[];run=0;unknown=False
            for speed,offset in sequence:
                row={'speed_knots':speed,'observed_at':stamp+timedelta(seconds=offset),'vessel_id':'test','__previous__':previous}
                value=evaluate(formula,row,aliases)
                if value is UNKNOWN or value is None:unknown=True
                else:run+=int(value)
                if speed is not None and speed<1:ids.append(run)
                previous=row
            split=None if unknown or len(ids)<2 else ids[0]!=ids[-1]
            trials.append({'scenario':label,'expected_split':expected,'actual_split':split})
        out.append(('breaks_fast_unknown_and_over_10_minute_gaps',all(t['actual_split']==t['expected_split'] for t in trials),{'trials':trials}))
    threshold=[]
    for node in tree.walk():
        if not isinstance(node,(exp.GT,exp.GTE,exp.LT,exp.LTE)):continue
        resolved=resolve(node,aliases)
        if not resolved.find(exp.DateDiff):continue
        # Ignore gap predicates: they compare individual reports rather than run extrema.
        if not resolved.find(exp.Min) or not resolved.find(exp.Max):continue
        if resolved.find(exp.Round):threshold.append(False)
        else:
            # Duration thresholds must compare seconds or a raw second-derived value;
            # DateDiff minute/hour boundary counts can admit runs shorter than six hours.
            units=[str(n.args.get('unit','')).lower().strip("'") for n in resolved.find_all(exp.DateDiff)]
            condition=resolved.transform(lambda n:exp.column('__start__') if isinstance(n,exp.Min) else exp.column('__end__') if isinstance(n,exp.Max) else n)
            start=utc('2024-10-09 01:02:37')
            results=[evaluate(condition,{'__start__':start,'__end__':start+timedelta(seconds=seconds)}) for seconds in (21599,21600,21601)]
            threshold.append(all(u=='second' for u in units) and results==[False,True,True])
    out.append(('six_hour_threshold_uses_unrounded_seconds',bool(threshold) and all(threshold),{}))
    order=tree.args.get('order');first=resolve(order.expressions[0].this,aliases) if order and order.expressions else None
    valid=first is not None and first.find(exp.DateDiff) is not None and first.find(exp.Round) is None and order.expressions[0].args.get('desc') is True
    out.append(('ranking_uses_unrounded_duration',valid,{}))
    return out


def semantic_checks(card,reference,ref_id):
    gold_sql=reference['sql']
    if ref_id=='fungi_sydney_may_gallery_diverse':
        # Pinned SQLGlot cannot parse ClickHouse's two LIMIT clauses. This is
        # solely the trusted reference's cohort predicate; actual gallery IDs,
        # count and per-name cap are compared separately to independent rows.
        gold_sql=re.sub(r'\s+LIMIT\s+2\s+BY\s+scientific_name\s+LIMIT\s+12\s*$', ' LIMIT 12',gold_sql,flags=re.I)
    tree=parse(card['sql']);gold=parse(gold_sql);checks=[]
    ok,details=scope_check(tree,gold,ref_id);checks.append(('source_cohort_scope',ok,details))
    if ref_id in {'raw_ports','normalized_ports','tampa_matrix','river_moving_matrix'}:
        exact=any((fname(n)=='uniqexact' and any(c.name=='vessel_id' for c in n.find_all(exp.Column))) or (isinstance(n,exp.Count) and isinstance(n.this,exp.Distinct) and any(c.name=='vessel_id' for c in n.find_all(exp.Column))) for n in tree.find_all(exp.Func))
        checks.append(('exact_distinct_vessel_measure',exact,{}))
    if ref_id=='elsa_longitude':
        ok=any(fname(n)=='argmax' and {'longitude','observed_at'}<={c.name for c in n.find_all(exp.Column)} for n in tree.find_all(exp.Func))
        checks.append(('last_reported_longitude_not_average',ok,{}))
    if ref_id=='river_speeds':
        checks.append(('mean_reported_speed',any(isinstance(n,exp.Avg) and any(c.name=='speed_knots' for c in n.find_all(exp.Column)) for n in tree.find_all(exp.Func)),{}))
        groups=[resolve(n,alias_map(tree)) for n in tree.args.get('group',exp.Group()).expressions]
        columns={c.name for n in groups for c in n.find_all(exp.Column)}
        checks.append(('hour_and_vessel_id_group_keys',{'observed_at','vessel_id'}<=columns and not any(n.find(exp.AggFunc) for n in groups),{}))
    if ref_id=='fungi_normalized_seasons':
        ok=any(fname(n.this)=='sum' and n.args.get('partition_by') for n in tree.find_all(exp.Window))
        checks.append(('per_region_annual_normalization',ok,{}))
    if ref_id=='river_slow_spells':checks.extend(segmentation_checks(tree))
    return checks


def gallery_check(card,story_id,reference,supplement):
    expected=records(reference);by_id={str(r['observation_uuid']):r for r in records(supplement or {})}
    source='complete independently queried photo-eligible cohort' if by_id else 'reference ordered selection only'
    if not by_id:by_id={str(r['observation_uuid']):r for r in expected}
    errors=[];seen=set();licenses=[];names=Counter()
    for row in records(card):
        ident=str(row.get('observation_uuid',''));known=by_id.get(ident)
        if not known:errors.append({'unknown_observation_uuid':ident});continue
        if ident in seen:errors.append({'duplicate_observation_uuid':ident})
        seen.add(ident);names[row.get('scientific_name')]+=1
        for field in ('scientific_name','observed_on','latitude','longitude'):
            if field not in row or not equal(row.get(field),known.get(field),'day' if field=='observed_on' else field):errors.append({'id':ident,'field':field,'expected':known.get(field),'actual':row.get(field)})
        for field in ('photo_id','photo_license','photo_credit'):
            if field not in known:errors.append({'id':ident,'missing_independent_reference':field})
            elif str(row.get(field,''))!=str(known[field]):errors.append({'id':ident,'field':field,'expected':known[field],'actual':row.get(field)})
        license=row.get('photo_license');licenses.append(license)
        if license not in {'CC0','CC-BY'}:errors.append({'id':ident,'unsupported_license':license})
        pid=row.get('photo_id');extension=known.get('extension')
        if extension and row.get('photo_url')!=f'https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{extension}':errors.append({'id':ident,'invalid_photo_url':row.get('photo_url')})
        if row.get('photo_page')!=f'https://www.inaturalist.org/photos/{pid}':errors.append({'id':ident,'invalid_photo_page':row.get('photo_page')})
    if len(seen)!=12:errors.append({'expected_distinct_gallery_rows':12,'actual':len(seen)})
    reference_ids={str(r['observation_uuid']) for r in expected}
    if seen!=reference_ids:errors.append({'selected_ids_differ_from_reference':{'missing':len(reference_ids-seen),'extra':len(seen-reference_ids)}})
    if story_id=='fungi-opposite-seasons' and max(names.values(),default=0)>2:errors.append({'max_per_scientific_name':dict(names.most_common(8))})
    return not errors,{'rows':len(seen),'independent_cohort_rows':len(by_id),'reference_coverage':source,'licenses':dict(Counter(licenses)),'examples':errors[:6]}


def playback_check(response,story_id,supplemental):
    """Audit the exact two prepared ship replays against independent raw fixes."""
    original=records(supplemental.get(story_id+'_raw_positions',{}))
    storms=records(supplemental.get('milton_raw_positions',{}))
    rows=records(response);overlay=records(response.get('storm',{}));errors=[]
    if not original or not storms:return False,{'error':'Independent raw playback reference is missing'}
    sample=response.get('sampling',{});interval=sample.get('interval_seconds',0)
    maximum=1200 if story_id=='elsa-and-milton' else 180
    if not isinstance(interval,int) or not 60<=interval<=maximum:return False,{'error':'Unexpected playback sampling interval','interval_seconds':interval}
    def pointkey(row):return str(row['vessel_id']),utc(row['observed_at']).timestamp(),float(row['longitude']),float(row['latitude'])
    def bucket(row):return str(row['vessel_id']),math.floor(utc(row['observed_at']).timestamp()/interval)
    expected={}
    for row in original:
        key=bucket(row)
        if key not in expected or pointkey(row)[1:]>pointkey(expected[key])[1:]:expected[key]=row
    actual={};compared=0
    for row in rows:
        key=bucket(row)
        if key in actual:errors.append({'duplicate_bucket':key})
        actual[key]=row;known=expected.get(key)
        if not known:errors.append({'unexpected_bucket':key});continue
        for field in ('vessel_id','observed_at','longitude','latitude','speed_knots','course_deg','heading_deg','vessel_name','vessel_type'):
            compared+=1
            if not equal(row.get(field),known[field],'hour' if field=='observed_at' else field):errors.append({'bucket':key,'field':field,'expected':known[field],'actual':row.get(field)})
    if set(actual)!=set(expected):errors.append({'missing_buckets':len(set(expected)-set(actual)),'extra_buckets':len(set(actual)-set(expected))})
    start=min(utc(r['observed_at']) for r in original);end=max(utc(r['observed_at']) for r in original)
    bounds={'west':min(r['longitude'] for r in original),'east':max(r['longitude'] for r in original),'south':min(r['latitude'] for r in original),'north':max(r['latitude'] for r in original)}
    for field,value in bounds.items():
        if not equal(response.get('bounds',{}).get(field),value,field):errors.append({'bounds':field,'expected':value,'actual':response.get('bounds',{}).get(field)})
    if not equal(response.get('range',{}).get('start'),start.isoformat(),'hour') or not equal(response.get('range',{}).get('end'),end.isoformat(),'hour'):errors.append({'error':'Playback range differs from independent source extent'})
    identifiers={str(r['vessel_id']) for r in original}
    if {str(r['vessel_id']) for r in rows}!=identifiers:errors.append({'error':'Playback changed the selected vessel cohort'})
    if sample.get('source_positions')!=len(original) or sample.get('returned_positions')!=len(rows) or sample.get('selected_tracks')!=len(identifiers):errors.append({'error':'Playback sampling counts disagree with source/result'})
    if len(rows)>12000 or len(identifiers)>24 or response.get('truncated') is not False:errors.append({'error':'Playback bounds/truncation contract failed'})
    if 'disconnect' not in sample.get('gap_policy','').lower():errors.append({'error':'Missing long-gap disconnection disclosure'})
    known_storms=[r for r in storms if start<=utc(r['observed_at'])<=end and bounds['west']-8<=r['longitude']<=bounds['east']+8 and bounds['south']-8<=r['latitude']<=bounds['north']+8]
    def stormkey(row):return row['storm_id'],utc(row['observed_at']).isoformat()
    expected_storm={stormkey(r):r for r in known_storms};actual_storm={stormkey(r):r for r in overlay}
    if len(actual_storm)!=len(overlay) or set(expected_storm)!=set(actual_storm):errors.append({'error':'Milton overlay point IDs/times differ from independent reference','expected_rows':len(expected_storm),'actual_rows':len(overlay)})
    for key,row in actual_storm.items():
        known=expected_storm.get(key)
        if known:
            for field,value in known.items():
                compared+=1
                if not equal(row.get(field),value,'hour' if field=='observed_at' else field):errors.append({'storm':key,'field':field,'expected':value,'actual':row.get(field)})
    if response.get('storm_overlay',{}).get('names')!=['MILTON']:errors.append({'error':'Unexpected storm names'})
    return not errors,{'source_positions':len(original),'returned_positions':len(rows),'storm_positions':len(overlay),'compared_values':compared,'interval_seconds':interval,'examples':errors[:5]}


def load_references(report_path,reference_path=None,wildlife_ref=None,maritime_ref=None,supplemental_ref=None):
    # Separate files remain an explicit option for rebuilding reference evidence;
    # committed tests never depend on machine-specific temporary paths.
    separate=(wildlife_ref,maritime_ref,supplemental_ref)
    if any(separate) and not all(separate):
        raise ValueError('Provide all three separate reference files, or --references.')
    if reference_path and any(separate):
        raise ValueError('Choose consolidated or separate reference files, not both.')
    candidate=Path(reference_path or DEFAULT_REFERENCES) if not any(separate) else None
    if candidate is not None:
        root=read(candidate);wild=root['wildlife'];marine=root['maritime'];supp=root.get('supplemental',[])
    else:
        wild=read(wildlife_ref);marine=read(maritime_ref);supp=read(supplemental_ref)
    refs=wild['query_evidence']|marine['reference_queries']
    supplemental={r['label']:r for r in supp} if isinstance(supp,list) else supp
    return refs,supplemental,str(candidate) if candidate else 'individual reference files'


def verify_report(path=DEFAULT_REPORT,reference_path=None,wildlife_ref=None,maritime_ref=None,supplemental_ref=None,stories=None):
    """Return an offline audit; optional stories accepts IDs or comma-separated IDs.

    A filtered audit ignores other stories' failures but fails when a requested
    story is absent. It does not turn partial rehearsals into successful ones.
    """
    requested=None
    if stories is not None:
        values=[stories] if isinstance(stories,str) else stories
        requested={part.strip() for value in values for part in value.split(',') if part.strip()}
        if not requested or requested-set(MAPPINGS):
            raise ValueError('Unknown or empty story selection: '+', '.join(sorted(requested-set(MAPPINGS))))
    report=read(path);refs,supp,provenance=load_references(path,reference_path,wildlife_ref,maritime_ref,supplemental_ref)
    output={'report':str(path),'references':provenance,'method':'Offline comparison with independent read-only references; no app/model calls. Numeric keys ignore row/column order; explicit rounding tolerances; raw SQL scope/segmentation checks.','stories':[],'limitations':['Playback positions are checked only when responses were captured. Image pixels and visual motion remain separate live UI checks. SQL checks cover the prepared stories, not arbitrary query equivalence.']}
    for story in report.get('stories',[]):
        if requested is not None and story.get('id') not in requested:continue
        sid=story['id'];item={'id':sid,'checks':[]};output['stories'].append(item);previous={}
        def check(step,name,ok,details=None):item['checks'].append({'step':step,'check':name,'passed':bool(ok),**({'details':details} if details else {})})
        if sid not in MAPPINGS:
            check('*','known_story',False);item['passed']=False;continue
        for step in story.get('steps',[]):
            step_id=step['id'];card=step.get('card');ref_id=MAPPINGS[sid].get(step_id)
            if not card:
                if step.get('passed'):check(step_id,'live_action_completed',True)
                continue
            if not ref_id:check(step_id,'known_reference',False);continue
            reference=refs[ref_id]
            try:
                if card['chart']=='gallery':
                    ok,detail=gallery_check(card,sid,reference,supp.get(sid+'_gallery_cohort'));check(step_id,'gallery_observations_and_licenses',ok,detail)
                else:
                    ok,detail=table_check(card,ref_id,reference,supp.get(sid+'_summary_metrics') if ref_id.endswith('_replay') else None);check(step_id,'all_returned_values',ok,detail)
                for name,ok,detail in semantic_checks(card,reference,ref_id):check(step_id,name,ok,detail)
                target=LOCAL_TARGETS.get((sid,step_id))
                if target:
                    original=previous.get(target)
                    ok=bool(original) and card['id']==original['id'] and card['data']==original['data'] and card['sql']==original['sql']
                    events=step.get('events',[])
                    ok=ok and any(e.get('type')=='local-request' for e in events) and not any(e.get('type')=='paint' for e in events)
                    check(step_id,'local_change_reuses_exact_data_without_model',ok)
                if sid=='elsa-and-milton' and step_id=='index':check(step_id,'explicit_update_preserves_card_id',previous.get('ports',{}).get('id')==card['id'])
            except Exception as exc:check(step_id,'validator_completed',False,{'error':type(exc).__name__+': '+str(exc)})
            previous[step_id]=card
        if sid in {'elsa-and-milton','mississippi-motion-and-stillness'}:
            ship_responses=[r for r in story.get('playback_responses',[]) if r.get('rows') and 'vessel_id' in r.get('columns',[])]
            for i,response in enumerate(ship_responses):
                try:
                    ok,details=playback_check(response,sid,supp);check('replay','captured_playback_'+str(i),ok,details)
                except Exception as exc:check('replay','captured_playback_'+str(i),False,{'error':type(exc).__name__+': '+str(exc)})
            if not ship_responses:item['playback']='not captured; card summary and SQL scope checked'
        required=set(MAPPINGS[sid]);seen={s['id'] for s in story.get('steps',[]) if s.get('card')}
        check('*','all_story_steps_present',required<=seen,{'missing':sorted(required-seen)})
        check('*','live_rehearsal_passed',story.get('passed') is True and not story.get('errors'),{'failure':story.get('failure')} if story.get('failure') else None)
        item['passed']=all(c['passed'] for c in item['checks'])
    if requested is not None:
        absent=requested-{s['id'] for s in output['stories']}
        for sid in sorted(absent):
            output['stories'].append({'id':sid,'checks':[{'step':'*','check':'requested_story_present','passed':False}],'passed':False})
    output['checks']=sum(len(s['checks']) for s in output['stories']);output['failed_checks']=sum(not c['passed'] for s in output['stories'] for c in s['checks'])
    output['passed']=bool(output['stories']) and all(s.get('passed',False) for s in output['stories'])
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',nargs='?',default=DEFAULT_REPORT,help='Rehearsal .json or .json.gz; default demos/rehearsal-evidence.json.gz')
    parser.add_argument('--references',help='Consolidated .json or .json.gz; default demos/reference-data.json.gz')
    parser.add_argument('--wildlife-ref');parser.add_argument('--maritime-ref');parser.add_argument('--supplemental-ref');parser.add_argument('--output')
    parser.add_argument('--stories',nargs='+',help='Audit only these story IDs; spaces and commas are accepted')
    args=parser.parse_args()
    try:result=verify_report(args.report,args.references,args.wildlife_ref,args.maritime_ref,args.supplemental_ref,stories=args.stories)
    except (OSError,ValueError,KeyError) as exc:parser.error(str(exc))
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(encoded)
    print(encoded,end='');raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
