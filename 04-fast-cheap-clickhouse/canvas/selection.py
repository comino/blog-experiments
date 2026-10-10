"""Derive drill-through scope from saved results, never from model-provided filters."""
import math
import sqlglot
from sqlglot import exp

FAMILIES = {'wildlife': {'wildlife', 'wildlife_gallery'}, 'wildlife_gallery': {'wildlife', 'wildlife_gallery'},
            'trips': {'trips'}, 'ais_positions': {'ais_positions'}, 'storm_tracks': {'storm_tracks'}}

def parse(sql):
    statements = sqlglot.parse(sql, read='clickhouse')
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError('Mark a view based on a single SELECT query.')
    return statements[0]

def literal(value):
    if value is None: return exp.Null()
    if isinstance(value, bool): return exp.Boolean(this=value)
    if isinstance(value, (int, float)) and math.isfinite(value): return exp.Literal.number(value)
    if isinstance(value, str): return exp.Literal.string(value)
    raise ValueError('This selected value cannot be used as a filter.')

def expand(expression, aliases, seen=frozenset()):
    def replace(node):
        if isinstance(node, exp.Column):
            name = node.name
            if not node.table and name in aliases and name not in seen:
                return expand(aliases[name], aliases, seen | {name})
            node = node.copy()
            node.set('table', None); node.set('db', None); node.set('catalog', None)
            return node
        return node
    return expression.copy().transform(replace)

def source(select):
    """Flatten only transparent single-table subqueries, including our own scope wrappers."""
    predicates=[]; current=select
    while True:
        if current.args.get('joins') or current.args.get('with_'):
            raise ValueError('Drill-through is available for views based on one source table. Mark a simpler view first.')
        aliases={e.alias_or_name:e.unalias() for e in current.expressions if e.alias_or_name}
        for condition in ('prewhere','where'):
            where=current.args.get(condition)
            if where:
                if where.find(exp.Select):
                    raise ValueError('Cannot safely drill through a subquery filter. Use a directly filtered source view instead.')
                predicates.append(expand(where.this,aliases))
        from_=current.args.get('from_')
        target=from_.this if from_ else None
        if isinstance(target,exp.Table):
            if target.args.get('sample'):
                raise ValueError('Cannot preserve a sampled source for drill-through. Use a filtered source view instead.')
            if target.db != 'exploration' or target.name not in FAMILIES:
                raise ValueError('This view does not have a supported data source for marking.')
            return target.name,predicates,aliases
        if isinstance(target,exp.Subquery) and isinstance(target.this,exp.Select):
            current=target.this
            if current.args.get('group') or current.args.get('limit') or current.args.get('distinct') or not all(isinstance(e,exp.Star) for e in current.expressions):
                raise ValueError('Cannot safely drill through this nested aggregation. Mark a source view instead.')
        else:
            raise ValueError('Cannot identify the source rows for this view.')

def build_scope(mark, state, selected_id):
    if not mark: return None
    if not isinstance(mark,dict) or mark.get('card_id') != selected_id:
        raise ValueError('Select the marked card before asking about it.')
    card=next((c for c in state['cards'] if c['id']==selected_id),None)
    if not card or mark.get('revision') != card['revision']:
        raise ValueError('That view changed after you marked it. Mark the updated data again.')
    if mark.get('kind') not in {'points','region'}: raise ValueError('Unsupported mark type.')
    tree=parse(card['sql']); base,predicates,_=source(tree)
    columns,rows=card['data']['columns'],card['data']['rows']
    aliases={e.alias_or_name:e.unalias() for e in tree.expressions if e.alias_or_name}
    indices=[]
    if mark.get('kind')=='region':
        if card['chart'] not in {'map','density','tracks'} or base not in {'wildlife','wildlife_gallery','ais_positions','storm_tracks'}:
            raise ValueError('Draw a region on a wildlife or maritime map.')
        box=mark.get('bounds',{})
        if set(box)!={'west','east','south','north'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in box.values()):
            raise ValueError('Invalid map selection.')
        if not (-180<=box['west']<box['east']<=180 and -90<=box['south']<box['north']<=90):
            raise ValueError('Choose a map area within the world bounds.')
        predicates += [exp.Between(this=exp.column('longitude'),low=literal(box['west']),high=literal(box['east'])), exp.Between(this=exp.column('latitude'),low=literal(box['south']),high=literal(box['north']))]
        if card['chart']=='tracks':
            # A replay is a selected cohort (often top 12 ships), so its region
            # must not quietly expand to every other vessel in the same area.
            key='vessel_id' if base=='ais_positions' else 'storm_id'
            if key not in columns:
                raise ValueError('This track view does not include its source identifiers.')
            identifiers=list(dict.fromkeys(row[columns.index(key)] for row in rows))
            if not identifiers or any(type(value) not in (str,int) or not 1<=len(str(value))<=64 for value in identifiers):
                raise ValueError('Choose a track view with valid source identifiers.')
            predicates.append(exp.In(this=exp.column(key),expressions=[literal(value) for value in identifiers]))
        description='Marked map area'
        selected={'bounds':box}
    else:
        indices=mark.get('row_indices')
        if not isinstance(indices,list) or not indices or len(indices)>5000 or any(type(i) is not int or i<0 or i>=len(rows) for i in indices):
            raise ValueError('Choose at least one point from the current result.')
        indices=sorted(set(indices))
        group=tree.args.get('group')
        dimensions=[]
        if group:
            expressions=group.expressions
            if group.args.get('all'):
                expressions=[e.unalias() for e in tree.expressions if not e.find(exp.AggFunc) and not e.find(exp.Window)]
            for expression in expressions:
                if isinstance(expression,exp.Literal) and not expression.is_string:
                    expression=tree.expressions[int(expression.this)-1].unalias()
                resolved=expand(expression,aliases)
                name=next((n for n,e in aliases.items() if expand(e,aliases).sql(dialect='clickhouse')==resolved.sql(dialect='clickhouse') and n in columns),None)
                if name is None: raise ValueError('The grouped key is not in this result; include it before marking.')
                dimensions.append((resolved,columns.index(name)))
        elif 'observation_uuid' in columns:
            dimensions=[(exp.column('observation_uuid'),columns.index('observation_uuid'))]
        else:
            dimensions=[(expand(expression,aliases),columns.index(name)) for name,expression in aliases.items() if name in columns and not expression.find(exp.AggFunc) and not expression.find(exp.Window) and not expression.find(exp.Select)]
        if not dimensions: raise ValueError('This summary has no keys to mark. Add a breakdown first.')
        values=[];seen=set()
        for i in indices:
            item=tuple(rows[i][column] for _,column in dimensions)
            if item not in seen: seen.add(item);values.append(item)
        regular=[v for v in values if None not in v]
        clauses=[]
        if regular:
            lhs=exp.Tuple(expressions=[e.copy() for e,_ in dimensions])
            clauses.append(exp.In(this=lhs,expressions=[exp.Tuple(expressions=[literal(v) for v in item]) for item in regular]))
        for item in values:
            if None not in item: continue
            clauses.append(exp.and_(*[(exp.Is(this=e.copy(),expression=exp.Null()) if value is None else exp.EQ(this=e.copy(),expression=literal(value))) for (e,_),value in zip(dimensions,item)]))
        predicates.append(exp.or_(*clauses))
        description=f'{len(indices)} marked '+('cell' if card['chart'] in {'heatmap','surface'} else 'point')+('s' if len(indices)!=1 else '')
        selected={'columns':columns,'rows':[rows[i] for i in indices[:8]],'count':len(indices)}
    predicate=exp.and_(*predicates).sql(dialect='clickhouse')
    if len(predicate)>150000: raise ValueError('Choose a smaller group of points for this follow-up.')
    return {'base_table':base,'predicate':predicate,'description':description,'selected':selected,'card_id':card['id'],'revision':card['revision'],'row_indices':indices}

def apply_scope(sql, scope):
    if not scope:return sql
    tree=parse(sql);tables=list(tree.find_all(exp.Table));family=FAMILIES[scope['base_table']]
    ctes={c.alias_or_name for c in tree.find_all(exp.CTE)};count=0
    for table in tables:
        if table.name in ctes and not table.db:continue
        if table.db!='exploration' or table.name not in family:
            raise ValueError('This request leaves the marked data. Clear the mark to explore a different source.')
        replacement=sqlglot.select('*').from_(f'exploration.{table.name}').where(scope['predicate'],dialect='clickhouse').subquery(table.alias_or_name)
        table.replace(replacement);count+=1
    if not count:raise ValueError('A marked-data request must query its source rows. Clear the mark to ask an unrelated question.')
    for column in tree.find_all(exp.Column):
        if column.db=='exploration':column.set('db',None)
    return tree.sql(dialect='clickhouse')
