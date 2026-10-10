import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selection import build_scope,apply_scope,parse
from sqlglot import exp
class MarkedData(unittest.TestCase):
 def setUp(self):
  self.card={'id':'a','revision':'r','chart':'heatmap','sql':"SELECT toMonth(observed_on) AS month,floor(latitude/10)*10+5 AS band,count() AS n FROM exploration.wildlife WHERE taxon_id=48662 AND observed_on >= '2025-01-01' AND observed_on < '2026-01-01' GROUP BY month,band",'data':{'columns':['month','band','n'],'rows':[[1,35,10],[2,45,20],[1,45,30]]}}
  self.mark={'kind':'points','card_id':'a','revision':'r','row_indices':[0,1]}
 def scope(self):return build_scope(self.mark,{'cards':[self.card]},'a')
 def test_exact_pairs_and_original_filters(self):
  p=self.scope()['predicate'];self.assertIn('((1, 35), (2, 45))',p);self.assertNotIn('(1, 45)',p);self.assertIn('taxon_id = 48662',p);self.assertIn("observed_on < '2026-01-01'",p)
 def test_foreign_stale_invalid_and_empty_marks_rejected(self):
  for change in [{'card_id':'other'},{'revision':'old'},{'row_indices':[]},{'row_indices':[-1]},{'row_indices':[99]},{'row_indices':[True]}]:
   with self.subTest(change=change),self.assertRaises(ValueError):build_scope(self.mark|change,{'cards':[self.card]},'a')
 def test_gallery_receives_same_scope(self):
  sql=apply_scope('SELECT toString(observation_uuid) AS observation_uuid FROM exploration.wildlife_gallery LIMIT 36',self.scope());tree=parse(sql)
  self.assertEqual([t.name for t in tree.find_all(exp.Table)],['wildlife_gallery']);self.assertIn('((1, 35), (2, 45))',sql)
 def test_scope_cannot_leave_source(self):
  for sql in ['SELECT count() FROM exploration.trips','SELECT 1','SELECT * FROM system.users','SELECT * FROM url(\'https://example.org\')']:
   with self.subTest(sql=sql),self.assertRaises(ValueError):apply_scope(sql,self.scope())
 def test_scope_applied_to_all_join_inputs(self):
  sql=apply_scope('SELECT a.taxon_id FROM exploration.wildlife a JOIN exploration.wildlife b USING(taxon_id)',self.scope());self.assertEqual(sql.count('taxon_id = 48662'),2)
 def test_chained_mark_preserves_prior_scope(self):
  self.card['sql']=apply_scope(self.card['sql'],self.scope());self.mark['row_indices']=[0];sql=self.scope()['predicate'];self.assertIn('((1, 35), (2, 45))',sql);self.assertIn('((1, 35))',sql)
 def test_map_region_is_raw_bbox(self):
  self.card['chart']='map';self.mark={'kind':'region','card_id':'a','revision':'r','bounds':{'west':-100,'east':-90,'south':25,'north':35}};p=self.scope()['predicate'];self.assertIn('longitude BETWEEN -100 AND -90',p);self.assertIn('latitude BETWEEN 25 AND 35',p)
  self.mark['bounds']['north']=100
  with self.assertRaises(ValueError):self.scope()
 def test_null_and_quotes_are_sql_literals(self):
  self.card['sql']='SELECT scientific_name, count() AS n FROM exploration.wildlife GROUP BY scientific_name';self.card['data']={'columns':['scientific_name','n'],'rows':[["O'Brian",4],[None,1]]};p=self.scope()['predicate'];self.assertIn('IS NULL',p);self.assertIn("O''Brian",p)
 def test_nested_aggregation_refused(self):
  self.card['sql']='SELECT * FROM (SELECT taxon_id,count() AS n FROM exploration.wildlife GROUP BY taxon_id)'
  with self.assertRaises(ValueError):self.scope()
if __name__=='__main__':unittest.main()
