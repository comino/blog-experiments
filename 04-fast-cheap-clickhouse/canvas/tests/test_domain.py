import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from domain import blank, apply_operations, validate_plan, resolve_intent

class Intent(unittest.TestCase):
    def test_new_questions_keep_the_selected_view(self):
        for prompt in ['Show monthly counts', 'Which species are most common?', 'Show photos from these marked cells', 'What about Europe?', 'Please map monarchs in 2025']:
            with self.subTest(prompt=prompt):self.assertEqual(resolve_intent(prompt,'selected'),'add')

    def test_explicit_edits_target_the_selection(self):
        for prompt in ['Only 2024, keep the same map.', 'Make it 3d', 'Show this as bars', 'Could you please change this map to monthly counts?', 'Filter to Europe', 'Show only birds', 'Break this down by month', 'Use blue', 'Add a legend']:
            with self.subTest(prompt=prompt):self.assertEqual(resolve_intent(prompt,'selected'),'update')

    def test_comparisons_forced_add_and_empty_canvas(self):
        self.assertEqual(resolve_intent('Make it a bar chart alongside this one','selected'),'add')
        self.assertEqual(resolve_intent('Only 2024','selected','add'),'add')
        self.assertEqual(resolve_intent('Only 2024',None),'add')

    def test_comparison_noun_does_not_override_an_explicit_edit(self):
        for prompt in ['Change this comparison chart to an activity index.',
                       'Update the comparison to show percentages.',
                       'Make this a comparison of monthly shares.']:
            with self.subTest(prompt=prompt):self.assertEqual(resolve_intent(prompt,'selected'),'update')
        for prompt in ['Compare Tampa with Port Everglades.',
                       'Show a comparison of the two ports.',
                       'Change this comparison into a new chart alongside it.']:
            with self.subTest(prompt=prompt):self.assertEqual(resolve_intent(prompt,'selected'),'add')

class Transactions(unittest.TestCase):
    def setUp(self):
        self.state = blank()
        self.op = {'action':'add','title':'Observations by month','sql':'SELECT 1','chart':'line','x':'month','y':'observations','z':'','color':'teal'}
        self.result = {'data':{'columns':['month','observations'],'rows':[['2025-01-01',7]]},'query_ms':12}
        self.state, _ = apply_operations(self.state,[self.op],[self.result],'Start')
        self.id = self.state['cards'][0]['id']

    def test_iteration_preserves_identity_and_layout_and_old_version(self):
        old=copy.deepcopy(self.state['cards'][0])
        op=self.op|{'action':'update','target_id':self.id,'title':'Only birds'}
        newer={'data':{'columns':['month','observations'],'rows':[['2025-01-01',3]]},'query_ms':8}
        state,changes=apply_operations(self.state,[op],[newer],'Only birds')
        c=state['cards'][0]
        self.assertEqual(c['id'],self.id)
        self.assertEqual([c[k] for k in ['left','top','width','height']],[old[k] for k in ['left','top','width','height']])
        self.assertEqual(c['history'][-1]['data'],old['data'])
        self.assertEqual(state['undo'][-1]['cards'],self.state['cards'])
        self.assertEqual(self.state['cards'][0],old)

    def test_add_keeps_selected_and_links_comparison(self):
        op=self.op|{'source_id':self.id,'title':'Comparison'}
        state,_=apply_operations(self.state,[op],[self.result],'Compare')
        self.assertEqual(state['cards'][0],self.state['cards'][0])
        self.assertEqual(state['cards'][1]['derived_from'],self.id)
        self.assertNotEqual(state['cards'][1]['id'],self.id)

    def test_visual_change_reuses_query_result(self):
        op=self.op|{'action':'update','target_id':self.id,'source_id':self.id,'sql':None,'chart':'bar'}
        state,changes=apply_operations(self.state,[op],[None],'Make bars')
        self.assertEqual(state['cards'][0]['data'],self.result['data'])
        self.assertTrue(changes[0]['reused_data'])
        self.assertEqual(state['cards'][0]['query_ms'],0)

    def test_invalid_second_operation_changes_nothing(self):
        old=copy.deepcopy(self.state)
        bad=self.op|{'x':'missing'}
        with self.assertRaises(ValueError):apply_operations(self.state,[self.op,bad],[self.result,self.result],'Two changes')
        self.assertEqual(self.state,old)

    def test_wrong_target_and_duplicate_update_rejected(self):
        op=self.op|{'action':'update','target_id':'unknown'}
        with self.assertRaises(ValueError):validate_plan({'operations':[op]},self.state,self.id)
        op['target_id']=self.id
        with self.assertRaises(ValueError):validate_plan({'operations':[op,op.copy()]},self.state,self.id)

    def test_photo_view_cannot_relabel_an_aggregate(self):
        op=self.op|{'action':'update','target_id':self.id,'source_id':self.id,'sql':None,'chart':'gallery'}
        with self.assertRaises(ValueError):apply_operations(self.state,[op],[None],'Photos')

    def test_map_requires_real_coordinate_columns(self):
        op=self.op|{'chart':'map','x':'longitude','y':'latitude'}
        with self.assertRaises(ValueError):apply_operations(self.state,[op],[self.result],'Map')

    def test_empty_plan_requires_a_question(self):
        self.assertEqual(validate_plan({'operations':[],'question':'Which year?'},self.state,None),[])
        with self.assertRaises(ValueError):validate_plan({'operations':[],'question':None},self.state,None)

if __name__=='__main__':unittest.main()
