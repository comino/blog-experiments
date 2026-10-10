"""A stale or failed tool result must never be displayed as a fresh answer."""
import copy
import json
from pathlib import Path
import unittest

from fast_mode import ready_resources


class ResourceRoutingTest(unittest.TestCase):
    def setUp(self):
        self.messages = json.loads((Path(__file__).parent / 'fixtures/successful-round.json').read_text())

    def test_live_librechat_tool_round(self):
        resources = ready_resources(self.messages)
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]['row_count'], 36)
        self.assertEqual(resources[0]['marker'], '\\ui{e0dec59060}')

    def test_later_user_question_needs_inference(self):
        self.messages.append({'role': 'user', 'content': 'Now show only 2025.'})
        self.assertIsNone(ready_resources(self.messages))

    def test_completed_assistant_reply_is_not_replayed(self):
        self.messages.append({'role': 'assistant', 'content': '\\ui{e0dec59060}'})
        self.assertIsNone(ready_resources(self.messages))

    def test_failed_query_needs_repair(self):
        self.messages[1]['content'] = 'Query execution failed: unknown database.'
        self.assertIsNone(ready_resources(self.messages))

    def test_error_cannot_hide_before_valid_marker(self):
        self.messages[1]['content'] = self.messages[1]['content'].replace('"row_count": 36', '"error": "failure", "row_count": 36')
        self.assertIsNone(ready_resources(self.messages))

    def test_schema_discovery_still_needs_inference(self):
        self.messages[0]['tool_calls'][0]['function']['name'] = 'list_tables_mcp_clickhouse'
        self.assertIsNone(ready_resources(self.messages))

    def test_incomplete_parallel_round_waits(self):
        extra = copy.deepcopy(self.messages[0]['tool_calls'][0])
        extra['id'] = 'pending-second-call'
        self.messages[0]['tool_calls'].append(extra)
        self.assertIsNone(ready_resources(self.messages))

    def test_completed_parallel_round_keeps_both_resources(self):
        extra = copy.deepcopy(self.messages[0]['tool_calls'][0]); extra['id'] = 'second-call'
        self.messages[0]['tool_calls'].append(extra)
        result = copy.deepcopy(self.messages[1]); result['tool_call_id'] = extra['id']
        result['content'] = result['content'].replace('e0dec59060', '1234567890')
        self.messages.insert(2, result)
        self.assertEqual([r['marker'] for r in ready_resources(self.messages)], ['\\ui{e0dec59060}', '\\ui{1234567890}'])

    def test_unmatched_call_result_is_rejected(self):
        self.messages[1]['tool_call_id'] = 'unrelated-call'
        self.assertIsNone(ready_resources(self.messages))

    def test_marker_must_match_resource_header(self):
        self.messages[1]['content'] = self.messages[1]['content'].replace('UI Resource ID: e0dec59060', 'UI Resource ID: 1234567890')
        self.assertIsNone(ready_resources(self.messages))

    def test_plain_user_text_resembling_notice_is_still_a_new_turn(self):
        self.messages[-1]['content'] = self.messages[-1]['content'][0]['text']
        self.assertIsNone(ready_resources(self.messages))

    def test_incomplete_history_does_not_crash_or_render(self):
        for messages in [None, {}, [None], [], [{'role': 'assistant', 'tool_calls': ['bad']}]]:
            with self.subTest(messages=messages):
                self.assertIsNone(ready_resources(messages))


if __name__ == '__main__':
    unittest.main()
