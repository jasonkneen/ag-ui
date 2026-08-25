"""make_json_safe: cycle detection must be PATH-scoped, not global.

A langgraph 1.2.x interrupt payload legitimately shares references (the
recommended flight IS one of the options list's dicts). The global seen-set
serialized the second appearance as the string "<recursive>", which reached
the dojo's interrupt renderer and crashed the page client-side
("Cannot use 'in' operator to search for 'airline' in <recursive>").
"""
import unittest

from ag_ui_langgraph.utils import make_json_safe


class TestMakeJsonSafe(unittest.TestCase):
    def test_shared_references_are_not_cycles(self):
        shared = {"airline": "KLM", "price": 650}
        payload = {
            "options": [shared, {"airline": "United", "price": 720}],
            "recommended": shared,
        }
        out = make_json_safe(payload)
        self.assertEqual(out["recommended"], {"airline": "KLM", "price": 650})
        self.assertEqual(out["options"][0], {"airline": "KLM", "price": 650})

    def test_a_true_cycle_is_still_caught(self):
        cyc = {}
        cyc["self"] = cyc
        self.assertEqual(make_json_safe(cyc), {"self": "<recursive>"})

    def test_sibling_lists_sharing_an_element(self):
        item = {"id": 1}
        out = make_json_safe({"a": [item], "b": [item]})
        self.assertEqual(out, {"a": [{"id": 1}], "b": [{"id": 1}]})


if __name__ == "__main__":
    unittest.main()
