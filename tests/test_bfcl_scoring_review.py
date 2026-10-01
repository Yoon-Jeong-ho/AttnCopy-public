import unittest
from attncopy.bfcl_scoring_review import exact, strict_arguments, native_json
from attncopy.bfcl_native import exact_value

class StrictReviewTests(unittest.TestCase):
    def test_nested_numeric_types_regression(self):
        self.assertTrue(exact_value({"x": [True]}, [{"x": [1]}]))
        self.assertFalse(exact({"x": [True]}, {"x": [1]}))
        self.assertFalse(exact({"x": [1.0]}, {"x": [1]}))
        self.assertTrue(exact({"b": [1, {"x": "é"}], "a": False}, {"a": False, "b": [1, {"x": "é"}]}))
        self.assertFalse(exact("ABC", "abc"))
        self.assertFalse(exact(float("nan"), float("nan")))
    def test_all_alternatives_optional_missing_and_undefined(self):
        entry={"function":[{"parameters":{"required":["x"],"properties":{"x":{"type":"object"},"optional":{"type":"string"},"unknown":{"type":"string"}}}}]}
        gold={"ground_truth":[{"f":{"x":[{"v":1},{"v":2}],"optional":["text",""],"unknown":[]}}]}
        r=strict_arguments(entry,gold,[{"f":{"x":{"v":2}}}])
        self.assertEqual(r["per_argument"],{"x":True,"optional":True,"unknown":None})
        self.assertEqual((r["argument_n"],r["unknown_argument_n"],r["correct_argument_n"]),(3,1,2))
        self.assertFalse(r["strict_call_correct_lower_bound"])
        self.assertFalse(r["strict_call_evaluable"])
        gold["ground_truth"][0]["f"].pop("unknown")
        self.assertTrue(strict_arguments(entry,gold,[{"f":{"x":{"v":2}}}])["strict_call_correct_lower_bound"])
        self.assertFalse(strict_arguments(entry,gold,[{"f":{"x":{"v":2},"extra":1}}])["strict_call_correct_lower_bound"])
    def test_final_json_protocol_and_reasoning(self):
        good='<tool_call>\n{"name":"f","arguments":{"text":"\\u00e9"}}\n</tool_call>'
        self.assertTrue(native_json(good)["native_fc_envelope_valid"])
        self.assertTrue(native_json("<think>\n<tool_call>\n{bad}\n</tool_call>\n</think>\n"+good)["native_payload_json_valid"])
        for raw in ["", good+"<tool_call>\n{bad}\n</tool_call>", good+"<tool_call>"]:
            self.assertFalse(native_json(raw)["native_payload_json_valid"])
        scalar='<tool_call>\n123\n</tool_call>'
        self.assertTrue(native_json(scalar)["native_payload_json_valid"])
        self.assertFalse(native_json(scalar)["native_fc_envelope_valid"])
        nan='<tool_call>\n{"name":"f","arguments":{"x":NaN}}\n</tool_call>'
        self.assertFalse(native_json(nan)["native_payload_json_valid"])
        duplicate='<tool_call>\n{"name":"f","arguments":{"x":1,"x":2}}\n</tool_call>'
        self.assertTrue(native_json(duplicate)["native_payload_json_valid"])
        self.assertFalse(native_json(duplicate)["native_fc_envelope_valid"])
        self.assertEqual(native_json(duplicate)["duplicate_json_key_n"],1)

if __name__ == "__main__":unittest.main()
