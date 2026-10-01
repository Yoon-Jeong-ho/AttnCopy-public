"""CPU-only strict parsed-value and final native JSON review; no generation changes."""
import json
import math
import re
from types import SimpleNamespace
from .bfcl_native import handler

REVISION = "bfcl-postprocess-r1-recursive-types"
PATTERN = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)

def exact(value, gold):
    """Recursive Python/JSON value types, exact strings, order-free object keys."""
    if type(value) is not type(gold):
        return False
    if isinstance(value, dict):
        return value.keys() == gold.keys() and all(exact(value[k], gold[k]) for k in value)
    if isinstance(value, (list, tuple)):
        return len(value) == len(gold) and all(exact(a, b) for a, b in zip(value, gold))
    if isinstance(value, float) and not (math.isfinite(value) and math.isfinite(gold)):
        return False
    return value == gold

def strict_arguments(entry, gold, parsed):
    name, expected = next(iter(gold["ground_truth"][0].items()))
    schema = entry["function"][0]["parameters"]
    required = set(schema["required"])
    valid = len(parsed) == 1 and type(parsed[0]) is dict and name in parsed[0] and type(parsed[0][name]) is dict
    pred = parsed[0][name] if valid else {}
    flags = {}
    for key, alternatives in expected.items():
        if not alternatives:
            flags[key] = None
        elif not valid:
            flags[key] = False
        elif key in pred:
            flags[key] = any(exact(pred[key], g) for g in alternatives)
        else:
            flags[key] = key not in required and "" in alternatives
    string_keys = [k for k, alts in expected.items() if
        schema["properties"].get(k, {}).get("type") == "string" or
        any(type(v) is str and v != "" for v in alts)]
    unknown = [k for k, v in flags.items() if v is None]
    call = valid and required <= pred.keys() and pred.keys() <= expected.keys() and all(v is True for v in flags.values())
    return {"strict_call_correct_lower_bound": bool(call), "strict_call_evaluable": not unknown,
            "argument_n": len(flags), "unknown_argument_n": len(unknown),
            "correct_argument_n": sum(v is True for v in flags.values()),
            "string_argument_n": len(string_keys),
            "unknown_string_argument_n": sum(flags[k] is None for k in string_keys),
            "correct_string_argument_n": sum(flags[k] is True for k in string_keys),
            "per_argument": flags, "unknown_arguments": unknown}

def native_json(raw):
    """Final handler-cleaned envelopes, not JSON validity of the XML/thinking output."""
    response = SimpleNamespace(choices=[SimpleNamespace(text=raw)],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    cleaned = handler()._parse_query_response_prompting(response)["model_responses"]
    matches = list(PATTERN.finditer(cleaned))
    residual = PATTERN.sub("", cleaned)
    framing = bool(matches) and "<tool_call>" not in residual and "</tool_call>" not in residual
    syntax = []; envelopes = []; duplicates = 0
    for match in matches:
        repeated = []
        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result: repeated.append(key)
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError("non-JSON constant: " + value)
        try:
            value = json.loads(match.group(1), object_pairs_hook=object_pairs, parse_constant=invalid_constant)
            syntax.append(True)
            envelopes.append(type(value) is dict and type(value.get("name")) is str and type(value.get("arguments")) is dict and not repeated)
        except (ValueError, TypeError):
            syntax.append(False); envelopes.append(False)
        duplicates += len(repeated)
    return {"final_payload_n": len(matches), "native_payload_json_valid": framing and all(syntax),
            "native_fc_envelope_valid": framing and all(envelopes),
            "duplicate_json_key_n": duplicates,
            "status": "valid_native_fc" if framing and all(envelopes) else
                "no_final_envelope" if not matches else "invalid_or_incomplete_native_envelope"}
