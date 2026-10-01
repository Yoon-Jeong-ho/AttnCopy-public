"""Independent scorer matching RefCall strict sorted-call/leaf definitions."""
import json
from collections import Counter


def canonical(calls):
    return sorted(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)


def leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, path + (i,))
    else:
        yield path, obj


def score(record, result):
    try:
        def pairs(p):
            if len({k for k, _ in p}) != len(p):
                raise ValueError('duplicate key')
            return dict(p)
        pred = json.loads(result['raw'], object_pairs_hook=pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        json.dumps(pred, allow_nan=False)
        valid = True
    except (ValueError, TypeError):
        pred, valid = None, False
    structure = isinstance(pred, list) and bool(pred) and all(isinstance(c, dict) and
                isinstance(c.get('name'), str) and isinstance(c.get('arguments'), dict) for c in pred)
    gold = record['answers']
    n_args, correct = 0, 0
    used, prediction_to_gold = set(), {}
    for gi, g in enumerate(gold):
        candidates = [(j, p) for j, p in enumerate(pred if structure else []) if j not in used and p['name'] == g['name']]
        # Historical same-name best top-level agreement; first tie retained.
        matched = max(candidates, key=lambda jp: sum(k in jp[1]['arguments'] and jp[1]['arguments'][k] == v
                      for k, v in g['arguments'].items()), default=None)
        if matched:
            used.add(matched[0])
            prediction_to_gold[matched[0]] = gi
        pl = dict(leaves(matched[1]['arguments'])) if matched else {}
        for path, value in leaves(g['arguments']):
            n_args += 1
            correct += path in pl and type(pl[path]) is type(value) and pl[path] == value
    event_pred = pred if structure else None
    event_parse = 'strict_json' if structure else None
    # Fenced payload is only a selection diagnostic: it never changes strict
    # JSON validity, full-call EM, or the argument denominator above.
    text = result['raw'].strip()
    for opening in ['```json\n','```\n']:
        if event_pred is None and text.startswith(opening) and text.endswith('\n```'):
            try:
                payload = json.loads(text[len(opening):-4], object_pairs_hook=pairs)
                if isinstance(payload,list) and payload and all(isinstance(c,dict) and isinstance(c.get('name'),str)
                   and isinstance(c.get('arguments'),dict) for c in payload):
                    event_pred, event_parse = payload, 'recognized_fenced_json_diagnostic'
            except (ValueError,TypeError):
                pass
    for event in result['events']:
        path = tuple(event['path'])
        event['gold_support_ambiguous'] = True
        event['gold_value_prefix_supported'] = None
        try:
            if event_pred is None or path[0] >= len(event_pred):
                continue
            name = event_pred[path[0]]['name']
            # An event cannot be assigned unambiguously to duplicate-name calls
            # using the final (possibly forced) arguments as its identity.
            if sum(c['name'] == name for c in gold) != 1 or sum(c['name'] == name for c in event_pred) != 1:
                continue
            obj = next(g for g in gold if g['name'] == name)
            for k in path[1:]:
                obj = obj[k]
            event['gold_support_ambiguous'] = False
            event['gold_support_parse'] = event_parse
            event['gold_value_prefix_supported'] = isinstance(obj, str) and obj.startswith(event['prefix'] + event['suffix'])
        except (KeyError, IndexError, TypeError):
            pass
    return {'valid_json': valid, 'structure_ok': structure,
            'full_call_em': structure and canonical(pred) == canonical(gold),
            'tool_name_em': structure and Counter(c['name'] for c in pred) == Counter(c['name'] for c in gold),
            'n_args': n_args, 'correct_args': correct, 'truncated': result['stop'] != 'eos'}
