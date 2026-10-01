"""Conservative JSON leaf detection and exact contextual proposals."""
import json
from dataclasses import dataclass


@dataclass
class Leaf:
    prefix: str
    start: int
    path: tuple


class OpenString(Exception):
    def __init__(self, leaf):
        self.leaf = leaf


def current_leaf(text):
    """Parse the whole partial JSON; only call-list arguments are eligible."""
    original = text
    wrapper = 0
    # A recognized opening JSON fence can surround a JSON call list. It is
    # never copied, and strict output scoring still rejects fenced responses.
    for opening in ['```json\n', '```\n']:
        if text.lstrip().startswith(opening):
            wrapper = len(text) - len(text.lstrip()) + len(opening)
            text = text[wrapper:]
            break
    i = 0

    def ws():
        nonlocal i
        while i < len(text) and text[i].isspace():
            i += 1

    def string(path, key=False):
        nonlocal i
        begin = i
        i += 1
        escaped = False
        while i < len(text):
            c = text[i]
            i += 1
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                return json.loads(text[begin:i])
        if not key:
            raise OpenString(Leaf(text[begin + 1:], begin + 1, path))
        raise ValueError('unfinished key')

    def value(path):
        nonlocal i
        ws()
        if i == len(text):
            raise ValueError('unfinished value')
        c = text[i]
        if c == '"':
            return string(path)
        if c == '{':
            i += 1
            ws()
            seen = set()
            if i < len(text) and text[i] == '}':
                i += 1
                return
            while True:
                ws()
                if i >= len(text) or text[i] != '"':
                    raise ValueError('key expected')
                key = string(path, True)
                if key in seen:
                    raise ValueError('duplicate key')
                seen.add(key)
                ws()
                if i >= len(text) or text[i] != ':':
                    raise ValueError('colon expected')
                i += 1
                value(path + (key,))
                ws()
                if i >= len(text):
                    raise ValueError('unfinished object')
                sep = text[i]
                i += 1
                if sep == '}':
                    return
                if sep != ',':
                    raise ValueError('comma expected')
        if c == '[':
            i += 1
            ws()
            if i < len(text) and text[i] == ']':
                i += 1
                return
            index = 0
            while True:
                value(path + (index,))
                index += 1
                ws()
                if i >= len(text):
                    raise ValueError('unfinished array')
                sep = text[i]
                i += 1
                if sep == ']':
                    return
                if sep != ',':
                    raise ValueError('comma expected')
        _, n = json.JSONDecoder().raw_decode(text[i:])
        i += n

    try:
        if not text.lstrip().startswith('['):
            return None
        value(())
    except OpenString as exc:
        leaf = exc.leaf
        p = leaf.path
        if len(p) >= 3 and isinstance(p[0], int) and p[1] == 'arguments':
            leaf.start += wrapper
            assert original[leaf.start:] == leaf.prefix
            return leaf
    except (ValueError, IndexError, RecursionError):
        pass
    return None


def safe(c):
    # Deliberate ASCII fallback for Unicode, escapes and all uncertain syntax.
    return 32 < ord(c) < 127 and not c.isspace() and c not in '"\\'


def source_candidates(query, leaf, tokenizer, output_ids, offsets, query_offset, budget):
    """No source-token grafting. Retokenize in output context and roundtrip."""
    prefix = leaf.prefix
    if len(prefix) < 3 or not all(map(safe, prefix)) or budget < 1:
        return []
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    found = []
    pos = query.find(prefix)
    while pos >= 0:
        end = pos + len(prefix)
        tail = ''
        for c in query[end:end + 64]:
            if not safe(c):
                break
            tail += c
        if tail:
            ids = tokenizer.encode(text + tail, add_special_tokens=False)
            if ids[:len(output_ids)] == output_ids:
                draft = ids[len(output_ids):][:min(8, budget)]
                extended = tokenizer.decode(output_ids + draft, skip_special_tokens=False)
                suffix = extended[len(text):] if extended.startswith(text) else ''
                if draft and suffix and tail.startswith(suffix) and all(map(safe, suffix)):
                    # A source token may include leading space; retain char/byte
                    # coordinates independently instead of equating source IDs.
                    stop = end + len(suffix)
                    token_span = [j + query_offset for j, (a, b) in enumerate(offsets)
                                  if a < stop and b > pos]
                    if token_span and offsets[token_span[0] - query_offset][0] <= pos and offsets[token_span[-1] - query_offset][1] >= stop:
                        found.append({'draft': draft, 'suffix': suffix,
                                      'char_start': pos, 'char_prefix_end': end, 'char_end': stop,
                                      'byte_start': len(query[:pos].encode('utf-8')),
                                      'byte_end': len(query[:stop].encode('utf-8')),
                                      'token_span': token_span, 'leaf_start': leaf.start,
                                      'path': list(leaf.path), 'prefix': prefix})
        pos = query.find(prefix, pos + 1)
    return found


def choose(candidates, row=None, threshold=0.0, margin=0.0):
    if not candidates:
        return None
    if row is None:
        if len({tuple(c['draft']) for c in candidates}) != 1:
            return None
        return candidates[0]
    if hasattr(row, 'device'):
        import torch
        # All candidate reductions and sorting stay on GPU; one small transfer
        # carries the best index/mass/runner-up margin to the controller.
        scores = torch.stack([row[c['token_span']].sum() for c in candidates])
        order = torch.argsort(scores, descending=True, stable=True)
        best = scores[order[0]]
        gap = best - scores[order[1]] if len(candidates) > 1 else best.new_tensor(float('inf'))
        j, mass, difference = torch.stack([order[0].to(best.dtype), best, gap]).tolist()
        j = int(j)
        if mass < threshold or difference < margin:
            return None
        return dict(candidates[j], attention_mass=mass, attention_margin=difference if len(candidates) > 1 else None)
    scores = [float(row[c['token_span']].sum()) for c in candidates]
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    j = order[0]
    if scores[j] < threshold:
        return None
    if len(order) > 1 and scores[j] - scores[order[1]] < margin:
        return None
    return dict(candidates[j], attention_mass=scores[j], attention_margin=(scores[j] - scores[order[1]] if len(order) > 1 else None))


def verified_prefix(draft, next_id, block_next_ids, eos_ids=()):
    """Return contiguous accepted prefix and first target mismatch, if any."""
    expected = [next_id] + list(block_next_ids[:-1])
    for j, (candidate, target) in enumerate(zip(draft, expected)):
        if candidate != target:
            return j, target
        if candidate in eos_ids:
            return j + 1, None
    return len(draft), None
