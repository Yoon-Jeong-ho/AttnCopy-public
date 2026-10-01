"""Copy only in native tool-call argument strings, outside all reasoning."""
import json
import re
from .safety import current_leaf,safe


def native_leaf(text):
    cursor=0
    call_index=0
    stripped=text.lstrip()
    if stripped.startswith('<think>'):
        end=text.find('</think>')
        if end<0:
            return None
        cursor=end+len('</think>')
    while True:
        while cursor<len(text) and text[cursor].isspace():
            cursor+=1
        opening='<tool_call>\n'
        if not text.startswith(opening,cursor):
            return None
        body_start=cursor+len(opening)
        closing=text.find('\n</tool_call>',body_start)
        if closing>=0:
            try:
                def pairs(items):
                    if len({k for k,_ in items})!=len(items):
                        raise ValueError('duplicate key')
                    return dict(items)
                obj=json.loads(text[body_start:closing],object_pairs_hook=pairs)
                if not isinstance(obj,dict) or not isinstance(obj.get('name'),str) or not isinstance(obj.get('arguments'),dict):
                    return None
            except (ValueError,TypeError):
                return None
            call_index+=1
            cursor=closing+len('\n</tool_call>')
            continue
        body=text[body_start:]
        if '<tool_call>' in body or '</tool_call>' in body or '<think>' in body or '</think>' in body:
            return None
        leaf=current_leaf('['+body)
        if leaf is None or not isinstance(leaf.path[2],str):
            return None
        leaf.start+=body_start-1
        leaf.path=(call_index,)+leaf.path[1:]
        name=re.match(r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,',body)
        leaf.call_name=json.loads(name[1]) if name else None
        if '<' in leaf.prefix or '>' in leaf.prefix:
            return None
        assert text[leaf.start:]==leaf.prefix
        return leaf


def native_candidates(query,leaf,tokenizer,output_ids,offsets,qoffset,budget):
    # Preserve actual old IDs even if earlier generated text has a noncanonical
    # token combination. Only the NEW continuation boundary must be canonical;
    # no token may cross it, and the entire decode must roundtrip exactly.
    prefix=leaf.prefix
    if len(prefix)<3 or not all(map(safe,prefix)) or '<' in prefix or '>' in prefix or budget<1:return []
    text=tokenizer.decode(output_ids,skip_special_tokens=False)
    candidates=[]
    pos=query.find(prefix)
    while pos>=0:
        end=pos+len(prefix);tail=''
        for c in query[end:end+64]:
            if not safe(c) or c in '<>':break
            tail+=c
        if tail:
            encoded=tokenizer(text+tail,add_special_tokens=False,return_offsets_mapping=True)
            boundary=len(text)
            spans=encoded['offset_mapping']
            if not any(a<boundary<b for a,b in spans):
                draft=[t for t,(a,b) in zip(encoded['input_ids'],spans) if a>=boundary and b>boundary][:min(8,budget)]
                extended=tokenizer.decode(output_ids+draft,skip_special_tokens=False)
                suffix=extended[len(text):] if extended.startswith(text) else ''
                if draft and suffix and tail.startswith(suffix) and all(map(safe,suffix)):
                    stop=end+len(suffix)
                    token_span=[j+qoffset for j,(a,b) in enumerate(offsets) if a<stop and b>pos]
                    if token_span and offsets[token_span[0]-qoffset][0]<=pos and offsets[token_span[-1]-qoffset][1]>=stop:
                        candidates.append({'draft':draft,'suffix':suffix,'char_start':pos,'char_prefix_end':end,'char_end':stop,
                           'byte_start':len(query[:pos].encode()),'byte_end':len(query[:stop].encode()),'token_span':token_span,
                           'leaf_start':leaf.start,'path':list(leaf.path),'prefix':prefix,'call_name':getattr(leaf,'call_name',None),
                           'tokenization':'canonical new boundary + exact old-ID decode roundtrip'})
        pos=query.find(prefix,pos+1)
    special=set(getattr(tokenizer,'all_special_ids',[]))
    return [c for c in candidates if '<' not in c['suffix'] and '>' not in c['suffix'] and not special.intersection(c['draft'])]
