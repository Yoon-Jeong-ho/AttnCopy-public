"""Separate native-FC engine; original pilot source remains immutable."""
import time
import torch
from transformers import DynamicCache
from .engine import Engine as PilotEngine
from .safety import choose,verified_prefix,safe
from .bfcl_safety import native_leaf,native_candidates


class BFCLEngine(PilotEngine):
    def detect_leaf(self,text):
        return native_leaf(text)

    def replay_verified(self,draft,logits,cache,generated,max_new):
        """Rebuild actual target-prefix KV one token at a time in BF16.

        Parallel cache has already been discarded. This intentionally trades
        acceleration for exact greedy-cache parity; every cost is timed.
        """
        accepted=0
        forwarded=0
        correction=None
        for candidate in draft:
            target=int(logits[-1].argmax())
            matched=candidate==target
            token=candidate if matched else target
            if matched:accepted+=1
            else:correction=target
            generated.append(token)
            if token in self.eos or len(generated)>=max_new:
                break
            logits,cache=self.forward([token],cache)
            forwarded+=1
            if not matched:
                break
        return logits,cache,accepted,forwarded,correction

    def prompt(self,record):
        text=record['prompt_text']
        full=self.tok(text,add_special_tokens=False,return_offsets_mapping=True)
        assert full['input_ids']==record['input_ids'], 'CPU/native and GPU tokenizer input mismatch'
        self.source_alignment=[]
        for source in record['sources']:
            begin=source['prompt_char_start'];end=begin+len(source['text'])
            assert text[begin:end]==source['text']
            overlap=[(j,(a-begin,b-begin)) for j,(a,b) in enumerate(full['offset_mapping']) if a<end and b>begin]
            if not overlap:continue
            assert [j for j,_ in overlap]==list(range(overlap[0][0],overlap[-1][0]+1))
            self.source_alignment.append((source['text'],[ab for _,ab in overlap],overlap[0][0],source['message_index']))
        return full['input_ids'],None,None

    @torch.inference_mode()
    def generate(self, record, arm, threshold=0., margin=0., max_new=None):
        max_new = max_new or record['max_new_tokens']
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        prompt_start = total_start
        prompt, offsets, qoffset = self.prompt(record)
        prompt_s = time.perf_counter() - prompt_start
        cache = DynamicCache(config=self.model.config)
        self.attention.set_enabled(arm in ['attention-only', 'attention+match-forced'])
        self.attention.events = []
        self.attention.row = None
        self.traces = []
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        logits, cache = self.forward(prompt, cache)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - prefill_start
        generated, events, eligible, candidate_opportunities = [], [], 0, 0
        proposal_s = 0.
        used = set()
        decode_start = time.perf_counter()
        while len(generated) < max_new:
            next_id = int(logits[-1].argmax())
            selected = None
            if arm not in ['greedy', 'attention-only'] and generated:
                t = time.perf_counter()
                raw = self.tok.decode(generated, skip_special_tokens=False)
                leaf = self.detect_leaf(raw)
                if leaf and len(leaf.prefix) >= 3 and all(map(safe,leaf.prefix)) and leaf.start not in used:
                    eligible += 1
                    candidates = []
                    for text, mapping, source_offset, message_index in self.source_alignment:
                        found = native_candidates(text, leaf, self.tok, generated, mapping, source_offset, max_new - len(generated))
                        candidates.extend(dict(c, source_message_index=message_index) for c in found)
                    candidate_opportunities += bool(candidates)
                    row = self.attention.row if arm == 'attention+match-forced' else None
                    selected = choose(candidates, row, threshold, margin)
                    if selected:
                        used.add(leaf.start)
                proposal_s += time.perf_counter() - t
            if selected:
                draft = selected['draft']
                old_length = cache.get_seq_length()
                block_logits, cache = self.forward(draft, cache, all_logits=True)
                block_next = block_logits.argmax(-1).tolist()
                accepted, mismatch = verified_prefix(draft, next_id, block_next, self.eos)
                expected = [next_id] + block_next[:-1]
                event = dict(selected, output_start=len(generated), block_length=len(draft),
                             compatible_tokens=accepted, compatible_block=accepted == len(draft),
                             conditional_wrong_tokens=sum(a != b for a, b in zip(draft, expected)),
                             cache_before=old_length, cache_after_draft=cache.get_seq_length())
                if arm == 'verifiedPLD':
                    # BF16 multi-token projection changes newly generated KV
                    # slightly even when provisional argmax tokens match. Crop
                    # the whole provisional block and recreate serial target KV.
                    cache.crop(old_length)
                    assert cache.get_seq_length() == old_length
                    event.update(parallel_accepted=accepted,cache_after_trim=old_length)
                    logits,cache,accepted,forwarded,correction=self.replay_verified(draft,logits,cache,generated,max_new)
                    assert cache.get_seq_length()==old_length+forwarded
                    event.update(accepted=accepted,rejected=len(draft)-accepted,replayed_forward_tokens=forwarded,
                                 cache_after_replay=cache.get_seq_length(),target_correction=correction,
                                 compatible_tokens=accepted,compatible_block=accepted==len(draft),
                                 verification='parallel proposal check + exact serial BF16 cache replay')
                    if generated and generated[-1] in self.eos:
                        events.append(event)
                        break
                else:
                    generated.extend(draft)
                    logits = block_logits[-1:]
                    event.update(forced=len(draft), wrong_event=accepted < len(draft))
                events.append(event)
            else:
                generated.append(next_id)
                if next_id in self.eos:
                    break
                if len(generated) < max_new:
                    logits, cache = self.forward([next_id], cache)
        raw = self.tok.decode(generated[:-1] if generated and generated[-1] in self.eos else generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - decode_start
        total_s = time.perf_counter() - total_start
        attention_ms = sum(a.elapsed_time(b) for a, b in self.attention.events)
        return {'id': record['id'], 'source': record['category'], 'arm': arm, 'category': record['category'],
                'token_ids': generated, 'stop': 'eos' if generated and generated[-1] in self.eos else 'max_new_tokens',
                'eos_id': generated[-1] if generated and generated[-1] in self.eos else None,
                'raw': raw, 'prompt_tokens': len(prompt), 'generated_tokens': len(generated),
                'prefill_s': prefill_s, 'decode_s': decode_s, 'total_s': total_s, 'prompt_tokenizer_s': prompt_s,
                'proposal_s': proposal_s, 'attention_gpu_s': attention_ms / 1000,
                'eligible_opportunities': eligible, 'candidate_opportunities': candidate_opportunities,
                'events': events, 'block_traces': self.traces,
                'backend': 'sdpa', 'dtype': self.config['dtype'], 'timing_boundary': 'prompt tokenization through final text decode and CUDA sync; excludes initialization, scorer, disk IO'}
