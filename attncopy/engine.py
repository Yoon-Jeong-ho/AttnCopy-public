"""SDPA decoding with fresh causal blocks and optional selected-row hooks."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from .paths import model_path
from .safety import current_leaf, source_candidates, choose, verified_prefix, safe

ARMS = ['greedy', 'attention-only', 'match-forced', 'attention+match-forced', 'verifiedPLD']


class AttentionRow:
    """Reuse projected/normalized Q and prefix K; compute one GPU row."""
    def __init__(self, model, layer, head):
        self.model, self.layer, self.head = model, layer, head
        self.enabled = False
        self.row = None
        self.events = []
        self.module = model.model.layers[layer].self_attn
        self.handles = []

    def set_enabled(self, enabled):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.enabled = enabled
        if enabled:
            self.handles = [self.module.q_norm.register_forward_hook(self.q_hook),
                            self.module.register_forward_hook(self.row_hook, with_kwargs=True)]

    def q_hook(self, module, args, output):
        if self.enabled:
            self.q = output[:, -1:, :, :].transpose(1, 2)

    def row_hook(self, module, args, kwargs, output):
        if not self.enabled:
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        cos, sin = kwargs['position_embeddings']
        q, _ = apply_rotary_pos_emb(self.q, self.q, cos[:, -1:], sin[:, -1:])
        cache = kwargs['past_key_values']
        keys = cache.layers[self.layer].keys
        group = module.config.num_attention_heads // module.config.num_key_value_heads
        k = keys[:, self.head // group, :, :]
        self.row = torch.softmax((q[:, self.head, :, :] @ k.transpose(-1, -2)).float() * module.scaling, dim=-1)[0, 0]
        end.record()
        self.events.append((start, end))


class Engine:
    def __init__(self, config):
        self.config = config
        self.tok = AutoTokenizer.from_pretrained(model_path(config), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path(config), local_files_only=True,
                     dtype=getattr(torch, config['dtype']), attn_implementation='sdpa').to('cuda').eval()
        self.attention = AttentionRow(self.model, config['layer'], config['head'])
        self.eos = set(config['eos_ids'])
        self.traces = []

    def prompt(self, record):
        # Identical original plain tool-call contract, no reference markers.
        system = ('You are a function-calling assistant. You are given a list of tools in JSON format. '
                  "Answer the user's request by calling one or more tools. Respond ONLY with a JSON list of "
                  'tool calls of the form [{"name": <tool name>, "arguments": {<argument name>: <value>, ...}}].\n'
                  'Tools:\n' + __import__('json').dumps(record['tools'], ensure_ascii=False))
        marker = '␀QUERY␀'
        assert marker not in record['query'] and marker not in system
        rendered = self.tok.apply_chat_template([{'role': 'system', 'content': system},
                   {'role': 'user', 'content': marker}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pre, post = rendered.split(marker)
        text = pre + record['query'] + post
        full = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        begin, end = len(pre), len(pre) + len(record['query'])
        overlap = [(j, (a - begin, b - begin)) for j, (a, b) in enumerate(full['offset_mapping']) if a < end and b > begin]
        assert overlap and [j for j, _ in overlap] == list(range(overlap[0][0], overlap[-1][0] + 1))
        return full['input_ids'], [ab for _, ab in overlap], overlap[0][0]

    @torch.inference_mode()
    def forward(self, ids, cache, all_logits=False):
        n = cache.get_seq_length()
        x = torch.tensor([ids], device='cuda')
        positions = torch.arange(n, n + len(ids), device='cuda')[None, :]
        out = self.model(input_ids=x, position_ids=positions, past_key_values=cache,
                         use_cache=True, logits_to_keep=0 if all_logits else 1)
        assert cache.get_seq_length() == n + len(ids)
        if len(ids) > 1 and n:
            self.traces.append({'prefix_cache_length': n, 'new_positions': [n, n + len(ids) - 1],
                                'block_length': len(ids), 'cache_after': cache.get_seq_length()})
        return out.logits[0], cache

    @torch.inference_mode()
    def generate(self, record, arm, threshold=0., margin=0., max_new=None):
        max_new = max_new or self.config['max_new_tokens']
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
                leaf = current_leaf(raw)
                if leaf and len(leaf.prefix) >= 3 and all(map(safe,leaf.prefix)) and leaf.start not in used:
                    eligible += 1
                    candidates = source_candidates(record['query'], leaf, self.tok, generated, offsets, qoffset, max_new - len(generated))
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
                    generated.extend(draft[:accepted])
                    cache.crop(old_length + accepted)
                    assert cache.get_seq_length() == old_length + accepted
                    event.update(accepted=accepted, rejected=len(draft) - accepted, cache_after_trim=cache.get_seq_length())
                    if generated and generated[-1] in self.eos:
                        events.append(event)
                        break
                    if mismatch is not None and len(generated) < max_new:
                        generated.append(mismatch)
                        if mismatch in self.eos:
                            events.append(event)
                            break
                        logits, cache = self.forward([mismatch], cache)
                    else:
                        logits = block_logits[-1:]
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
        raw = self.tok.decode(generated, skip_special_tokens=True)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - decode_start
        total_s = time.perf_counter() - total_start
        attention_ms = sum(a.elapsed_time(b) for a, b in self.attention.events)
        return {'id': record['id'], 'source': record['source'], 'arm': arm,
                'token_ids': generated, 'stop': 'eos' if generated and generated[-1] in self.eos else 'max_new_tokens',
                'eos_id': generated[-1] if generated and generated[-1] in self.eos else None,
                'raw': raw, 'prompt_tokens': len(prompt), 'generated_tokens': len(generated),
                'prefill_s': prefill_s, 'decode_s': decode_s, 'total_s': total_s, 'prompt_tokenizer_s': prompt_s,
                'proposal_s': proposal_s, 'attention_gpu_s': attention_ms / 1000,
                'eligible_opportunities': eligible, 'candidate_opportunities': candidate_opportunities,
                'events': events, 'block_traces': self.traces,
                'backend': 'sdpa', 'dtype': self.config['dtype'], 'timing_boundary': 'prompt tokenization through final text decode and CUDA sync; excludes initialization, scorer, disk IO'}
