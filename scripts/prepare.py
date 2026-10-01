"""CPU-only frozen data/lineage; never open RefCall final/test files."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/pilot-20261001'
SOURCE = Path(os.environ.get('ATTNCOPY_SOURCE_DIR', str(ROOT / 'data/source/RefCall'))).expanduser().resolve()
MODEL = Path(os.environ.get('ATTNCOPY_MODEL_DIR', str(ROOT / 'data/models/Qwen3-0.6B-c1899de'))).expanduser().resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def template(r):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in ['description', 'default', 'example', 'examples']}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    t = sorted((dict(name=t['name'], parameters=clean(t['parameters'])) for t in r['tools']), key=lambda t: t['name'])
    return hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()


def main():
    if (RUN / 'frozen.json').exists():
        raise RuntimeError('frozen data exists; do not resample')
    paths = [SOURCE / 'data/bi9/raw-v1/xlam/dev.jsonl', SOURCE / 'data/bi9/plain-external-20260929/hermes/dev.jsonl']
    expected = ['674865a4e0ae124b9781dfcf40c3f673629b6e76dfaf7b1cc909878ec1e9117a',
                'c7157e0205a46ab1e48a2330882c391e996b4b12e032e091dfc5e2aff1fe3684']
    pools = []
    for p, h in zip(paths, expected):
        assert digest(p) == h
        rows = [json.loads(line) for line in p.open()]
        random.Random(42).shuffle(rows)
        pools.append(rows)
    hermes_queries = {r['query'] for r in pools[1]}
    dev = [r for r in pools[0] if r['query'] not in hermes_queries][:40]
    blocked = {template(r) for r in dev}
    test = [r for r in pools[1] if template(r) not in blocked][:160]
    assert len(dev) == 40 and len(test) == 160
    assert {r['source'] for r in dev}.isdisjoint({r['source'] for r in test})
    assert {template(r) for r in dev}.isdisjoint({template(r) for r in test})
    assert {r['query'] for r in dev}.isdisjoint({r['query'] for r in test})
    for name, rows in [('dev', dev), ('test', test)]:
        for r in rows:
            r['attncopy_template_group'] = template(r)
            r['attncopy_source_group'] = hashlib.sha256(r['query'].encode()).hexdigest()
        (RUN / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    cfg = {'model_path': str(MODEL), 'model_revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
           'seed': 42, 'dtype': 'float32', 'backend': 'sdpa', 'layer': 27, 'head': 0,
           'max_new_tokens': 512, 'eos_ids': [151643, 151645],
           'max_copy_tokens': 8, 'max_copy_chars': 64, 'min_prefix_chars': 3,
           'selection': 'single prespecified final layer/head; dev 25th percentile mass and 0.01 margin; no sweep',
           'dev_n': 40, 'test_n': 160, 'attention_margin': 0.01}
    dump(ROOT / 'configs/pilot.json', cfg)
    relevant = ['AGENTS.md', 'docs/DATA.md', 'refcall/metrics.py', 'refcall/prompt.py', 'refcall/data/splits.py',
                'refcall/data/hermes.py', 'scripts/prepare_bi9_plain_external.py',
                'data/bi9/sources-v2/source_manifest.json', 'data/bi9/plain-external-20260929/manifest.json',
                'docs/BI9_PLAIN_V5_PROTOCOL.md', 'docs/BI9_POINTER_PILOT_PROTOCOL.md']
    lineage = {str(SOURCE / p): digest(SOURCE / p) for p in relevant}
    lineage.update({str(p): digest(p) for p in paths})
    lineage.update({str(p): digest(p) for p in MODEL.iterdir() if p.is_file()})
    frozen = {'run_id': 'pilot-20261001', 'seed': 42, 'config_sha256': digest(ROOT / 'configs/pilot.json'),
              'files': {str(RUN / f'{s}.jsonl'): digest(RUN / f'{s}.jsonl') for s in ['dev', 'test']},
              'source_model_sha256': lineage,
              'ids': {'dev': [r['id'] for r in dev], 'test': [r['id'] for r in test]},
              'template_groups': {'dev': len({template(r) for r in dev}), 'test': len({template(r) for r in test})},
              'implementation_agent': {'provider': 'openai', 'model': 'gpt-6.1-sol', 'cli_version': '0.159.2',
                                       'session_id': os.environ.get('CODEX_SESSION_ID')},
              'heldout_scope': 'existing project dev pools only; zero AttnCopy training; unknown Qwen pretraining contamination',
              'grouping_limits': 'dataset sources and normalized tool schema templates disjoint; latent natural-language template equivalence unknown',
              'licenses': {'xlam': 'CC-BY-4.0; existing authorized cache, no authentication/download', 'hermes': 'Apache-2.0', 'Qwen3': 'Apache-2.0',
                           'RefCall_code': 'no LICENSE found; inspected definitions, no source code copied/imported'},
              'source_gates': 'RefCall final/test/BFCL sealed files never opened; ToolACE training material excluded'}
    dump(RUN / 'frozen.json', frozen)
    print(json.dumps({'frozen': str(RUN), 'dev': len(dev), 'test': len(test), 'groups': frozen['template_groups']}))


if __name__ == '__main__':
    main()
