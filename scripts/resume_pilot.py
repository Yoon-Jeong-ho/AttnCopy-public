"""Resume only missing pairs into a fresh segment; historical files immutable."""
from contextlib import ExitStack
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attncopy.paths import require_authorized_host, gpu_lock_files
from scripts.prepare import ROOT,RUN,digest,dump
from scripts.pilot import append,read,gpu_snapshot,fixtures

SNAP = ROOT/'runs/pilot-recovery-snapshot-20260930T1620Z'
OUTPUT = ROOT/'runs/pilot-resume-20261001'
ARMS = ['greedy','attention-only','match-forced','attention+match-forced','verifiedPLD']


def validate_rows(rows,records):
    seen = set()
    expected_order = []
    for i,rec in enumerate(records):
        order = ARMS[i%5:]+ARMS[:i%5]
        expected_order.extend((rec['id'],a) for a in order)
    positions = {pair:i for i,pair in enumerate(expected_order)}
    previous = -1
    for row in rows:
        pair = row['id'],row['arm']
        assert pair not in seen and pair in positions
        seen.add(pair)
        i = row['example_index']
        assert records[i]['id'] == row['id']
        order = ARMS[i%5:]+ARMS[:i%5]
        assert row['order'] == order and row['split'] == 'test'
        assert row['backend']=='sdpa' and row['dtype']=='bfloat16'
        assert len(row['token_ids']) == row['generated_tokens']
        assert positions[pair] > previous
        previous = positions[pair]
    return seen


def recovery_plan():
    manifest = SNAP/'MANIFEST.json'
    assert digest(manifest)=='5cfaf6986f34d7599b0da2819b1f4b20c545a3be0f37f8703246bc7f383749be'
    original = json.loads(manifest.read_text())
    for rel,entry in original['artifacts'].items():
        assert digest(SNAP/rel)==entry['sha256'],f'snapshot changed: {rel}'
        assert digest(ROOT/rel)==entry['sha256'],f'original changed: {rel}'
    records = read(SNAP/'runs/pilot-20261001/test.jsonl')
    rows = read(SNAP/'runs/pilot-20261001/test-predictions.jsonl')
    assert len(rows)==272 and len(records)==160
    seen = validate_rows(rows,records)
    todo = [(i,arm) for i,rec in enumerate(records) for arm in ARMS[i%5:]+ARMS[:i%5] if (rec['id'],arm) not in seen]
    assert len(todo)==528 and todo[:3]==[(54,'attention-only'),(54,'match-forced'),(54,'attention+match-forced')]
    cfg = json.loads((ROOT/'configs/pilot.json').read_text())
    frozen = json.loads((SNAP/'runs/pilot-20261001/frozen-v2.json').read_text())
    selected = json.loads((SNAP/'runs/pilot-20261001/selection.json').read_text())
    assert cfg['seed']==42 and cfg['dtype']=='bfloat16' and cfg['backend']=='sdpa'
    assert cfg['layer']==selected['layer']==27 and cfg['head']==selected['head']==0
    assert cfg['max_new_tokens']==512 and cfg['max_copy_tokens']==8 and cfg['max_copy_chars']==64
    assert cfg['eos_ids']==[151643,151645] and selected['margin']==cfg['attention_margin']==.01
    assert selected['threshold']==1.3354860129766166e-05
    assert digest(ROOT/'configs/pilot.json')==frozen['config_sha256']==selected['config_sha256']
    assert [r['id'] for r in records]==frozen['ids']['test']
    assert len(read(SNAP/'runs/pilot-20261001/dev-predictions.jsonl'))==200
    return records,rows,todo,cfg,frozen,selected,original


def merge_rows(old,new,records):
    validate_rows(old,records)
    validate_rows(new,records)
    by_pair = {}
    for segment,rows in [('segment1',old),('segment2',new)]:
        for row in rows:
            pair = row['id'],row['arm']
            assert pair not in by_pair
            by_pair[pair] = dict(row,source_segment=segment,latency_boundary_excluded=row['example_index']==54)
    assert len(by_pair)==800
    return [by_pair[rec['id'],arm] for i,rec in enumerate(records) for arm in ARMS[i%5:]+ARMS[:i%5]]


def check_identity(rows,rec_id):
    by_arm = {r['arm']:r for r in rows if r['id']==rec_id}
    if len(by_arm)!=5:
        return
    g = by_arm['greedy']
    for arm in ['attention-only','verifiedPLD']:
        r = by_arm[arm]
        assert (r['token_ids'],r['stop'],r['eos_id']) == (g['token_ids'],g['stop'],g['eos_id']),f'identity mismatch {rec_id} {arm}'


def lease_gpu(stack,gpu,out):
    from scripts.prepare import SOURCE
    files = [RUN/'controller.lock',*gpu_lock_files(SOURCE, gpu)]
    assert len(files)>=2
    hashes = {str(p):digest(p) for p in files}
    for p in files:
        handle = stack.enter_context(p.open('r+'))
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = gpu_snapshot()
    row = [r.split(', ') for r in state['gpus'].splitlines() if r.startswith(f'{gpu},')][0]
    assert int(row[2])<=8 and int(row[3])==0 and row[1] not in state['processes'],f'GPU {gpu} not idle'
    expected_allocation = json.loads((SNAP/'runs/pilot-20261001/pilot-allocation.json').read_text())
    assert gpu == expected_allocation['gpu'] and row[1] == expected_allocation['uuid'], 'resume GPU identity changed without authorization'
    os.environ['CUDA_VISIBLE_DEVICES']=row[1]
    dump(out/'allocation.json',{'gpu':gpu,'uuid':row[1],'locks_sha256':hashes,'snapshot':state,'pid':os.getpid()})
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--gpu',type=int,default=0)
    args = parser.parse_args()
    require_authorized_host()
    records,old,todo,cfg,frozen,selected,manifest = recovery_plan()
    if args.dry_run:
        print(json.dumps({'old_rows':len(old),'new_rows_needed':len(todo),'first_missing':todo[:3],'snapshot_verified':True}))
        return
    OUTPUT.mkdir(exist_ok=False)
    dump(OUTPUT/'claim.json',{'pid':os.getpid(),'command':sys.argv,'time':time.time(),'snapshot_sha256':digest(SNAP/'MANIFEST.json')})
    try:
        with ExitStack() as stack:
            locks = lease_gpu(stack,args.gpu,OUTPUT)
            for path,h in frozen['source_model_sha256'].items():
                assert digest(path)==h,f'source/model identity changed: {path}'
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            import torch,transformers
            from attncopy.engine import Engine
            from attncopy.metrics import score
            original_runtime = json.loads((SNAP/'runs/pilot-20261001/pilot-lineage.json').read_text())
            assert torch.__version__ == original_runtime['torch'] and transformers.__version__ == original_runtime['transformers']
            qwen_module = sys.modules['transformers.models.qwen3.modeling_qwen3']
            assert digest(qwen_module.__file__) == original_runtime['qwen_runtime_sha256'], 'Qwen runtime changed'
            torch.set_num_threads(1);torch.manual_seed(42)
            dump(OUTPUT/'lineage.json',{'original_lineage':json.loads((SNAP/'runs/pilot-20261001/pilot-lineage.json').read_text()),
                 'resume_code_sha256':digest(__file__),'selection_sha256':digest(SNAP/'runs/pilot-20261001/selection.json'),
                 'config_sha256':digest(ROOT/'configs/pilot.json'),'source_model_asserted':True,
                 'torch':torch.__version__,'transformers':transformers.__version__,'command':sys.argv,'session':os.environ.get('CODEX_SESSION_ID')})
            engine = Engine(cfg)
            warmup = engine.generate(fixtures()[0],'greedy',max_new=32)
            append(OUTPUT/'warmup.jsonl',warmup)
            old_warmup = read(SNAP/'runs/pilot-20261001/pilot-warmup.jsonl')[0]
            assert (warmup['token_ids'],warmup['stop'])==(old_warmup['token_ids'],old_warmup['stop'])
            new = []
            for i,arm in todo:
                rec = records[i]
                result = engine.generate(rec,arm,selected['threshold'],selected['margin'])
                result.update(split='test',example_index=i,order=ARMS[i%5:]+ARMS[:i%5])
                result['scores']=score(rec,result)
                append(OUTPUT/'segment-2.jsonl',result)
                new.append(result)
                try:
                    check_identity(old+new,rec['id'])
                except AssertionError:
                    dump(OUTPUT/'identity-failure.json',{'id':rec['id'],'case_index':i,'arm':arm,'raw_preserved':True})
                    raise
                print(f'resume case {i+1}/160 {arm} ({len(new)}/{len(todo)} missing rows)',flush=True)
            merged = merge_rows(old,new,records)
            for rec in records:
                check_identity(merged,rec['id'])
            recovery_plan()  # All original/snapshot bytes must still match.
            assert all(digest(p)==h for p,h in locks.items())
            dump(OUTPUT/'complete.json',{'merged_test_rows':800,'dev_rows':200,'original_test_rows':272,'new_test_rows':528,
                 'new_raw_sha256':digest(OUTPUT/'segment-2.jsonl'),'boundary_case_1based':55,
                 'paired_latency_complete_cases_by_segment':[54,105],'boundary_accuracy_retained':True,
                 'identity_pass':True,'time':time.time()})
    except Exception:
        (OUTPUT/'failure.log').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
