"""Complete unchanged pilot inference; retain/report every identity failure."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attncopy.paths import require_authorized_host, gpu_lock_files
from scripts.prepare import ROOT,digest,dump
from scripts.resume_pilot import SNAP,OUTPUT as FIRST_RECOVERY,recovery_plan,validate_rows,ARMS,lease_gpu
from scripts.pilot import append,fixtures

OUT=ROOT/'runs/pilot-final-segment-20261001'


def read(path):
    with path.open() as f:return [json.loads(l) for l in f]


def merged_pilot(require_complete=True):
    records,old,*_=recovery_plan()
    segments=[('segment1',old),('segment2',read(FIRST_RECOVERY/'segment-2.jsonl'))]
    if (OUT/'segment-3.jsonl').exists():segments.append(('segment3',read(OUT/'segment-3.jsonl')))
    pairs={}
    for segment,rows in segments:
        validate_rows(rows,records)
        for row in rows:
            pair=row['id'],row['arm']
            assert pair not in pairs
            pairs[pair]=dict(row,source_segment=segment,latency_boundary_excluded=row['example_index']==54)
    if require_complete:assert len(pairs)==800
    return records,[pairs[r['id'],a] for i,r in enumerate(records) for a in ARMS[i%5:]+ARMS[:i%5] if (r['id'],a) in pairs]


def identity_failures(rows):
    failures=[]
    grouped={}
    for r in rows:grouped.setdefault(r['id'],{})[r['arm']]=r
    for id,arms in grouped.items():
        if len(arms)!=5:continue
        g=arms['greedy']
        for arm in ['attention-only','verifiedPLD']:
            r=arms[arm]
            if (r['token_ids'],r['stop'],r['eos_id'])!=(g['token_ids'],g['stop'],g['eos_id']):
                failures.append({'id':id,'case_index':g['example_index'],'arm':arm,'greedy_tokens':len(g['token_ids']),
                                 'arm_tokens':len(r['token_ids']),'full_identity':False})
    return failures


def main():
    require_authorized_host()
    assert (FIRST_RECOVERY/'failure.log').exists() and (FIRST_RECOVERY/'identity-failure.json').exists()
    records,rows=merged_pilot(False)
    immutable={str(p):digest(p) for p in FIRST_RECOVERY.iterdir() if p.is_file()}
    seen={(r['id'],r['arm']) for r in rows}
    todo=[(i,a) for i,r in enumerate(records) for a in ARMS[i%5:]+ARMS[:i%5] if (r['id'],a) not in seen]
    assert len(todo)==350 and todo[0]==(90,'greedy')
    OUT.mkdir(exist_ok=False)
    dump(OUT/'claim.json',{'command':sys.argv,'pid':os.getpid(),'time':time.time(),'missing_pairs':len(todo),'failure_policy':'preserve/report equality failures; unchanged inference, no speed conclusion'})
    try:
        with ExitStack() as stack:
            locks=lease_gpu(stack,0,OUT)
            _,_,_,cfg,frozen,selected,_=recovery_plan()
            for p,h in frozen['source_model_sha256'].items():assert digest(p)==h
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            import torch,transformers
            from attncopy.engine import Engine
            from attncopy.metrics import score
            torch.set_num_threads(1);torch.manual_seed(42)
            dump(OUT/'lineage.json',{'controller_sha256':digest(__file__),'source_model_asserted':True,
                 'config_sha256':digest(ROOT/'configs/pilot.json'),'selection_sha256':digest(SNAP/'runs/pilot-20261001/selection.json'),
                 'prior_segment_immutable_sha256':immutable,'original_engine_sha256':digest(ROOT/'attncopy/engine.py'),
                 'torch':torch.__version__,'transformers':transformers.__version__,'command':sys.argv,'session':os.environ.get('CODEX_SESSION_ID')})
            engine=Engine(cfg)
            warmup=engine.generate(fixtures()[0],'greedy',max_new=32)
            append(OUT/'warmup.jsonl',warmup)
            original_warmup=read(SNAP/'runs/pilot-20261001/pilot-warmup.jsonl')[0]
            assert (warmup['token_ids'],warmup['stop'])==(original_warmup['token_ids'],original_warmup['stop'])
            for j,(i,arm) in enumerate(todo):
                r=engine.generate(records[i],arm,selected['threshold'],selected['margin'])
                r.update(split='test',example_index=i,order=ARMS[i%5:]+ARMS[:i%5])
                r['scores']=score(records[i],r)
                append(OUT/'segment-3.jsonl',r)
                rows.append(r)
                if j%5==4:
                    failures=identity_failures(rows)
                    dump(OUT/'identity-audit.json',{'failures':failures,'speed_conclusion_allowed':not failures})
                print(f'final segment case {i+1}/160 {arm} ({j+1}/{len(todo)})',flush=True)
            records,merged=merged_pilot()
            failures=identity_failures(merged)
            recovery_plan()
            assert all(digest(p)==h for p,h in immutable.items())
            assert all(digest(p)==h for p,h in locks.items())
            dump(OUT/'complete.json',{'dev_rows':200,'merged_test_rows':800,'segment_rows':[272,178,350],
                'segment_complete_paired_cases':[54,35,70],'boundary_case_1based':55,'boundary_accuracy_retained':True,
                'identity_failures':failures,'identity_pass':not failures,'speed_conclusion_allowed':not failures,
                'inference_completed':True,'natural_exit':True,'new_raw_sha256':digest(OUT/'segment-3.jsonl'),'time':time.time()})
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
