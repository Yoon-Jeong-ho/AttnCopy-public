"""Single BFCL GPU controller, after completed pilot and native safety gates."""
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
from scripts.resume_pilot import SNAP,ARMS
from scripts.pilot import gpu_snapshot
from scripts.finish_pilot import OUT as FINAL_PILOT,merged_pilot
from scripts.pilot import append

OUT=ROOT/'runs/bfcl-658'


def lease_gpu(stack,gpu,out):
    """New BFCL resource allocation; immutable pilot resume remains pinned0."""
    from scripts.prepare import SOURCE
    files=[RUN/'controller.lock',*gpu_lock_files(SOURCE, gpu)]
    assert len(files)>=2
    hashes={str(p):digest(p) for p in files}
    for p in files:
        handle=stack.enter_context(p.open('r+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=gpu_snapshot()
    row=[r.split(', ') for r in state['gpus'].splitlines() if r.startswith(f'{gpu},')][0]
    assert int(row[2])<=8 and int(row[3])==0 and row[1] not in state['processes'],f'GPU {gpu} not idle'
    import subprocess
    clocks=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=name,uuid,clocks.current.graphics,clocks.current.memory','--format=csv,noheader'],text=True)
    assert 'RTX A6000' in clocks,'allocated GPU is not equivalent A6000'
    os.environ['CUDA_VISIBLE_DEVICES']=row[1]
    dump(out/'allocation.json',{'gpu':gpu,'uuid':row[1],'locks_sha256':hashes,'snapshot':state,'pid':os.getpid(),
         'name_clocks':clocks,'authorization':'latest user permits any genuinely idle allocated equivalent server9 GPU',
         'authorization_document_sha256':digest(ROOT/'docs/RESOURCE_UPDATE.md')})
    return hashes


def read(path):
    with path.open() as f:return [json.loads(l) for l in f]


def identity(a,b):
    return (a['token_ids'],a['stop'],a['eos_id'])==(b['token_ids'],b['stop'],b['eos_id'])


def core_hashes():
    files=['attncopy/engine.py','attncopy/safety.py','attncopy/bfcl_engine.py','attncopy/bfcl_safety.py',
           'attncopy/bfcl_native.py','scripts/bfcl_run.py','configs/bfcl-658.json']
    return {p:digest(ROOT/p) for p in files}


def validate_frozen():
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert digest(OUT/'runtime-inputs.jsonl')==frozen['runtime_sha256']
    assert digest(OUT/'gold-strata.jsonl')==frozen['strata_sha256']
    assert digest(ROOT/'configs/bfcl-658.json')==frozen['config_sha256']
    assert digest(RUN/'selection.json')==frozen['selection_sha256']
    source=json.loads((ROOT/'docs/BFCL_SOURCE_MANIFEST.json').read_text())
    base=ROOT/'data/bfcl-upstream-6ea5797'
    for rel,entry in source['artifacts'].items():assert digest(base/rel)==entry['sha256']
    config=json.loads((ROOT/'configs/bfcl-658.json').read_text())
    assert config['dtype']=='bfloat16' and config['backend']=='sdpa' and config['seed']==42
    assert config['layer']==27 and config['head']==0 and config['threshold']==1.3354860129766166e-05
    assert config['attention_margin']==.01 and config['max_new_tokens']==4096
    assert config['eos_ids']==[151643,151645] and config['max_copy_tokens']==8 and config['max_copy_chars']==64
    rows=read(OUT/'runtime-inputs.jsonl')
    assert len(rows)==658 and [r['id'] for r in rows]==frozen['source_ids']['simple_python']+frozen['source_ids']['live_simple']
    for row in rows:
        assert not {'answers','gold','ground_truth','strata'}.intersection(row)
    return frozen,config,rows


def native_smoke(engine,rows,stage):
    import torch
    from transformers import DynamicCache
    from attncopy.engine import Engine as OriginalEngine
    from attncopy.bfcl_engine import BFCLEngine
    from attncopy.safety import current_leaf
    fixtures=read(OUT/'native-fixtures.jsonl')
    warmup=engine.generate(fixtures[0],'greedy',max_new=32)
    append(stage/'warmup.jsonl',warmup)
    success=[]
    triggers=0
    for rec in fixtures+rows[:2]:
        g=engine.generate(rec,'greedy')
        p=engine.generate(rec,'verifiedPLD',engine.config['threshold'],engine.config['attention_margin'])
        a=engine.generate(rec,'attention-only')
        for row in [g,p,a]:append(stage/'raw.jsonl',row)
        assert identity(g,p) and identity(g,a),f'native smoke identity: {rec["id"]}'
        triggers+=len(p['events'])
        success.append({'id':rec['id'],'token_eos_identity':True,'copy_triggers':len(p['events']),'stop':g['stop']})
    assert triggers>0,'native smoke must exercise actual drafts'
    # Reproduce the original failing context without changing original modules.
    class ReplayPilot(BFCLEngine):
        def detect_leaf(self,text):return current_leaf(text)
        def prompt(self,record):
            ids,offsets,qoffset=OriginalEngine.prompt(self,record)
            self.source_alignment=[(record['query'],offsets,qoffset,0)]
            return ids,offsets,qoffset
    regression=ReplayPilot.__new__(ReplayPilot)
    regression.__dict__.update(engine.__dict__)
    original_records,original_rows=merged_pilot()
    regression_results=[]
    for index in [89,128,135]:
        rec=dict(original_records[index],category=original_records[index]['source'],max_new_tokens=512)
        g=regression.generate(rec,'greedy',max_new=512)
        p=regression.generate(rec,'verifiedPLD',engine.config['threshold'],engine.config['attention_margin'],max_new=512)
        old=next(r for r in original_rows if r['example_index']==index and r['arm']=='greedy')
        append(stage/'pilot-failure-regression.jsonl',g);append(stage/'pilot-failure-regression.jsonl',p)
        assert identity(g,old),'repeated original greedy changed'
        assert identity(g,p) and p['events'],'serial replay failed preserved-case regression'
        regression_results.append({'case_1based':index+1,'identity':True,'greedy_tokens':len(g['token_ids']),'repaired_tokens':len(p['token_ids']),'draft_events':len(p['events'])})
    # Isolate block-vs-single BF16 KV on the first accepted original draft.
    original=dict(original_records[89],category=original_records[89]['source'],max_new_tokens=512)
    old=next(r for r in original_rows if r['example_index']==89 and r['arm']=='greedy')
    first=next(r for r in original_rows if r['example_index']==89 and r['arm']=='verifiedPLD')['events'][0]
    prompt,_,_=OriginalEngine.prompt(regression,original)
    caches=[]
    regression.attention.set_enabled(False)
    with torch.inference_mode():
        for _ in range(2):
            cache=DynamicCache(config=engine.model.config)
            regression.forward(prompt,cache)
            for token in old['token_ids'][:first['output_start']]:regression.forward([token],cache)
            caches.append(cache)
        prefix=caches[0].get_seq_length()
        regression.forward(first['draft'],caches[0],all_logits=True)
        for token in first['draft']:regression.forward([token],caches[1])
        previous_equal=True;max_key=max_value=0.;different_elements=0
        for parallel,serial in zip(caches[0].layers,caches[1].layers):
            previous_equal &= torch.equal(parallel.keys[:,:,:prefix],serial.keys[:,:,:prefix]) and torch.equal(parallel.values[:,:,:prefix],serial.values[:,:,:prefix])
            max_key=max(max_key,float((parallel.keys[:,:,prefix:].float()-serial.keys[:,:,prefix:].float()).abs().max()))
            max_value=max(max_value,float((parallel.values[:,:,prefix:].float()-serial.values[:,:,prefix:].float()).abs().max()))
            different_elements+=int((parallel.keys[:,:,prefix:]!=serial.keys[:,:,prefix:]).sum())+int((parallel.values[:,:,prefix:]!=serial.values[:,:,prefix:]).sum())
        assert previous_equal
    dump(stage/'passed.json',{'native':success,'native_triggers':triggers,'preserved_pilot_regression_identity':True,
         'preserved_failure_regressions':regression_results,
         'bf16_kv_diagnostic':{'prefix_length':prefix,'old_prefix_bit_exact':previous_equal,'max_new_key_difference':max_key,
                              'max_new_value_difference':max_value,'different_new_kv_elements':different_elements},
         'verified_variant':'parallel block + exact serial BF16 cache replay','speed_claim':'none for original failed pilot or ordinary parallel PLD'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','evaluate'],required=True)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--wait-for-gpu',action='store_true');args=parser.parse_args()
    require_authorized_host()
    assert (FINAL_PILOT/'complete.json').exists(),'pilot must complete naturally first'
    merged_pilot()  # Every original row remains immutable; all 1000 rows present.
    frozen,config,rows=validate_frozen()
    slug=args.stage if args.attempt==1 else f'{args.stage}-attempt-{args.attempt}'
    stage=OUT/slug;stage.mkdir(exist_ok=False)
    dump(stage/'claim.json',{'pid':os.getpid(),'command':sys.argv,'time':time.time(),'code_sha256':core_hashes()})
    if args.stage=='evaluate':
        passed=json.loads((OUT/'safety-passed.json').read_text())
        smoke_path=ROOT/passed['directory']
        assert digest(smoke_path/'passed.json')==passed['passed_sha256']
        assert args.gpu==json.loads((smoke_path/'allocation.json').read_text())['gpu']
        smoke_code=json.loads((smoke_path/'claim.json').read_text())['code_sha256']
        assert smoke_code==core_hashes(),'BFCL generation/scorer code changed since safety smoke'
    try:
        with ExitStack() as stack:
            while True:
                candidate=ExitStack()
                try:
                    locks=lease_gpu(candidate,args.gpu,stage)
                    stack.enter_context(candidate.pop_all())
                    break
                except (BlockingIOError,AssertionError) as error:
                    candidate.close()
                    if not args.wait_for_gpu:raise
                    dump(stage/'waiting-for-gpu.json',{'gpu':args.gpu,'pid':os.getpid(),'time':time.time(),'reason':str(error),'model_loaded':False})
                    if (stage/'cancel-wait').exists():
                        dump(stage/'wait-cancelled.json',{'gpu':args.gpu,'model_loaded':False,'time':time.time()})
                        return
                    time.sleep(30)
            for p,h in frozen['model_tokenizer_sha256'].items():assert digest(p)==h
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            import torch,transformers
            from attncopy.bfcl_engine import BFCLEngine
            torch.set_num_threads(1);torch.manual_seed(42)
            dump(stage/'lineage.json',{'frozen_sha256':digest(OUT/'frozen.json'),'code_sha256':core_hashes(),
                  'config_sha256':digest(ROOT/'configs/bfcl-658.json'),'model_tokenizer_asserted':True,
                  'torch':torch.__version__,'transformers':transformers.__version__,'command':sys.argv,'session':os.environ.get('CODEX_SESSION_ID'),
                  'qwen_runtime_sha256':digest(sys.modules['transformers.models.qwen3.modeling_qwen3'].__file__),
                  'verified_variant':'parallel verification + exact serial BF16 cache replay','native_formatting':'frozen official CPU formatting outside timed generation',
                  'timing':'canonical tokenizer/prefill/decode/matching/block+replay/attention/final decode; excludes model init, official parser/AST and disk IO'})
            engine=BFCLEngine(config)
            if args.stage=='smoke':
                native_smoke(engine,rows,stage)
                dump(OUT/'safety-passed.json',{'directory':str(stage.relative_to(ROOT)),'passed_sha256':digest(stage/'passed.json')})
            else:
                warmup=engine.generate(read(OUT/'native-fixtures.jsonl')[0],'greedy',max_new=32)
                append(stage/'warmup.jsonl',warmup)
                original_warmup=read(smoke_path/'warmup.jsonl')[0]
                assert identity(warmup,original_warmup),'native warmup changed'
                for i,rec in enumerate(rows):
                    order=ARMS[i%5:]+ARMS[:i%5];completed={}
                    for arm in order:
                        assert not rec['context_rejected'],'context-rejected rows require explicit denominator status'
                        result=engine.generate(rec,arm,config['threshold'],config['attention_margin'])
                        result.update(case_index=i,category_index=rec['category_index'],order=order)
                        append(stage/'predictions.jsonl',result);completed[arm]=result
                    for arm in ['attention-only','verifiedPLD']:
                        if not identity(completed['greedy'],completed[arm]):
                            dump(stage/'identity-failure.json',{'id':rec['id'],'case_index':i,'arm':arm,'raw_preserved':True})
                            raise RuntimeError('BFCL full-token/EOS identity failure')
                    print(f'BFCL {i+1}/658 {rec["category"]} triggers '+str({a:len(r['events']) for a,r in completed.items()}),flush=True)
                assert len(read(stage/'predictions.jsonl'))==3290
            assert all(digest(p)==h for p,h in locks.items())
            assert core_hashes()==json.loads((stage/'claim.json').read_text())['code_sha256']
            dump(stage/'complete.json',{'stage':args.stage,'time':time.time(),'pid':os.getpid(),'natural_exit':True})
    except Exception:
        (stage/'failure.log').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
