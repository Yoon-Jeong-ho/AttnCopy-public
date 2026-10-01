"""Single controller: GPU lease -> overhead -> smoke -> dev freeze -> test."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attncopy.paths import require_authorized_host, gpu_lock_files
from scripts.prepare import ROOT,RUN,SOURCE,digest,dump


def append(path,row):
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return [json.loads(l) for l in path.open()]


def gpu_snapshot():
    q = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    p = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    return {'gpus':q,'processes':p,'timestamp':time.time()}


def lease(stack,stage):
    before = gpu_snapshot()
    pin = RUN/'gpu-pin.json'
    candidates = [json.loads(pin.read_text())['gpu']] if pin.exists() else [0,2,7]
    for gpu in candidates:
        local = ExitStack()
        paths = gpu_lock_files(SOURCE, gpu)
        if not paths:
            continue
        identities = {str(p):digest(p) for p in paths}
        try:
            # NFS exclusive advisory locks need O_RDWR, but never write/truncate
            # source lock files. Their bytes are verified unchanged on release.
            for p in paths:
                handle = local.enter_context(p.open('r+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            now = gpu_snapshot()
            gpurow = [r.strip().split(', ') for r in now['gpus'].splitlines() if r.startswith(f'{gpu},')][0]
            uuid = gpurow[1]
            if int(gpurow[2]) > 8 or int(gpurow[3]) != 0 or uuid in now['processes']:
                local.close()
                continue
            stack.enter_context(local.pop_all())
            os.environ['CUDA_VISIBLE_DEVICES'] = uuid
            if not pin.exists():
                dump(pin,{'gpu':gpu,'uuid':uuid})
            dump(RUN/f'{stage}-allocation.json',{'gpu':gpu,'uuid':uuid,'locks_sha256':identities,
                 'host':socket.gethostname(),'pid':os.getpid(),'user_allocation':'one genuinely idle server9 GPU authorized',
                 'before':before,'at_lease':now})
            return identities
        except BlockingIOError:
            local.close()
    raise RuntimeError('no idle authorized GPU with all existing locks available')


def fixtures():
    def rec(id,query,value):
        return {'id':id,'source':'fixture','tools':[{'name':'echo','description':'Return the exact supplied text.',
                 'parameters':{'text':{'type':'str','description':'Text to return.'}}}],
                 'query':query,'answers':[{'name':'echo','arguments':{'text':value}}]}
    return [rec('fixture-long','Call echo with text "ABCD987654321abcdefghijk".','ABCD987654321abcdefghijk'),
            rec('fixture-unicode','Call echo with text "한글café".','한글café'),
            rec('fixture-duplicate','Call echo with text "ABCDEFGH12345", not "ABCDEFGH99999".','ABCDEFGH12345'),
            rec('fixture-space','Call echo with text "London bridge".','London bridge')]


def smoke(engine,records,outdir):
    import torch
    from transformers import DynamicCache
    from attncopy.safety import verified_prefix
    outdir.mkdir()
    successes = []
    for rec in fixtures()+records[:2]:
        baseline = engine.generate(rec,'greedy')
        verified = engine.generate(rec,'verifiedPLD')
        append(outdir/'raw.jsonl',baseline)
        append(outdir/'raw.jsonl',verified)
        identity = (baseline['token_ids'],baseline['stop'],baseline['eos_id']) == (verified['token_ids'],verified['stop'],verified['eos_id'])
        successes.append({'id':rec['id'],'identity':identity,'triggers':len(verified['events'])})
        if not identity:
            dump(outdir/'failure.json',successes)
            raise RuntimeError('verifiedPLD token/EOS smoke mismatch: '+rec['id'])
    assert sum(x['triggers'] for x in successes) > 0, 'smoke must exercise actual copying'
    # Selected GPU QK row versus eager's same last query/head, diagnostic only.
    prompt,_,_ = engine.prompt(fixtures()[0])
    engine.attention.set_enabled(True)
    observed = []
    def compare(module,args,kwargs,output):
        observed.append(float((engine.attention.row-output[1][0,engine.config['head'],-1].float()).abs().max()))
    handle = engine.attention.module.register_forward_hook(compare,with_kwargs=True)
    engine.model.set_attn_implementation('eager')
    with torch.inference_mode():
        engine.forward(prompt,DynamicCache(config=engine.model.config))
    handle.remove()
    engine.model.set_attn_implementation('sdpa')
    engine.attention.set_enabled(False)
    # BF16 softmax/output casting: selected float row can differ by 1 BF16 ULP.
    assert max(observed) < .004, observed
    # Real KV rejection: wrong first or later draft, then crop and reproduce
    # the sequential target's next logits/cache length from the same prefix.
    checks = []
    with torch.inference_mode():
        for wanted in [0,1,3]:
            cache = DynamicCache(config=engine.model.config)
            logits,cache = engine.forward(prompt,cache)
            next_id = int(logits[-1].argmax())
            draft, sequential = [], []
            serial = DynamicCache(config=engine.model.config)
            sl,serial = engine.forward(prompt,serial)
            for j in range(3):
                token = int(sl[-1].argmax())
                draft.append(token)
                sl,serial = engine.forward([token],serial)
                sequential.append(sl[-1].clone())
            if wanted < 3:
                draft[wanted] = (draft[wanted]+1) % engine.model.config.vocab_size
            old = cache.get_seq_length()
            bl,cache = engine.forward(draft,cache,all_logits=True)
            accepted,mismatch = verified_prefix(draft,next_id,bl.argmax(-1).tolist())
            assert accepted == wanted,(wanted,accepted)
            cache.crop(old+accepted)
            assert cache.get_seq_length() == old+accepted
            if mismatch is not None:
                ll,cache = engine.forward([mismatch],cache)
                expected = sequential[accepted]
                assert int(ll[-1].argmax()) == int(expected.argmax()), 'trim continuation argmax mismatch'
            checks.append({'accepted':accepted,'rejected':3-accepted,'prefix':old,'trimmed':old+accepted})
    dump(outdir/'passed.json',{'greedy_eos_identity':successes,'attention_row_max_abs_error':max(observed),
                              'real_kv_trim_checks':checks,'backend':'sdpa','dtype':engine.config['dtype']})
    print('SMOKE PASSED',flush=True)


def run_split(engine,records,split,threshold,margin,out):
    from attncopy.engine import ARMS
    from attncopy.metrics import score
    for i,rec in enumerate(records):
        rotation = i % len(ARMS)
        order = ARMS[rotation:]+ARMS[:rotation]
        results = {}
        for arm in order:
            result = engine.generate(rec,arm,threshold,margin)
            result.update(split=split,example_index=i,order=order)
            result['scores'] = score(rec,result)
            results[arm] = result
            append(out,result)
        greedy = results['greedy']
        for arm in ['attention-only','verifiedPLD']:
            row = results[arm]
            if (row['token_ids'],row['stop'],row['eos_id']) != (greedy['token_ids'],greedy['stop'],greedy['eos_id']):
                dump(RUN/f'{split}-identity-failure.json',{'example_index':i,'id':rec['id'],'arm':arm,
                         'raw_path':str(out),'all_five_predictions_preserved':True})
                raise RuntimeError(f'{split} identity failure: {rec["id"]} {arm}')
        print(f'{split} {i+1}/{len(records)} triggers '+str({a:len(r['events']) for a,r in results.items()}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['overhead','smoke','pilot'],required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    require_authorized_host()
    own_lock = (RUN/'controller.lock').open('a+')
    fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stage = args.stage
    slug = stage if args.attempt == 1 else f'{stage}-attempt-{args.attempt}'
    claim = RUN/f'{slug}-claim.json'
    with claim.open('x') as f:
        json.dump({'pid':os.getpid(),'time':time.time(),'stage':stage},f)
    try:
        with ExitStack() as stack:
            locks = lease(stack,slug)
            os.environ['HF_HUB_OFFLINE']='1'
            os.environ['TRANSFORMERS_OFFLINE']='1'
            import torch,transformers
            from attncopy.engine import Engine
            torch.set_num_threads(1)
            torch.manual_seed(42)
            config = json.loads((ROOT/'configs/pilot.json').read_text())
            frozen = json.loads((RUN/'frozen-v2.json').read_text())
            assert digest(ROOT/'configs/pilot.json') == frozen['config_sha256']
            for p,h in frozen['files'].items():
                assert digest(p) == h
            code = {str(p.relative_to(ROOT)):digest(p) for folder in ['attncopy','scripts','tests'] for p in (ROOT/folder).glob('*.py')}
            dump(RUN/f'{slug}-lineage.json',{'code_sha256':code,'config_sha256':digest(ROOT/'configs/pilot.json'),
                 'frozen_sha256':digest(RUN/'frozen-v2.json'),'torch':torch.__version__,'transformers':transformers.__version__,
                 'command':sys.argv,'pid':os.getpid(),'session':os.environ.get('CODEX_SESSION_ID'),
                 'qwen_runtime_sha256':digest(sys.modules['transformers.models.qwen3.modeling_qwen3'].__file__)})
            init = time.perf_counter()
            engine = Engine(config)
            print('MODEL LOADED',time.perf_counter()-init,flush=True)
            dev = read(RUN/'dev.jsonl')
            warmup = engine.generate(fixtures()[0],'greedy',max_new=32)
            append(RUN/f'{slug}-warmup.jsonl',warmup)
            if stage == 'overhead':
                # Four matched dev prompts; two order rotations, no copying.
                for i,rec in enumerate(dev[:4]):
                    for arm in (['greedy','attention-only'] if i%2 == 0 else ['attention-only','greedy']):
                        result = engine.generate(rec,arm,max_new=128)
                        append(RUN/'overhead.jsonl',result)
                print('OVERHEAD COMPLETE',flush=True)
            elif stage == 'smoke':
                assert (RUN/'overhead-complete.json').exists()
                smoke_dir = RUN/slug
                smoke(engine,dev,smoke_dir)
                dump(RUN/'smoke-passed.json',{'path':str(smoke_dir/'passed.json'),'sha256':digest(smoke_dir/'passed.json')})
            else:
                assert (RUN/'smoke-passed.json').exists()
                run_split(engine,dev,'dev',0.,config['attention_margin'],RUN/'dev-predictions.jsonl')
                rows = read(RUN/'dev-predictions.jsonl')
                masses = sorted(e['attention_mass'] for r in rows if r['arm']=='attention+match-forced' for e in r['events'])
                threshold = masses[int(.25*(len(masses)-1))] if masses else 1.
                selected = {'layer':config['layer'],'head':config['head'],'threshold':threshold,
                     'margin':config['attention_margin'],'rule':config['selection'],'dev_event_n':len(masses),
                     'dev_predictions_sha256':digest(RUN/'dev-predictions.jsonl'),'config_sha256':digest(ROOT/'configs/pilot.json')}
                dump(RUN/'selection.json',selected)
                print('DEV SELECTION FROZEN',selected,flush=True)
                test = read(RUN/'test.jsonl')
                run_split(engine,test,'test',threshold,config['attention_margin'],RUN/'test-predictions.jsonl')
            assert all(digest(p) == h for p,h in locks.items()), 'source lock bytes changed'
            dump(RUN/f'{slug}-complete.json',{'stage':stage,'time':time.time(),'pid':os.getpid()})
    except Exception:
        (RUN/f'{slug}-failure.log').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
