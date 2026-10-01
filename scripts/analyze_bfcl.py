"""Official CPU AST + exact parsed strings, all 658 cases, all five arms."""
import collections
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare import ROOT,digest,dump
from attncopy.bfcl_native import score_official,UPSTREAM
from attncopy.argument_features import at_path,TOKEN_BINS,FRAGMENT_BINS,TYPE_BINS

OUT=ROOT/'runs/bfcl-658'
ARMS=['greedy','attention-only','match-forced','attention+match-forced','verifiedPLD']


def read(path):
    with path.open() as f:return [json.loads(l) for l in f]


def equal(g,r):
    return (g['token_ids'],g['stop'],g['eos_id'])==(r['token_ids'],r['stop'],r['eos_id'])


def gold_string_flags(entry,gold,scored,strata):
    name=next(iter(gold['ground_truth'][0]))
    parsed=scored['parsed_calls']
    valid=len(parsed)==1 and name in parsed[0] and isinstance(parsed[0][name],dict)
    params=parsed[0][name] if valid else {}
    flags={}
    for leaf in strata:
        if leaf.get('non_string'):continue
        path=leaf['path']
        correct=False
        if valid:
            if path[0] not in params and leaf['optional_missing']:correct=True
            else:
                try:
                    value=at_path(params,path)
                    correct=isinstance(value,str) and value in leaf['accepted']
                except (KeyError,IndexError,TypeError):pass
        flags[tuple(path)]=correct
    return flags


def event_support(event,gold):
    expected=gold['ground_truth'][0]
    name=event.get('call_name')
    if name is None:return None
    if name not in expected:return False
    path=event['path'][2:]
    if not path or path[0] not in expected[name]:return False
    for alternative in expected[name][path[0]]:
        try:value=at_path(alternative,path[1:])
        except (KeyError,IndexError,TypeError):continue
        if isinstance(value,str) and value.startswith(event['prefix']+event['suffix']):return True
    return False


def aggregate(rows,baseline):
    n=len(rows);events=[e for r in rows for e in r['events']]
    known=[e for e in events if e['gold_prefix_supported'] is not None]
    accepted=sum(e.get('accepted',0) for e in events);forced=sum(e.get('forced',0) for e in events)
    total=sum(r['total_s'] for r in rows)
    return {'n_examples':n,'official_ast_correct':sum(r['official_scores']['official_ast_valid'] for r in rows),
      'official_ast_accuracy':sum(r['official_scores']['official_ast_valid'] for r in rows)/n,
      'parsed_exact_typed_call_em':sum(r['official_scores']['parsed_raw_value_call_em'] for r in rows)/n,
      'gold_argument_n':sum(r['official_scores']['gold_argument_n'] for r in rows),
      'correct_exact_arguments':sum(r['official_scores']['correct_arguments'] for r in rows),
      'gold_string_leaf_n':sum(len(r['string_flags']) for r in rows),
      'correct_exact_string_leaves':sum(sum(r['string_flags'].values()) for r in rows),
      'exact_raw_string_leaf_em':sum(sum(r['string_flags'].values()) for r in rows)/sum(len(r['string_flags']) for r in rows),
      'parse_status_counts':dict(collections.Counter(r['official_scores']['parse_status'] for r in rows)),
      'official_error_types':dict(collections.Counter(r['official_scores']['official_ast']['error_type'] for r in rows if not r['official_scores']['official_ast_valid'])),
      'truncated_n':sum(r['stop']!='eos' for r in rows),'greedy_token_eos_identity_n':sum(equal(baseline[r['id']],r) for r in rows),
      'generated_tokens':sum(r['generated_tokens'] for r in rows),'prefill_s':sum(r['prefill_s'] for r in rows),
      'decode_s':sum(r['decode_s'] for r in rows),'total_s':total,'mean_total_s':total/n,
      'greedy_over_arm_total_ratio':sum(baseline[r['id']]['total_s'] for r in rows)/total,
      'median_paired_total_ratio':statistics.median(baseline[r['id']]['total_s']/r['total_s'] for r in rows),
      'tokens_per_s':sum(r['generated_tokens'] for r in rows)/total,
      'tokenizer_s':sum(r['prompt_tokenizer_s'] for r in rows),'proposal_s':sum(r['proposal_s'] for r in rows),
      'attention_gpu_s':sum(r['attention_gpu_s'] for r in rows),'trigger_n':len(events),
      'eligible_opportunities':sum(r['eligible_opportunities'] for r in rows),'candidate_opportunities':sum(r['candidate_opportunities'] for r in rows),
      'compatible_block_n':sum(e['compatible_block'] for e in events),
      'compatible_block_precision':sum(e['compatible_block'] for e in events)/len(events) if events else None,
      'gold_prefix_support_known_n':len(known),'gold_prefix_support_unknown_n':len(events)-len(known),
      'gold_prefix_supported_n':sum(e['gold_prefix_supported'] for e in known),
      'gold_prefix_support_precision_known':sum(e['gold_prefix_supported'] for e in known)/len(known) if known else None,
      'example_coverage':sum(bool(r['events']) for r in rows)/n,'forced_tokens':forced,'accepted_tokens':accepted,
      'draft_tokens':sum(e['block_length'] for e in events),'mean_draft_length':sum(e['block_length'] for e in events)/len(events) if events else None,
      'mean_accepted_length':accepted/len(events) if events else None,'mean_forced_length':forced/len(events) if events else None,
      'token_coverage':(forced+accepted)/sum(r['generated_tokens'] for r in rows),
      'forced_wrong_events':sum(e.get('wrong_event',False) for e in events),
      'forced_conditional_wrong_tokens':sum(e['conditional_wrong_tokens'] for e in events if 'forced' in e),
      'verified_serial_replayed_forward_tokens':sum(e.get('replayed_forward_tokens',0) for e in events),
      'official_cpu_score_s':sum(r['official_cpu_score_s'] for r in rows)}


def subgroup_tables(strata,rows):
    lookup={(r['id'],r['arm']):r for r in rows}
    tables={}
    dimensions={'token_length':('token_bin',TOKEN_BINS),'fragmentation':('fragment_bin',FRAGMENT_BINS),
                'type':('type_bin',TYPE_BINS),'source_present':('source_present_any_accepted',[False,True])}
    for name,(key,bins) in dimensions.items():
        table={}
        for value in bins:
            leaves=[s for s in strata if not s.get('non_string') and s[key]==value]
            pairs=[(s['id'],tuple(s['path'])) for s in leaves]
            counts={arm:sum(lookup[id,arm]['string_flags'][path] for id,path in pairs) for arm in ARMS}
            comparisons={}
            for base in ['greedy','match-forced']:
                a=[lookup[id,'attention+match-forced']['string_flags'][path] for id,path in pairs]
                b=[lookup[id,base]['string_flags'][path] for id,path in pairs]
                gain=sum(x and not y for x,y in zip(a,b));loss=sum(y and not x for x,y in zip(a,b))
                comparisons[base]={'wrong_to_correct':gain,'correct_to_wrong':loss,'both_correct':sum(x and y for x,y in zip(a,b)),
                                   'both_wrong':sum(not x and not y for x,y in zip(a,b)),
                                   'net_change':gain-loss,'net_change_over_n':(gain-loss)/len(pairs) if pairs else None}
            table[str(value)]={'string_leaf_n':len(pairs),'example_n':len({s['id'] for s in leaves}),'small_n':len(pairs)<20,
                               'correct':counts,'em':{a:counts[a]/len(pairs) if pairs else None for a in ARMS},'paired':comparisons}
        assert sum(v['string_leaf_n'] for v in table.values())==sum(not s.get('non_string') for s in strata)
        tables[name]=table
    return tables


def main():
    evaluation=OUT/'evaluate'
    assert (evaluation/'complete.json').exists(),'official658 GPU evaluation incomplete'
    frozen=json.loads((OUT/'frozen.json').read_text())
    raw=read(evaluation/'predictions.jsonl');runtime=read(OUT/'runtime-inputs.jsonl');strata=read(OUT/'gold-strata.jsonl')
    assert len(raw)==3290 and len({(r['id'],r['arm']) for r in raw})==3290
    assert digest(OUT/'runtime-inputs.jsonl')==frozen['runtime_sha256'] and digest(OUT/'gold-strata.jsonl')==frozen['strata_sha256']
    base=UPSTREAM/'berkeley-function-call-leaderboard/bfcl_eval/data'
    entries={};gold={}
    for category in ['simple_python','live_simple']:
        entries.update({r['id']:r for r in read(base/f'BFCL_v4_{category}.json')})
        gold.update({r['id']:r for r in read(base/'possible_answer'/f'BFCL_v4_{category}.json')})
    by_strata=collections.defaultdict(list)
    for s in strata:by_strata[s['id']].append(s)
    score_dir=OUT/'scoring';score_dir.mkdir(exist_ok=False)
    # Warm imports outside individual CPU scoring costs; no API requests.
    score_official(entries[runtime[0]['id']],gold[runtime[0]['id']],'')
    for row in raw:
        rec=runtime[row['case_index']]
        assert rec['id']==row['id'] and row['category']==rec['category']
        assert row['order']==ARMS[row['case_index']%5:]+ARMS[:row['case_index']%5]
        assert row['prompt_tokens']==len(rec['input_ids']) and row['generated_tokens']==len(row['token_ids'])
        start=time.perf_counter();row['official_scores']=score_official(entries[row['id']],gold[row['id']],row['raw'])
        row['official_cpu_score_s']=time.perf_counter()-start
        row['string_flags']=gold_string_flags(entries[row['id']],gold[row['id']],row['official_scores'],by_strata[row['id']])
        for event in row['events']:
            event['gold_prefix_supported']=event_support(event,gold[row['id']])
            source=next(s['text'] for s in rec['sources'] if s['message_index']==event['source_message_index'])
            assert source[event['char_start']:event['char_prefix_end']]==event['prefix']
            assert source[event['char_prefix_end']:event['char_end']]==event['suffix']
            assert source.encode()[event['byte_start']:event['byte_end']].decode()==event['prefix']+event['suffix']
            assert len(event['draft'])<=8 and len(event['suffix'])<=64
            if 'accepted' in event:
                assert event['cache_after_trim']==event['cache_before']
                assert event['cache_after_replay']==event['cache_before']+event['replayed_forward_tokens']
        sidecar={k:v for k,v in row.items() if k!='string_flags'}
        sidecar['string_flags']=[{'path':list(p),'correct':v} for p,v in row['string_flags'].items()]
        with (score_dir/'scored-predictions.jsonl').open('a') as f:f.write(json.dumps(sidecar,ensure_ascii=False)+'\n')
    baseline={r['id']:r for r in raw if r['arm']=='greedy'}
    tables={};subgroups={}
    for category,n in [('simple_python',400),('live_simple',258),('combined',658)]:
        rows=[r for r in raw if category=='combined' or r['category']==category]
        ss=[s for s in strata if category=='combined' or s['category']==category]
        tables[category]={}
        for arm in ARMS:
            selected=[r for r in rows if r['arm']==arm]
            assert len(selected)==n
            tables[category][arm]=aggregate(selected,baseline)
        for arm in ['attention-only','verifiedPLD']:assert tables[category][arm]['greedy_token_eos_identity_n']==n
        subgroups[category]=subgroup_tables(ss,rows)
    summaries={'scope':'full official simple_python400 + live_simple258; local HF native FC adapter, not all BFCL or a leaderboard submission',
       'upstream_commit':'6ea57973c7a6097fd7c5915698c54c17c5b1b6c8','model_registry':'Qwen/Qwen3-0.6B-FC',
       'tables':tables,'subgroups':subgroups,'frozen':{k:v for k,v in frozen.items() if k!='source_ids'},
       'raw_predictions_sha256':digest(evaluation/'predictions.jsonl'),'scored_predictions_sha256':digest(score_dir/'scored-predictions.jsonl'),
       'analysis_script_sha256':digest(__file__),'scorer_sha256':digest(ROOT/'attncopy/bfcl_native.py'),
       'source_manifest_sha256':digest(ROOT/'docs/BFCL_SOURCE_MANIFEST.json'),'runtime_lineage':json.loads((evaluation/'lineage.json').read_text()),
       'native_safety':json.loads((ROOT/json.loads((OUT/'safety-passed.json').read_text())['directory']/'passed.json').read_text()),
       'allocation':json.loads((evaluation/'allocation.json').read_text()),
       'limits':['one original model/seed, one rotated-order pass; no timing repetition uncertainty estimates',
                 'official string normalization is separated from exact parsed string leaves; all accepted alternatives retained',
                 'fixed features use first non-empty reference alternative; optional empty alternatives also encode omission; undefined references separately counted',
                 '7 empty-alternative parameters in2 cases remain in full658 AST denominator',
                 'native formatting outside timed GPU generation; parser/AST timed separately on CPU; initialization/IO excluded',
                 'verifiedPLD uses exact serial BF16 cache replay, including its costs; ordinary parallel PLD acceleration not claimed',
                 'forced arms include block-logit audit and change lengths; conditional compatibility is a parallel target proxy',
                 'pretrained model contamination unknown; no training, provider API tools, tool execution or LLM judges'],
       'command':sys.argv}
    t=tables['combined'];a=t['attention+match-forced'];g=t['greedy'];m=t['match-forced']
    summaries['verdict']={'official_ast_quality_gain_over_greedy':a['official_ast_correct']-g['official_ast_correct'],
        'official_ast_quality_gain_over_match':a['official_ast_correct']-m['official_ast_correct'],
        'raw_string_leaf_net_over_greedy':a['correct_exact_string_leaves']-g['correct_exact_string_leaves'],
        'raw_string_leaf_net_over_match':a['correct_exact_string_leaves']-m['correct_exact_string_leaves'],
        'quality_go':a['official_ast_correct']>m['official_ast_correct'] or a['correct_exact_string_leaves']>m['correct_exact_string_leaves'],
        'cost_inclusive_speed_go':a['total_s']<m['total_s'],'verified_equality_pass':True}
    dump(ROOT/'results/bfcl-658.json',summaries)
    lines=['# BFCL official two-category evaluation','',
       '공식 upstream의 simple_python400+live_simple258 전체 데이터·QwenFCHandler native prompt/response parser·official AST를 사용했다. 모델은 원본 Qwen3-0.6B BF16/SDPA이며 local HF greedy 어댑터 평가다. 전체 BFCL 점수나 공식 leaderboard 제출은 아니다.','']
    for category,table in tables.items():
        lines += [f'## {category}','','| Arm | Official AST correct/N | Exact parsed string leaves | Token/EOS identity | Total generation s | Greedy/arm ratio | Truncated |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            v=table[arm]
            lines.append(f'| {arm} | {v["official_ast_correct"]}/{v["n_examples"]} | {v["correct_exact_string_leaves"]}/{v["gold_string_leaf_n"]} | {v["greedy_token_eos_identity_n"]}/{v["n_examples"]} | {v["total_s"]:.2f} | {v["greedy_over_arm_total_ratio"]:.3f} | {v["truncated_n"]} |')
        lines.append('')
    verdict=summaries['verdict']
    lines += ['## Go/no-go','',f'품질: **{"GO (exploratory)" if verdict["quality_go"] else "NO-GO"}**. attention+match의 AST 정답 변화는 greedy 대비 {verdict["official_ast_quality_gain_over_greedy"]:+d}, match-only 대비 {verdict["official_ast_quality_gain_over_match"]:+d}; 정확한 string-leaf 변화는 각각 {verdict["raw_string_leaf_net_over_greedy"]:+d}/{verdict["raw_string_leaf_net_over_match"]:+d}이다.',
         f'총비용 속도: **{"GO (single-pass observation)" if verdict["cost_inclusive_speed_go"] else "NO-GO"}**. attention+match / match-only 총 생성 시간비={a["total_s"]/m["total_s"]:.4f}. 품질 향상을 인정하는 데 속도 향상을 조건으로 삼지 않았다.',
         'attention-only와 BF16 serial-replay verified control은 658개 전체 token/EOS 동치 gate를 통과했다. 원래 파일럿의 병렬 PLD 동치 실패3건은 삭제하거나 대체하지 않았다.',
         'AST의 case/일부 문장부호/공백 정규화와 정확한 parsed string EM은 별도다. JSON escaping 차이는 파싱으로만 해소하며 문자열을 lower/strip하지 않았다. 모든 실패·잘림과 빈 gold alternative2개 case를 전체 분모에 유지했다.',
         '기존 dev threshold/head/margin을 재튜닝하지 않았다. source/gold strata는 model prediction 전에 고정했으며 zero/low-N 그룹도 보존했다. 상세 subgroup 개선·악화·both-correct/wrong·분모와 모든 비용·hash는 [small JSON](bfcl-658.json)에 있다.',
         '총 생성 비용에는 canonical tokenization, matching/host sync, attention, causal block 및 serial replay, 최종 text decode가 포함된다. native formatting은 사전 CPU 준비이며 parser/AST 비용은 별도 CPU 기록이다. 초기화·IO는 제외했다. 단일 timing pass의 관측치이며 일반적 우월성이나 실제 도구 실행 성공률을 뜻하지 않는다.',
         '', '[Reproduction](../docs/REPRODUCE.md), [unofficial implementation pilot](PILOT.md), [pilot argument subgroups](ARGUMENT_SUBGROUPS.md).']
    (ROOT/'results/BFCL_658.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdict))


if __name__=='__main__':main()
