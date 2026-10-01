"""Validate every raw prediction, denominator and identity before small tables."""
import copy
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare import ROOT,RUN,SOURCE,digest,dump
from attncopy.engine import ARMS
from attncopy.metrics import score
from attncopy.scoring_review import reviewed_score
from scripts.resume_pilot import SNAP,OUTPUT as RECOVERY,recovery_plan,merge_rows
from scripts.finish_pilot import OUT as FINAL_RECOVERY,merged_pilot


def read(path):
    return [json.loads(l) for l in path.open()]


def ratio(a,b):
    return a/b if b else None


def payload_score(record,row):
    text = row['raw'].strip()
    for opening in ['```json\n','```\n']:
        if text.startswith(opening) and text.endswith('\n```'):
            r = dict(row,raw=text[len(opening):-4],events=[])
            return reviewed_score(record,r)
    return row['scores']


def aggregate(rows,greedy,records):
    n = len(rows)
    events = [e for r in rows for e in r['events']]
    known_gold = [e for e in events if e['gold_value_prefix_supported'] is not None]
    forced = sum(e.get('forced',0) for e in events)
    accepted = sum(e.get('accepted',0) for e in events)
    total = sum(r['total_s'] for r in rows)
    timed=[r for r in rows if not r.get('latency_boundary_excluded',False)]
    paired_total=sum(r['total_s'] for r in timed)
    baseline = sum(greedy[r['id']]['total_s'] for r in timed)
    identities = sum((r['token_ids'],r['stop'],r['eos_id']) ==
                     (greedy[r['id']]['token_ids'],greedy[r['id']]['stop'],greedy[r['id']]['eos_id']) for r in rows)
    arg_den = sum(r['scores']['n_args'] for r in rows)
    payload = [payload_score(records[r['id']],r) for r in rows]
    return {'n_examples':n,'n_gold_args':arg_den,
            'correct_args':sum(r['scores']['correct_args'] for r in rows),
            'argument_em':ratio(sum(r['scores']['correct_args'] for r in rows),arg_den),
            'legacy_cached_argument_em':ratio(sum(r['legacy_cached_scores']['correct_args'] for r in rows),arg_den),
            'offline_full_call_em':sum(r['scores']['full_call_em'] for r in rows)/n,
            'tool_name_em':sum(r['scores']['tool_name_em'] for r in rows)/n,
            'strict_json_valid':sum(r['scores']['valid_json'] for r in rows)/n,
            'structure_ok':sum(r['scores']['structure_ok'] for r in rows)/n,
            'invalid_json_count':sum(not r['scores']['valid_json'] for r in rows),
            'truncated_count':sum(r['stop']!='eos' for r in rows),
            'fenced_payload_call_em_diagnostic':sum(s['full_call_em'] for s in payload)/n,
            'fenced_payload_argument_em_diagnostic':ratio(sum(s['correct_args'] for s in payload),arg_den),
            'greedy_token_eos_identity_count':identities,'greedy_token_eos_identity_rate':identities/n,
            'generated_tokens':sum(r['generated_tokens'] for r in rows),'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in rows),
            'sum_prefill_s':sum(r['prefill_s'] for r in rows),'sum_decode_s':sum(r['decode_s'] for r in rows),
            'sum_total_s':total,'mean_total_s':total/n,'paired_latency_n':len(timed),'paired_sum_total_s':paired_total,
            'speed_ratio_greedy_over_arm':baseline/paired_total,
            'median_paired_total_ratio':statistics.median(greedy[r['id']]['total_s']/r['total_s'] for r in timed),
            'sum_prompt_tokenizer_s':sum(r['prompt_tokenizer_s'] for r in rows),
            'sum_proposal_s':sum(r['proposal_s'] for r in rows),'sum_attention_gpu_s':sum(r['attention_gpu_s'] for r in rows),
            'trigger_count':len(events),'eligible_opportunities':sum(r['eligible_opportunities'] for r in rows),
            'candidate_opportunities':sum(r['candidate_opportunities'] for r in rows),
            'compatible_block_count':sum(e['compatible_block'] for e in events),
            'own_prefix_greedy_compatible_block_precision':ratio(sum(e['compatible_block'] for e in events),len(events)),
            'gold_prefix_support_known':len(known_gold),'gold_prefix_support_unknown':len(events)-len(known_gold),
            'gold_prefix_supported_count':sum(e['gold_value_prefix_supported'] for e in known_gold),
            'gold_prefix_support_precision_known_only':ratio(sum(e['gold_value_prefix_supported'] for e in known_gold),len(known_gold)),
            'example_coverage':sum(bool(r['events']) for r in rows)/n,
            'forced_tokens':forced,'accepted_tokens':accepted,'draft_tokens':sum(e['block_length'] for e in events),
            'token_coverage_forced_or_accepted':ratio(forced+accepted,sum(r['generated_tokens'] for r in rows)),
            'forced_wrong_events':sum(e.get('wrong_event',False) for e in events),
            'forced_conditional_wrong_tokens':sum(e['conditional_wrong_tokens'] for e in events if 'forced' in e),
            'mean_draft_block_length':ratio(sum(e['block_length'] for e in events),len(events)),
            'mean_forced_block_length':ratio(forced,sum('forced' in e for e in events)),
            'mean_accepted_block_length':ratio(accepted,sum('accepted' in e for e in events)),
            'pld_rejected_tokens':sum(e.get('rejected',0) for e in events),
            'causal_block_count':sum(len(r['block_traces']) for r in rows)}


def fmt(value):
    return 'NA' if value is None else f'{value:.3f}'


def main():
    assert (FINAL_RECOVERY/'complete.json').exists(), 'recovered pilot incomplete'
    recovery_plan()
    lineage = json.loads((RUN/'pilot-lineage.json').read_text())
    for file,h in lineage['code_sha256'].items():
        assert digest(ROOT/file) == h, f'run code changed: {file}'
    tables = {}
    for split,n in [('dev',40),('test',160)]:
        records = read(SNAP/'runs/pilot-20261001'/f'{split}.jsonl')
        ids = [r['id'] for r in records]
        recmap = {r['id']:r for r in records}
        raw = (read(SNAP/'runs/pilot-20261001/dev-predictions.jsonl') if split=='dev' else
               merged_pilot()[1])
        assert len(records) == n and len(raw) == n*len(ARMS)
        assert len({(r['id'],r['arm']) for r in raw}) == len(raw)
        for row in raw:
            assert row['id'] in recmap and row['arm'] in ARMS
            assert row['scores'] == score(recmap[row['id']],copy.deepcopy(row)), 'cached/raw scorer mismatch'
            row['legacy_cached_scores'] = dict(row['scores'])
            row['scores'] = reviewed_score(recmap[row['id']],row)
            assert row['generated_tokens'] == len(row['token_ids'])
            for event in row['events']:
                assert len(event['draft']) <= 8 and len(event['suffix']) <= 64
                query = recmap[row['id']]['query']
                assert query[event['char_prefix_end']:event['char_end']] == event['suffix']
                assert query[event['char_start']:event['char_prefix_end']] == event['prefix']
                assert query.encode()[event['byte_start']:event['byte_end']].decode() == event['prefix']+event['suffix']
                if 'accepted' in event:
                    assert event['cache_after_trim'] == event['cache_before']+event['accepted']
            for trace in row['block_traces']:
                assert trace['new_positions'] == [trace['prefix_cache_length'],trace['cache_after']-1]
        greedy = {r['id']:r for r in raw if r['arm']=='greedy'}
        assert set(greedy) == set(ids)
        table = {}
        for arm in ARMS:
            rows = [r for r in raw if r['arm']==arm]
            assert {r['id'] for r in rows} == set(ids)
            table[arm] = aggregate(rows,greedy,recmap)
        # Failed verified identities remain evidence and block speed claims;
        # they are never removed from accuracy denominators.
        assert table['attention-only']['greedy_token_eos_identity_count'] == n
        tables[split] = table
        if split=='test':
            segments={}
            for segment in ['segment1','segment2','segment3']:
                segments[segment]={arm:aggregate([r for r in raw if r['arm']==arm and r['source_segment']==segment and not r['latency_boundary_excluded']],greedy,recmap) for arm in ARMS}
            tables['timing_segments']=segments
    overhead = read(RUN/'overhead.jsonl')
    times = {a:sum(r['total_s'] for r in overhead if r['arm']==a) for a in ['greedy','attention-only']}
    evidence = {str(p.relative_to(ROOT)):digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='controller.lock'}
    evidence.update({str(p.relative_to(ROOT)):digest(p) for p in RECOVERY.rglob('*') if p.is_file()})
    evidence.update({str(p.relative_to(ROOT)):digest(p) for p in FINAL_RECOVERY.rglob('*') if p.is_file()})
    frozen = json.loads((RUN/'frozen-v2.json').read_text())
    selection = json.loads((RUN/'selection.json').read_text())
    output = {'run_id':'pilot-20261001','tables':tables,'selection':selection,
              'analysis_code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/analyze.py',ROOT/'attncopy/scoring_review.py']},
              'argument_assignment':'current RefCall definition: maximum correct typed leaves among same-name calls, then exact calls/key sets; historical cached raw scores retained',
              'additional_evaluation_source_sha256':{str(p):digest(p) for p in [(SOURCE / 'refcall_plain/scoring.py'),(SOURCE / 'refcall_plain/codec.py'),(SOURCE / 'refcall/reference.py')]},
              'attention_overhead':{'n_examples':4,'cap':128,'generated_tokens_per_arm':195,
                  'sum_total_s':times,'relative_increase':times['attention-only']/times['greedy']-1},
              'source_model_sha256':frozen['source_model_sha256'],'raw_artifact_sha256':evidence,
              'implementation_agent':frozen['implementation_agent'],'runtime_lineage':lineage,
              'schema_audit':frozen['schema_audit'],'selection_sha256':digest(RUN/'selection.json'),
              'recovery':json.loads((FINAL_RECOVERY/'complete.json').read_text()),
              'pilot_scope':'unofficial implementation pilot xLAM dev40/Hermes dev-derived test160; not official full benchmark',
              'limits':['one model, one seed, interrupted rotated-order pass; no repeated timing confidence intervals',
                        'case55 retained for all accuracy/identity; excluded from normal paired latency (159 cases); complete cases by segment 54/35/70',
                        frozen['grouping_limits'],frozen['heldout_scope'],
                        'strict accuracy rejects Markdown fences; separately labelled payload diagnostics do not repair primary scores',
                        'forced arms include full-block logit audit; host proposal sync and attention hooks count in total',
                        'total includes prompt tokenization, prefill, proposals, forwarding and final decode; initialization/scorer/disk IO excluded',
                        'dev attention arm uses permissive threshold 0 for prescribed selection; fixed test uses selected threshold',
                        'own-prefix compatible precision is conditional target-greedy compatibility, not global token equality after divergence',
                        'gold prefix support identifies unique-name calls; ambiguous calls stay explicitly unknown; no real tools executed']}
    test = tables['test']
    cost_advantage = test['attention+match-forced']['paired_sum_total_s'] < test['match-forced']['paired_sum_total_s']
    quality_advantage = test['attention+match-forced']['offline_full_call_em'] > test['match-forced']['offline_full_call_em']
    identity_pass=test['verifiedPLD']['greedy_token_eos_identity_count']==160
    output['verdict'] = {'total_speed_go':cost_advantage if identity_pass else None,'strict_quality_go':quality_advantage,
                         'expansion_go':cost_advantage and quality_advantage and identity_pass,'verified_identity_pass':identity_pass,
                         'speed_conclusion_allowed':identity_pass,'descriptive_attention_over_match_time_advantage':cost_advantage}
    dump(ROOT/'results/pilot-20261001.json',output)
    lines = ['# AttnCopy initial pilot','',
             '서버9 GPU0, 원본 Qwen3-0.6B BF16/SDPA. 구현 에이전트 gpt-6.1-sol. seed42, dev40(xLAM), test160(Hermes). 비공식 구현파일럿이며 공식 벤치마크 점수가 아니다.',
             '개별 함수 스키마와 정규화 요청 중복은 0건이다. 공개 모델의 원래 학습 오염과 잠재 언어 템플릿 중복은 확인할 수 없다.','',
             '## Fixed test (160 accuracy / 159 paired-latency examples)','','| Arm | Arg EM | Offline call EM | JSON valid | Token/EOS identity | Descriptive total s | Paired greedy/arm speed | Triggers | Compatible precision |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        t = test[arm]
        lines.append(f'| {arm} | {fmt(t["argument_em"])} | {fmt(t["offline_full_call_em"])} | {fmt(t["strict_json_valid"])} | {t["greedy_token_eos_identity_count"]}/160 | {t["sum_total_s"]:.2f} | {fmt(t["speed_ratio_greedy_over_arm"])} | {t["trigger_count"]} | {fmt(t["own_prefix_greedy_compatible_block_precision"])} |')
    lines += ['','## Selection and interpretation','',
              f'선택한 layer/head는 사전 고정 27/0, dev trigger {selection["dev_event_n"]}개의 mass 25분위 threshold={selection["threshold"]:.8f}, margin=0.01이다. test 결과로 재선택하지 않았다.',
              f'초기 attention-only 비용은 4개/195토큰에서 {100*output["attention_overhead"]["relative_increase"]:.2f}% 증가했다. 별도 warmup과 초기화 제외, CUDA 동기화, arm 순환 순서를 사용했다.',
              f'품질 판정: **{"GO" if quality_advantage else "NO-GO"}**. strict full-call EM에서 attention+match가 match-only보다 높은지로 판정했다.',
              f'전체 속도 판정: **결론 불가 / 확대 NO-GO**. verifiedPLD 동치 gate 실패로 lossless/가속 결론을 금지한다. 경계 제외 attention+match / match-only 시간비={test["attention+match-forced"]["paired_sum_total_s"]/test["match-forced"]["paired_sum_total_s"]:.4f}는 중단 실행의 기술적 관측치다.',
              f'확대 판정: **NO-GO**. verifiedPLD 전체 토큰·EOS 동치는 dev40/40, test{test["verifiedPLD"]["greedy_token_eos_identity_count"]}/160이다. 세 실패의 원시 출력·cache 이벤트를 보존했다.',
              '', 'strict JSON/scorer는 Markdown fence를 거부한다. 별도 fenced-payload 진단은 primary 정확도를 대체하지 않는다. 호출 순서는 EM에서 무시하며 실제 도구 실행 성공률이 아니다.',
              'Trigger precision은 각 arm의 실제 prefix에서 target-greedy와 맞는 전체 block 비율이다. Gold 지원은 문자열 prefix의 지원이며 완전한 인자 정답을 뜻하지 않는다. 중복 함수명/해석 불능은 unknown으로 남겼다.',
              '총 시간에는 prompt tokenizer, prefill, proposal/host sync, attention, causal block/검증, 최종 decode가 포함된다. forced arm의 full-block logits 정확성 audit도 포함한다. 초기화·scorer·파일 IO는 제외한다.',
              '세 segment의54/35/70개 완전 case를 각 segment 안에서 pairing했다. case55는 정확도/동치 분모에 유지하고 일반 paired latency에서 제외했다. 합산 시간은 중단 실행의 기술적 총계이며 연속 실행 성능을 뜻하지 않는다. 다운타임·모델 재로딩은 제외했다.',
              '각 arm은 한 번만 측정했으므로 시스템 잡음/순서 효과의 반복측정 불확실성은 추정하지 않았다. 생성 길이가 다른 forced arm의 시간 차이는 알고리즘 효과와 길이 변화를 모두 포함한다.',
              '', '## Evidence and reproduction','',
              '[Complete small metrics and artifact hashes](pilot-20261001.json), [reproduction](../docs/REPRODUCE.md). Raw predictions/logs/failed smoke remain ignored on server9 in `runs/pilot-20261001`. Source/model/config/code hashes link every row to the aggregates.',
              '13개 작은검사(기존11+순서/최대매칭2), 실제 PLD 6개 smoke, 0/부분/전체 KV trim, attention row(eager 대비 최대 오차 0.00086683)를 검증했다. BatchEncoding 정렬검사 실패 및 첫 0-trigger smoke를 삭제하지 않았다.']
    (ROOT/'results/PILOT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output['verdict']))


if __name__ == '__main__':
    main()
