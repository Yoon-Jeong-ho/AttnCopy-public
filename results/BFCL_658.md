# BFCL official two-category evaluation

공식 upstream의 simple_python400+live_simple258 전체 데이터·QwenFCHandler native prompt/response parser·official AST를 사용했다. 모델은 원본 Qwen3-0.6B BF16/SDPA이며 local HF greedy 어댑터 평가다. 전체 BFCL 점수나 공식 leaderboard 제출은 아니다.

## simple_python

| Arm | Official AST correct/N | Exact parsed string leaves | Token/EOS identity | Total generation s | Greedy/arm ratio | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 344/400 | 657/719 | 400/400 | 2244.62 | 1.000 | 0 |
| attention-only | 344/400 | 657/719 | 400/400 | 2273.43 | 0.987 | 0 |
| match-forced | 272/400 | 516/719 | 292/400 | 2249.87 | 0.998 | 0 |
| attention+match-forced | 292/400 | 553/719 | 325/400 | 2281.77 | 0.984 | 0 |
| verifiedPLD | 344/400 | 657/719 | 400/400 | 2259.56 | 0.993 | 0 |

## live_simple

| Arm | Official AST correct/N | Exact parsed string leaves | Token/EOS identity | Total generation s | Greedy/arm ratio | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 162/258 | 399/621 | 258/258 | 1751.49 | 1.000 | 1 |
| attention-only | 162/258 | 399/621 | 258/258 | 1776.42 | 0.986 | 1 |
| match-forced | 113/258 | 302/621 | 190/258 | 1764.01 | 0.993 | 1 |
| attention+match-forced | 126/258 | 321/621 | 205/258 | 1790.31 | 0.978 | 1 |
| verifiedPLD | 162/258 | 399/621 | 258/258 | 1766.56 | 0.991 | 1 |

## combined

| Arm | Official AST correct/N | Exact parsed string leaves | Token/EOS identity | Total generation s | Greedy/arm ratio | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 506/658 | 1056/1340 | 658/658 | 3996.12 | 1.000 | 1 |
| attention-only | 506/658 | 1056/1340 | 658/658 | 4049.85 | 0.987 | 1 |
| match-forced | 385/658 | 818/1340 | 482/658 | 4013.87 | 0.996 | 1 |
| attention+match-forced | 418/658 | 874/1340 | 530/658 | 4072.08 | 0.981 | 1 |
| verifiedPLD | 506/658 | 1056/1340 | 658/658 | 4026.12 | 0.993 | 1 |

## Go/no-go

품질: **GO (exploratory)**. attention+match의 AST 정답 변화는 greedy 대비 -88, match-only 대비 +33; 정확한 string-leaf 변화는 각각 -182/+56이다.
총비용 속도: **NO-GO**. attention+match / match-only 총 생성 시간비=1.0145. 품질 향상을 인정하는 데 속도 향상을 조건으로 삼지 않았다.
attention-only와 BF16 serial-replay verified control은 658개 전체 token/EOS 동치 gate를 통과했다. 원래 파일럿의 병렬 PLD 동치 실패3건은 삭제하거나 대체하지 않았다.
AST의 case/일부 문장부호/공백 정규화와 정확한 parsed string EM은 별도다. JSON escaping 차이는 파싱으로만 해소하며 문자열을 lower/strip하지 않았다. 모든 실패·잘림과 빈 gold alternative2개 case를 전체 분모에 유지했다.
기존 dev threshold/head/margin을 재튜닝하지 않았다. source/gold strata는 model prediction 전에 고정했으며 zero/low-N 그룹도 보존했다. 상세 subgroup 개선·악화·both-correct/wrong·분모와 모든 비용·hash는 [small JSON](bfcl-658.json)에 있다.
총 생성 비용에는 canonical tokenization, matching/host sync, attention, causal block 및 serial replay, 최종 text decode가 포함된다. native formatting은 사전 CPU 준비이며 parser/AST 비용은 별도 CPU 기록이다. 초기화·IO는 제외했다. 단일 timing pass의 관측치이며 일반적 우월성이나 실제 도구 실행 성공률을 뜻하지 않는다.

[Reproduction](../docs/REPRODUCE.md), [unofficial implementation pilot](PILOT.md), [pilot argument subgroups](ARGUMENT_SUBGROUPS.md).

## Subsequent CPU strict-value review

See [final reviewed report](FINAL_REVIEW.md) and [recursive-type/protocol revision](BFCL_VALIDATION.json). Legacy AST/string-leaf/timing values are unchanged. Legacy top-level typed call/argument fields in bfcl-658.json remain historical; recursive exact metrics use the separate review revision and explicitly count7 undefined gold arguments in2 cases.
