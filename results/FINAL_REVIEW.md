# Final bounded AttnCopy review

공식 데이터 두 카테고리 전체 simple_python400 + live_simple258, 총658개/3290 arm행. 고정 upstream/native QwenFCHandler prompt·response cleaner·official AST를 사용한 local HF adapter 결과이며 전체 BFCL/leaderboard 제출 점수는 아니다. Qwen3-0.6B 하나, BF16/SDPA, seed42, 기존 dev head27/0·threshold·margin 그대로다.

## Accuracy and costs

아래 strict whole-call은 recursive type/value postprocess-r1 하한이며 전체 case 분모를 유지한다. Gold argument1871개 중 empty-alternative7개(2 case)는 정의되지 않아, 인자 EM은 정의된1864개를 명시한다. 문자열 인자1067개와 nested string-leaf1340개는 별도로 표기한다. AST는 upstream의 문자열 정규화를 유지하고 raw 문자열은 lower/strip하지 않는다. 모든 허용 gold를 비교하며 escaping 차이만 파싱으로 해소한다.

### simple_python

| Arm | AST correct/N | Strict whole-call lower bound/N | Exact args / defined N | Raw string args/N | Raw string leaves/N | Native JSON/N | Mean total s | Prefill s | Decode s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 344/400 | 334/400 | 1077/1159 | 553/588 | 657/719 | 398/400 | 5.6116 | 12.758 | 2231.348 | 2244.622 |
| attention-only | 344/400 | 334/400 | 1077/1159 | 553/588 | 657/719 | 398/400 | 5.6836 | 13.027 | 2259.895 | 2273.432 |
| match-forced | 272/400 | 253/400 | 939/1159 | 433/588 | 516/719 | 383/400 | 5.6247 | 12.835 | 2236.346 | 2249.869 |
| attention+match-forced | 292/400 | 276/400 | 975/1159 | 467/588 | 553/719 | 384/400 | 5.7044 | 12.974 | 2268.253 | 2281.768 |
| verifiedPLD | 344/400 | 334/400 | 1077/1159 | 553/588 | 657/719 | 398/400 | 5.6489 | 12.826 | 2246.082 | 2259.556 |

### live_simple

| Arm | AST correct/N | Strict whole-call lower bound/N | Exact args / defined N | Raw string args/N | Raw string leaves/N | Native JSON/N | Mean total s | Prefill s | Decode s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 162/258 | 151/258 | 546/705 | 387/479 | 399/621 | 236/258 | 6.7887 | 8.960 | 1742.057 | 1751.493 |
| attention-only | 162/258 | 151/258 | 546/705 | 387/479 | 399/621 | 236/258 | 6.8853 | 9.097 | 1766.854 | 1776.420 |
| match-forced | 113/258 | 105/258 | 442/705 | 298/479 | 302/621 | 220/258 | 6.8372 | 8.821 | 1754.573 | 1764.005 |
| attention+match-forced | 126/258 | 118/258 | 464/705 | 317/479 | 321/621 | 223/258 | 6.9392 | 9.038 | 1780.801 | 1790.313 |
| verifiedPLD | 162/258 | 151/258 | 546/705 | 387/479 | 399/621 | 236/258 | 6.8471 | 8.922 | 1757.075 | 1766.563 |

### combined

| Arm | AST correct/N | Strict whole-call lower bound/N | Exact args / defined N | Raw string args/N | Raw string leaves/N | Native JSON/N | Mean total s | Prefill s | Decode s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 506/658 | 485/658 | 1623/1864 | 940/1067 | 1056/1340 | 634/658 | 6.0731 | 21.719 | 3973.405 | 3996.115 |
| attention-only | 506/658 | 485/658 | 1623/1864 | 940/1067 | 1056/1340 | 634/658 | 6.1548 | 22.125 | 4026.748 | 4049.851 |
| match-forced | 385/658 | 358/658 | 1381/1864 | 731/1067 | 818/1340 | 603/658 | 6.1001 | 21.655 | 3990.918 | 4013.875 |
| attention+match-forced | 418/658 | 394/658 | 1439/1864 | 784/1067 | 874/1340 | 607/658 | 6.1886 | 22.012 | 4049.054 | 4072.081 |
| verifiedPLD | 506/658 | 485/658 | 1623/1864 | 940/1067 | 1056/1340 | 634/658 | 6.1187 | 21.748 | 4003.157 | 4026.119 |

## Selection versus speed

attention+match는 match-only보다 AST +33/658 (+5.02pp), string-leaf +56/1340 (+4.18pp), raw string argument +53/1067이다. Greedy 대비 AST -88/658, string-leaf -182/1340, raw string argument -156/1067이다. 강제형이 greedy와 다르다는 사실은 gate 실패가 아니며 정확도 변화로 평가했다.
총비용은 attention+match / match-only = 1.014501, 즉 1.45% 느렸다. 빠른 unhooked SDPA greedy 대비 attention-only overhead는 1.34%이다. 별도 초기4-case 계측 +2.80%와 정식658 결과를 합치지 않았다.
판단: match-only 대비 강제형의 상대 정확도 이점은 관측했지만, greedy 대비 정확도 저하 및 비용 포함 속도 이득 없음으로 현재 설정은 확대 NO-GO. 긴/분할 많은 인자의 정확도 가설은 아래 부족한 분모 때문에 미결이다. 추가 sweep이나 GPU 실행은 하지 않았다. 단일 timing pass/하나의 모델·head·seed이며 인과적 attention 선택 우월성이나 일반적 성능을 입증하지 않는다.

| Arm | Triggers | Coverage cases/N | Mean draft tokens | Compatible-block proxy | Gold-prefix support | Forced-wrong events proxy |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0 | 0/658 | n/a | 0/0 | 0/0 | 0 |
| attention-only | 0 | 0/658 | n/a | 0/0 | 0/0 | 0 |
| match-forced | 378 | 286/658 | 1.9153439153439153 | 185/378 | 168/378 | 193 |
| attention+match-forced | 192 | 169/658 | 2.2291666666666665 | 54/192 | 47/192 | 138 |
| verifiedPLD | 408 | 286/658 | 1.9264705882352942 | 191/408 | 178/408 | 0 |

Compatible/wrong는 각 강제형 causal block의 target-logit 조건부 proxy이며 gold correctness나 실제 greedy 전체 출력을 대신하지 않는다. Gold-prefix support도 복사 suffix가 허용 정답 접두사인지의 진단이다. Attention arm은 발동빈도를 줄였지만 trigger precision은 match-only보다 낮았다. 블록 audit/토큰화/matching/host sync/attention/replay/최종 decode 비용이 total에 포함된다. Native formatting은 사전 CPU 준비, AST/parser 비용과 초기화/IO는 별도다.

## Exploratory frozen string subgroups

출력 전에 고정한 문자열 특징만 사용했다. 아래 correct와 gain/loss는 string-leaf 기준, 모든 그룹 분모·zero/low-N을 유지하며 탐색적이다. Greedy 대비 또는 match 대비 좋은 부분집합을 골라 주결과로 사용하지 않았다.

| Feature | N leaves (examples) | Greedy | Match | Attention+match | Gain/loss vs greedy | Gain/loss vs match |
|---|---:|---:|---:|---:|---:|---:|
| token_length: 0 | 55 (25) | 18 | 14 | 14 | 0/4 | 0/0 |
| token_length: 1-4 | 1030 (510) | 877 | 698 | 746 | 1/132 | 49/1 |
| token_length: 5-16 | 229 (161) | 152 | 98 | 106 | 0/46 | 8/0 |
| token_length: 17-64 | 25 (22) | 8 | 7 | 7 | 1/2 | 0/0 |
| token_length: >=65 | 1 (1) | 1 | 1 | 1 | 0/0 | 0/0 |
| fragmentation: empty | 55 (25) | 18 | 14 | 14 | 0/4 | 0/0 |
| fragmentation: (0,0.25] | 668 (386) | 537 | 426 | 455 | 0/82 | 29/0 |
| fragmentation: (0.25,0.5] | 420 (316) | 355 | 283 | 302 | 1/54 | 19/0 |
| fragmentation: (0.5,1] | 197 (153) | 146 | 95 | 103 | 1/44 | 9/1 |
| fragmentation: >1 | 0 (0) | 0 | 0 | 0 | 0/0 | 0/0 |
| type: empty | 55 (25) | 18 | 14 | 14 | 0/4 | 0/0 |
| type: URL | 13 (13) | 9 | 6 | 6 | 0/3 | 0/0 |
| type: email | 4 (4) | 2 | 1 | 1 | 0/1 | 0/0 |
| type: identifier | 186 (152) | 135 | 71 | 79 | 0/56 | 9/1 |
| type: natural_language_proxy | 441 (300) | 318 | 304 | 306 | 1/13 | 2/0 |
| type: other_string | 641 (395) | 574 | 422 | 468 | 1/107 | 46/0 |
| source_present: False | 416 (268) | 273 | 236 | 244 | 0/29 | 8/0 |
| source_present: True | 924 (480) | 783 | 582 | 630 | 2/155 | 49/1 |

17–64tokens는25 leaves/22 examples, >=65tokens는1/1, token/문자>1 그룹은0이다. 긴 인자 및 매우 많은 token 분할에 대한 판단을 이 표에서 일반화하지 않는다. URL13/email4의 작은 분모도 그대로 남겼다. 카테고리별 전체 subgroup은 원래 small JSON에 있다.

## Verification, failures and lineage

Attention-only와 수정된 serial-replay verified는 전체658개에서 token/stop/EOS 모두 greedy와 일치했다. 각 arm은 EOS657, max_new_tokens 잘림1; 모든 실패/no-call도658 분모에 남겼다. Context limit40960에 비해 입력 max394/819이며 입력 잘림/제외/거절0. 의미적 거절 여부는 자동 추정하지 않았다. Native JSON validity는 공식 cleaner 이후 하나 이상 완성된 tool_call envelope의 모든 payload가 표준 JSON인지이며 전체 XML/thinking output의 plain-JSON validity가 아니다. Scalar/duplicate key는 FC protocol에서 별도로 거절한다.

기존 pilot PLD3건(case90/129/136)은 첫차이 index106/283/106 및157/160 identity 실패를 보존했다. 첫683-prefix two-token block에서 old KV bitexact, new K maxdiff4.0/V.75(92706 elements). 위치·trim 기록과 CPU causal mask는 일관되지만 실제 GPU logits/top2 margin이 없어 near-tie/완전한 원인은 미확정이다. Lossless/일반 병렬 PLD 속도 주장은 금지한다. BFCL의 별도 parallel draft + serial BF16 KV replay는 세 원래 실패 및658개 전체 empirical gate를 통과했으며 모든 비용을 포함했다. [진단 evidence](PLD_DIAGNOSTIC_REVIEW.json).

CPU postprocess-r1은 legacy scorer의 top-level 타입 검사/Python nested equality를 recursive exact로 정정했다. simple_python13/82/87/103의20 arm행(각 arm5개 인자/3개 whole-call)이 달라졌다. 공식 AST/원래 string-leaf·생성·threshold는 불변이며 기존 sidecar/JSON은 그대로 남겨 결과집합을 구분했다. Undefined gold7개도 조용히 오답으로 확정하지 않고 별도 표시했다. [정정 revision/전체ID](BFCL_VALIDATION.json).

GPU7 이동: 2026-09-30 19:04:13UTC 유휴1MiB/0%, foreign GPU7 process 없음과 기존 shared locks 확인 후 smoke; 19:10:27UTC PID[private process identifiers withheld]로 본평가 시작, 2026-10-01 00:47:53UTC 자연종료. RTX A6000 UUID [private GPU UUID withheld], 유휴 clock210/405MHz, 관측 running1800/7601MHz. 다른 GPU 작업을 종료/변경/공유하지 않았다. 기존 중단된 GPU0 pilot54/35/70 timing segments와 혼합하지 않는다. 관리 Sol session[private agent session withheld]/CLI0.159.2/gpt-6.1-sol 유지; 중간 관리 stream 재접속 오류가 있었으나 이 BFCL GPU PID·연속 예측은 유지됐다.

Prepared frozen.json의 code hashes는 준비 시점 기록이다. 생성의 최종 별도 core revision은 smoke/evaluate lineage.json에 고정되어있고 종료 후 모두 재검증했다. Runtime 입력/IDs/gold strata/config/model/tokenizer/selection hashes는 고정됐다. 이후 CPU review 모듈과 보고서만 추가했고 원래 pilot engine 및 BFCL generation core는 덮어쓰지 않았다.

Raw predictions SHA256: `8f6967fbbfb7f2d5172c7d055e00158cabf43847ca9471b86c8da9e2b3b68f7f`. Legacy scored sidecar SHA256: `e626f790ce860a99af9a1a25ec2ef2883dc5a2a18f416f48d15505cbf3a577a6`. 전체3290 unique(ID,arm), 공식400/258 IDs, upstream source hashes, core frozen hashes, arm rotation, offsets, KV trim/replay, 공식 score 재계산과 동치 gate가 검증됐다.

[Legacy AST/string/cost JSON](bfcl-658.json), [strict/protocol review](BFCL_VALIDATION.json), [completed implementation pilot](PILOT.md), [reproduction](../docs/REPRODUCE.md), [environment](../docs/ENVIRONMENT.md). Raw 데이터/예측/모델/큰 로그는 서버에 ignored로 보존한다.
