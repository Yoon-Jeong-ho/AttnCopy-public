# Exploratory pilot argument subgroups

비공식 xLAM40/Hermes160 구현파일럿의 사후 기술 분석이다. 전체 공식 벤치마크나 확증적 endpoint가 아니다. 잘못된/누락 출력은 분모에 유지했다.
Gold의 원래 unescaped 문자열과 고정 tokenizer로 문자·토큰·분할·유형을 계산했다. natural-language는 표면형 proxy다. 같은 함수명은 최대 정답 leaf 매칭을 사용하며 중복 호출의 그룹 귀속은 모호할 수 있다. N<20은 작은 표본으로 표시한다.

## dev: strings 67, non-string leaves 68

### strict_json_primary / token_length

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| 1-4 | 53 | 24 | 0 | 0 | 0 | 0/0 | 0/0 |
| 5-16 (small N) | 10 | 9 | 0 | 0 | 0 | 0/0 | 0/0 |
| 17-64 (small N) | 4 | 3 | 0 | 0 | 0 | 0/0 | 0/0 |
| >=65 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### strict_json_primary / fragmentation

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0,0.25] (small N) | 19 | 14 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0.25,0.5] | 33 | 22 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0.5,1] (small N) | 15 | 12 | 0 | 0 | 0 | 0/0 | 0/0 |
| >1 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### strict_json_primary / type

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| URL (small N) | 1 | 1 | 0 | 0 | 0 | 0/0 | 0/0 |
| email (small N) | 1 | 1 | 0 | 0 | 0 | 0/0 | 0/0 |
| identifier | 22 | 12 | 0 | 0 | 0 | 0/0 | 0/0 |
| natural_language_proxy (small N) | 12 | 10 | 0 | 0 | 0 | 0/0 | 0/0 |
| other_string | 31 | 16 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / token_length

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| 1-4 | 53 | 24 | 41 | 24 | 24 | 1/18 | 0/0 |
| 5-16 (small N) | 10 | 9 | 7 | 2 | 2 | 0/5 | 0/0 |
| 17-64 (small N) | 4 | 3 | 3 | 2 | 2 | 0/1 | 0/0 |
| >=65 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / fragmentation

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0,0.25] (small N) | 19 | 14 | 17 | 9 | 9 | 0/8 | 0/0 |
| (0.25,0.5] | 33 | 22 | 23 | 15 | 15 | 1/9 | 0/0 |
| (0.5,1] (small N) | 15 | 12 | 11 | 4 | 4 | 0/7 | 0/0 |
| >1 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / type

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |
| URL (small N) | 1 | 1 | 0 | 0 | 0 | 0/0 | 0/0 |
| email (small N) | 1 | 1 | 1 | 0 | 0 | 0/1 | 0/0 |
| identifier | 22 | 12 | 18 | 6 | 6 | 0/12 | 0/0 |
| natural_language_proxy (small N) | 12 | 10 | 11 | 10 | 10 | 0/1 | 0/0 |
| other_string | 31 | 16 | 21 | 12 | 12 | 1/10 | 0/0 |

## test: strings 969, non-string leaves 131

### strict_json_primary / token_length

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| 1-4 | 558 | 137 | 0 | 0 | 0 | 0/0 | 0/0 |
| 5-16 | 367 | 118 | 0 | 0 | 0 | 0/0 | 0/0 |
| 17-64 | 29 | 23 | 0 | 0 | 0 | 0/0 | 0/0 |
| >=65 (small N) | 12 | 8 | 0 | 0 | 0 | 0/0 | 0/0 |

### strict_json_primary / fragmentation

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0,0.25] | 471 | 130 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0.25,0.5] | 224 | 90 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0.5,1] | 271 | 83 | 0 | 0 | 0 | 0/0 | 0/0 |
| >1 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### strict_json_primary / type

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| URL | 20 | 9 | 0 | 0 | 0 | 0/0 | 0/0 |
| email | 24 | 15 | 0 | 0 | 0 | 0/0 | 0/0 |
| identifier | 436 | 121 | 0 | 0 | 0 | 0/0 | 0/0 |
| natural_language_proxy | 214 | 83 | 0 | 0 | 0 | 0/0 | 0/0 |
| other_string | 272 | 98 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / token_length

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| 1-4 | 558 | 137 | 432 | 242 | 286 | 1/147 | 59/15 |
| 5-16 | 367 | 118 | 292 | 176 | 187 | 0/105 | 26/15 |
| 17-64 | 29 | 23 | 16 | 14 | 15 | 0/1 | 1/0 |
| >=65 (small N) | 12 | 8 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / fragmentation

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| (0,0.25] | 471 | 130 | 356 | 213 | 238 | 1/119 | 35/10 |
| (0.25,0.5] | 224 | 90 | 169 | 93 | 107 | 0/62 | 21/7 |
| (0.5,1] | 271 | 83 | 215 | 126 | 143 | 0/72 | 30/13 |
| >1 (small N) | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 |

### recognized_fenced_payload_diagnostic / type

| Bin | String N | Example N | Greedy correct | Match correct | Attention+match correct | vs greedy gain/loss | vs match gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| empty (small N) | 3 | 2 | 0 | 0 | 0 | 0/0 | 0/0 |
| URL | 20 | 9 | 16 | 14 | 7 | 0/9 | 0/7 |
| email | 24 | 15 | 21 | 9 | 11 | 0/10 | 2/0 |
| identifier | 436 | 121 | 363 | 186 | 218 | 1/146 | 48/16 |
| natural_language_proxy | 214 | 83 | 133 | 114 | 116 | 0/17 | 7/5 |
| other_string | 272 | 98 | 207 | 109 | 136 | 0/71 | 29/2 |

전체 5-arm correct/N, both correct/wrong, net change/N와 hash는 [small JSON](argument-subgroups.json)에 있다. Fenced-payload 진단은 primary strict JSON 점수를 대체하지 않는다. 원시 문자열과 개인정보는 이 요약에 포함하지 않았다.
