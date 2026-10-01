# AttnCopy initial pilot

서버9 GPU0, 원본 Qwen3-0.6B BF16/SDPA. 구현 에이전트 gpt-6.1-sol. seed42, dev40(xLAM), test160(Hermes). 비공식 구현파일럿이며 공식 벤치마크 점수가 아니다.
개별 함수 스키마와 정규화 요청 중복은 0건이다. 공개 모델의 원래 학습 오염과 잠재 언어 템플릿 중복은 확인할 수 없다.

## Fixed test (160 accuracy / 159 paired-latency examples)

| Arm | Arg EM | Offline call EM | JSON valid | Token/EOS identity | Descriptive total s | Paired greedy/arm speed | Triggers | Compatible precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.000 | 0.000 | 0.000 | 160/160 | 671.64 | 1.000 | 0 | NA |
| attention-only | 0.000 | 0.000 | 0.000 | 160/160 | 676.78 | 0.992 | 0 | NA |
| match-forced | 0.000 | 0.000 | 0.000 | 51/160 | 662.15 | 1.014 | 420 | 0.581 |
| attention+match-forced | 0.000 | 0.000 | 0.000 | 78/160 | 676.22 | 0.993 | 202 | 0.426 |
| verifiedPLD | 0.000 | 0.000 | 0.000 | 157/160 | 654.96 | 1.026 | 567 | 0.418 |

## Selection and interpretation

선택한 layer/head는 사전 고정 27/0, dev trigger 21개의 mass 25분위 threshold=0.00001335, margin=0.01이다. test 결과로 재선택하지 않았다.
초기 attention-only 비용은 4개/195토큰에서 2.80% 증가했다. 별도 warmup과 초기화 제외, CUDA 동기화, arm 순환 순서를 사용했다.
품질 판정: **NO-GO**. strict full-call EM에서 attention+match가 match-only보다 높은지로 판정했다.
전체 속도 판정: **결론 불가 / 확대 NO-GO**. verifiedPLD 동치 gate 실패로 lossless/가속 결론을 금지한다. 경계 제외 attention+match / match-only 시간비=1.0212는 중단 실행의 기술적 관측치다.
확대 판정: **NO-GO**. verifiedPLD 전체 토큰·EOS 동치는 dev40/40, test157/160이다. 세 실패의 원시 출력·cache 이벤트를 보존했다.

strict JSON/scorer는 Markdown fence를 거부한다. 별도 fenced-payload 진단은 primary 정확도를 대체하지 않는다. 호출 순서는 EM에서 무시하며 실제 도구 실행 성공률이 아니다.
Trigger precision은 각 arm의 실제 prefix에서 target-greedy와 맞는 전체 block 비율이다. Gold 지원은 문자열 prefix의 지원이며 완전한 인자 정답을 뜻하지 않는다. 중복 함수명/해석 불능은 unknown으로 남겼다.
총 시간에는 prompt tokenizer, prefill, proposal/host sync, attention, causal block/검증, 최종 decode가 포함된다. forced arm의 full-block logits 정확성 audit도 포함한다. 초기화·scorer·파일 IO는 제외한다.
세 segment의54/35/70개 완전 case를 각 segment 안에서 pairing했다. case55는 정확도/동치 분모에 유지하고 일반 paired latency에서 제외했다. 합산 시간은 중단 실행의 기술적 총계이며 연속 실행 성능을 뜻하지 않는다. 다운타임·모델 재로딩은 제외했다.
각 arm은 한 번만 측정했으므로 시스템 잡음/순서 효과의 반복측정 불확실성은 추정하지 않았다. 생성 길이가 다른 forced arm의 시간 차이는 알고리즘 효과와 길이 변화를 모두 포함한다.

## Evidence and reproduction

[Complete small metrics and artifact hashes](pilot-20261001.json), [reproduction](../docs/REPRODUCE.md). Raw predictions/logs/failed smoke remain ignored on server9 in `runs/pilot-20261001`. Source/model/config/code hashes link every row to the aggregates.
13개 작은검사(기존11+순서/최대매칭2), 실제 PLD 6개 smoke, 0/부분/전체 KV trim, attention row(eager 대비 최대 오차 0.00086683)를 검증했다. BatchEncoding 정렬검사 실패 및 첫 0-trigger smoke를 삭제하지 않았다.
