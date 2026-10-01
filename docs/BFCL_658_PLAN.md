# Authorized next stage: BFCL two official categories
The user explicitly extended scope after the fixed implementation pilot. Complete/preserve that pilot first, then evaluate the entire simple_python (400) and live_simple (258) categories at upstream ShishirPatil/gorilla commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8. This is 658 cases, not all BFCL or all single_turn. No further categories/sweeps/training.

## Existing preparation
- Unmodified sparse upstream checkout: data/bfcl-upstream-6ea5797; exact HEAD verified.
- Official data/gold counts and source hashes: docs/BFCL_SOURCE_MANIFEST.json.
- Official registry explicitly includes Qwen/Qwen3-0.6B-FC -> QwenFCHandler, is_fc_model=True, underscore_to_dot=False.
- Official native handler: bfcl_eval/model_handler/local_inference/qwen_fc.py. Its prompt and parser expect JSON objects inside <tool_call> XML tags; do not use generic OSS prompting AST decoder or the pilot JSON-list prompt.
- Isolated CPU environment data/bfcl-cpu-env, upstream import dependencies installed only from PyPI. Logs/reports: runs/bfcl-preparation. Existing GPU env is untouched. No provider API requests, authentication or secrets were needed.
- Supervisor performed metadata/import preparation only; engine adapter, full CPU integration and BFCL GPU evaluation still need implementation by the same gpt-6.1-sol session.

## Contract
Preserve official questions, function docs, gold alternatives and AST evaluator without patching. Use the official QwenFCHandler prompt/parser where possible, retaining exact model-native tool output framing. All five arms share the exact same prompt, tokenizer/chat template, parser, stops and evaluator. Explicitly record any local-HF adapter behavior (including thinking/generation mode) before any BFCL predictions. Do not silently call the pilot's list JSON prompt an official native-FC evaluation.
Extend copy eligibility only to valid JSON argument string values within recognized native tool-call envelopes; keys, names, XML, reasoning and text remain ineligible. Test wrapper/offset/contextual token alignment and fallback on ambiguity. Retain original prefix-KV/new-position causal blocks, PLD off-by-one and rejected-KV crop tests.
Keep pilot generation sources immutable for historical hash checks; use separate BFCL adapter/engine modules or an archived content-hashed snapshot if shared code must evolve. Preserve all original pilot raw files, configuration, threshold and results.

Freeze the 658 source IDs and category denominators, shared configuration, dev-selected pilot threshold/head/margin, code/data/model hashes and seed before BFCL generation. Do not use BFCL gold in runtime candidate selection, gating or threshold adjustment. Separate runtime input from offline gold scoring.
Measure category input-token distributions using the actual final prompt and checkpoint context limit before launch. Never truncate/exclude secretly. Report any over-limit/rejected/failed/truncated cases in the full denominator.
Official AST normalizes strings (case and selected punctuation/whitespace). Report official AST accuracy separately from parsed raw-string argument EM with all accepted gold alternatives; JSON escape serialization differences vanish only through parsing, not string normalization. Keep category-level totals and explicit error statuses.
Before predictions, freeze source-copy eligibility/length/token-fragmentation/URL-email-ID strata using source/gold features, with fixed definitions and all denominators. Retain zero/low-N groups. No manufactured long examples or output-driven subset choice.

First finish CPU integration tests with the official evaluator and native parser. Recheck GPU0 UUID/idle/current processes and existing shared locks after the pilot releases it. Never overlap GPU controllers. Use the same original cached Qwen3-0.6B, BF16/SDPA, one GPU, one controlled pass of five arms. Equality gate only for verifiedPLD and attention-only; forced accuracy gains/losses are separate from speed.
Use a small safety smoke to validate the new adapter, not to tune accuracy or thresholds. If a correctness gate fails, preserve evidence and repair before full evaluation. A null pilot performance result alone is not a safety blocker for this separately authorized BFCL stage.
Track to completion, preserve raw predictions and aggregates, commit/push source/config/small tables and reproduce commands, verify private remote SHA. Report selection/accuracy and total speed separately in Korean. Report concrete time/resource/adapter blockers promptly. No API execution or LLM judges.

## Additional required pilot analysis
Complete docs/ARGUMENT_SUBGROUP_PLAN.md from the already generated pilot predictions on CPU. Clearly label the 40/160 xLAM/Hermes results as an implementation pilot, and handcrafted inputs as unit/smoke diagnostics. They are not official benchmark scores.

## Completion record
The historical preparation/implementation steps above completed with the same Sol session. The latest explicit user allocation instruction superseded GPU0-only rechecking: equivalent idle GPU7 was verified/locked and recorded. Full658 inference naturally completed, CPU official AST and separate strict/protocol validation passed. See results/FINAL_REVIEW.md; no category expansion or extra GPU run followed.
