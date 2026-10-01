# Exploratory argument subgroup analysis plan
Recorded by the supervisor while the fixed pilot test is running, before inspecting test correctness or subgroup outcomes. This is a post-hoc analysis request, not a preregistered confirmatory endpoint.

## Scope and frozen run
Do not change running code, selection.json, threshold, model, data, test order, or generation configuration. Complete and preserve the current run. Use only existing raw predictions and cached tokenizer on CPU; no new GPU evaluation, synthetic long-string transformations, new datasets, or tuning.

The xLAM dev40 / Hermes test160 sample is an implementation pilot drawn from the recorded upstream development pools. It is not an official full benchmark result and does not use an official benchmark evaluator. Handcrafted cases are unit/smoke diagnostics only. Update README and final report accordingly. Preserve source dataset IDs, original split and subset-selection provenance. The parent is researching official public benchmark/split/evaluator/licensing separately; do not launch that evaluation.

## Scientific interpretation
Forced attention+match may differ from greedy; accuracy gains on long or heavily segmented arguments are the hypothesis. Full token/EOS equality gates apply to verifiedPLD and the attention-only control, not forced-copy arms. Keep accuracy and total latency conclusions separate. Do not require speed to acknowledge an accuracy gain, and do not imply an accuracy gain when unsupported.

## Fixed descriptive features
Compute features from every gold argument string leaf before comparison with predictions, using original unescaped strings and the evaluated checkpoint tokenizer with add_special_tokens=False. Count Unicode code points for characters; report UTF-8 bytes additionally if useful.
- Token-length bins: 0, 1–4, 5–16, 17–64, >=65 tokens.
- Fragmentation ratio: token_count / character_count. Empty strings form a separate bin; remaining bins: (0,0.25], (0.25,0.5], (0.5,1], >1. Boundary values belong to the lower interval.
- Exclusive type classification in this priority order: empty; URL (anchored http:// or https:// or www., no whitespace); email (anchored [^\\s@]+@[^\\s@]+\\.[^\\s@]+); identifier (ASCII [A-Za-z0-9_.:/#@+-]+, no whitespace, contains at least one digit or one of _.:/#@+-); natural_language_proxy (contains whitespace, excluding prior types); other_string.
- The natural-language label is a surface-form proxy, not a semantic detector. Single words without identifier markers remain other_string. Non-string leaves are counted separately and excluded from string-feature denominators.
These definitions are fixed for descriptive analysis; do not pick bins/types after observing successes or failures.

## Tables and denominators
Report dev and test separately; the test table is exploratory, not a claim about the full public benchmark. Include all prespecified bins, even N=0. Flag small N descriptively, without hiding it.
For each grouping report string-leaf N, example N, per-arm correct/N argument EM, and paired attention+match vs greedy and vs match-only: wrong→correct, correct→wrong, both correct, both wrong, plus net change/N. Use the same call/argument matching and strict JSON rules as the main scorer; document duplicate-call ambiguity. Invalid or missing output must not disappear from denominators.
If extracting a recognized fenced JSON payload, show this only as a separate, clearly labelled diagnostic using the same groups and raw predictions. Never replace primary strict-JSON scores.
Record feature-plan hash, raw prediction hashes, model/tokenizer identity, subgroup script hash, and exact CPU reproduction command. No source raw strings or personal details in committed summaries.

## Completion
Read this document after the current pilot finishes. Add the CPU analysis and small tables, verify counts and raw-to-summary lineage, update README/result caveats, and commit/push reviewed source and aggregates. Do not launch another experiment.
