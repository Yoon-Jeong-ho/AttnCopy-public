# Pre-BFCL verification repair

The immutable BF16 pilot failed verifiedPLD full-token equality on test case90
after an initially accepted two-token block. Its original/failed predictions,
stop/EOS information, cache events and hashes remain intact. No old engine,
threshold, input, warmup, or prediction was replaced. The remaining fixed arms
continue in a fresh segment with the same original engine, recording failures;
the pilot cannot support a lossless or speed conclusion.

Preserved diagnostic evidence shows bit-exact old prefix KV but different new
KV after a two-token BF16 block versus sequential forwards (max key difference4.0,
value0.75,92706 differing elements). Recorded positions and crop lengths are
consistent, and installed SDPA causal-mask reconstruction checks prevent future
token attention. Actual same-prefix GPU logits and top2 margins were not saved.
A numerical cache pathway remains a hypothesis; a near-tie and the complete
causal explanation of all three failures have not been established.
See [preserved diagnostic review](../results/PLD_DIAGNOSTIC_REVIEW.json).

The separate BFCL verified arm performs the causal parallel draft forward and
provisional shifted-logit check, then crops **all provisional new KV** back to
the actual prefix. It recreates each emitted token's KV with the same one-token
forward and one-token logits as unhooked greedy, checking target argmax serially
and emitting the first target mismatch. The resulting cache follows the serial
greedy computation. EOS/budget handling mirrors greedy. This follows the serial computation and empirically passed preserved-case
and all658 token/EOS regressions; parallel and replay costs are included in timing.
It is explicitly labelled `verifiedPLD with BF16 serial cache replay`, rather
than claiming ordinary parallel PLD speed. Source KV is never grafted.

Head27/0, threshold0.000013354860129766166, margin0.01, source matching/bounds,
model/tokenizer, native input, BF16/SDPA and seed42 remain fixed. No BFCL model
predictions or accuracy outcomes selected this repair. The formal658 run can
start only after original pilot inference finishes naturally, preserved-case
regression and native token/EOS safety smoke pass, and the single GPU lease is
rechecked. Official handler, questions, functions, gold and AST stay unmodified.
