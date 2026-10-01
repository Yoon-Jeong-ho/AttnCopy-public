# AttnCopy: a bounded negative experiment record

Public research snapshot: [Yoon-Jeong-ho/AttnCopy-public](https://github.com/Yoon-Jeong-ho/AttnCopy-public). This repository starts with a fresh publication history; private operational originals are preserved separately.

AttnCopy tests training-free contiguous copying from input text into tool-call argument strings, using attention plus text matching to select a source span. In this bounded Qwen3-0.6B experiment, the tested method did not improve accuracy or latency relative to optimized greedy decoding. The hypothesis about long or heavily tokenized arguments remains unverified.

## Evaluation scope

One original Qwen3-0.6B checkpoint, BF16/SDPA, seed42. Five arms share the same model-native prompt, parser, input and stopping settings. Layer/head and copy thresholds were fixed using the earlier implementation dev set, without test/BFCL tuning.

The formal data scope is the complete BFCL simple_python400 and live_simple258 categories (658 examples/3290 arm predictions), pinned to upstream6ea57973c7a6097fd7c5915698c54c17c5b1b6c8. This is not all BFCL or a leaderboard submission. It uses official QwenFCHandler formatting/response cleaning and AST evaluation through a local Hugging Face greedy adapter. No tool execution, API inference or LLM judge is involved.

| Arm | Official AST correct/658 | Exact parsed string arguments/1067 | Exact string leaves/1340 | Mean total generation seconds |
|---|---:|---:|---:|---:|
| Optimized greedy | 506 | 940 | 1056 | 6.073 |
| Attention-only | 506 | 940 | 1056 | 6.155 |
| Match forced copy | 385 | 731 | 818 | 6.100 |
| Attention+match forced copy | 418 | 784 | 874 | 6.189 |
| Verified control with serial KV replay | 506 | 940 | 1056 | 6.119 |

Attention+match recovered33 AST-correct calls and53 exact string arguments relative to match-only, but remained below greedy by88 calls and156 string arguments. It was1.45% slower than match-only; attention-only overhead was1.34%. Thus the current setting has no combined accuracy/latency benefit and does not justify expanding this experiment. These are single-pass observations from one model/head/seed, not a general impossibility result.

## Method and verification limits

Forced copy may change greedy output; its accuracy changes and errors are measured. The verified control is held to full token/stop/EOS equality. Copying is restricted to safe JSON argument string values, with character/byte/contextual-token alignment, whitespace stopping and at most8 tokens/64 new characters. Source KV is never grafted; new tokens use causal forwards at output positions.

The earlier xLAM40/Hermes160 run is an unofficial implementation pilot. Its original parallel verifiedPLD had three full-token mismatches and cannot support a lossless or speed claim. Old prefix KV was bit-exact in a preserved diagnostic, while new block/sequential BF16 KV differed. Recorded positions/trim and CPU causal-mask checks were consistent, but actual same-prefix GPU logits/top2 margins were not saved; the complete numerical cause and a near-tie explanation remain unproved.

The separate formal BFCL verified control discards provisional block KV and recreates emitted KV by serial greedy forwards. It passed658/658 empirical token/stop/EOS gates and the preserved failing-case regressions. Its parallel verification and serial replay costs are included. This is not ordinary parallel PLD acceleration or a universal losslessness proof.

## Long arguments and data limitations

Gold/source string features were fixed before outputs. There are only25 string leaves of17–64 tokens and one leaf of at least65 tokens; the token/character>1 group has zero leaves. URL/email groups contain13/4 leaves. Exploratory groups retain all denominators and zero/low-N bins. They do not validate the long/heavily fragmented argument hypothesis.

Official AST string normalization is reported separately from exact parsed strings; only escaping differences vanish through parsing. Malformed/no-call/truncated predictions remain in full category denominators. Seven empty gold-alternative parameters in two examples are explicitly undefined for raw exact-value scoring. A separate CPU recursive-type scoring revision preserves the legacy sidecar and leaves AST/string metrics unchanged.

Model revision, dataset sources/licenses, configuration, scientific hashes and small aggregate results should be retained in the public record. Raw datasets/checkpoints/predictions and private operational lineage stay outside Git. Publication preparation changes no experimental outcomes and requires no new GPU run.

## Reports and reproduction

[Start here](START_HERE.md), [final review](results/FINAL_REVIEW.md), [official tables](results/BFCL_658.md), [reproduction](docs/REPRODUCE.md), [publication provenance](docs/PUBLICATION.md).

## Related work
- [LargePiG](https://arxiv.org/abs/2410.11366) derives a pointer distribution from attention and copy probability from differences between upper/final layer vocabulary distributions. AttnCopy instead selects bounded contiguous source spans within protected JSON string values.
- [LLMA](https://arxiv.org/abs/2304.04487) drafts reference tokens and verifies them in parallel for greedy-equivalent generation. Here forced-copy quality changes and a separate verified control are evaluated independently.
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) drafts prompt continuations by n-gram matching without a draft model. Our independent bounded string-lookup adaptation adds byte/character/contextual-token boundaries and an attention selection arm; upstream implementation was not copied.
- [SwitchSD](https://arxiv.org/abs/2609.20186) trains probes for copy intent and switches between neural drafting and context copying. This pilot has neither probe/drafter training nor added reference tokens.

[Contracts](AGENTS.md), [plan](RUN_PLAN.md), [implementation notes](docs/IMPLEMENTATION_NOTES.md),
[authorized official scope](docs/BFCL_658_PLAN.md), [repair and limits](docs/BFCL_NUMERICAL_REPAIR.md),
[environment](docs/ENVIRONMENT.md), [source/license manifest](docs/BFCL_SOURCE_MANIFEST.json).
Raw datasets/predictions/checkpoints/secrets/large logs stay ignored on server9.
Git contains source/config/tests/small tables and reproduction. No repository license
was chosen without the user; upstream reuse conditions are recorded separately.
