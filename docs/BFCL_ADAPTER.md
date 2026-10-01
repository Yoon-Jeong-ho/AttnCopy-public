# Native FC adapter and pre-generation decisions

The source is the unmodified upstream commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` in
`data/bfcl-upstream-6ea5797`. All question/function/gold/handler/AST file hashes
are verified against `BFCL_SOURCE_MANIFEST.json`. Original RefCall final files
remain sealed/unread. Root/data license is Apache-2.0; source attribution is
[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8).

CPU preparation invokes the registered `QwenFCHandler` pure methods, including
`_pre_query_processing_prompting`, `add_first_turn_message_prompting`, and
`_format_prompt`, without a client/server constructor or API call. Runtime
inputs contain the exact native prompt and original user-message source spans,
with no gold/strata fields. The response follows the official native cleaner
`_parse_query_response_prompting` before `decode_ast`. Official AST is imported
and called unchanged with Language.PYTHON and Qwen/Qwen3-0.6B-FC. Its registry
uses is_fc_model=True, underscore_to_dot=False.

Generation is local HF target argmax with one-token SDPA greedy as the unhooked
denominator. It replaces the upstream server/Completions backend and is disclosed
as an adapter evaluation. Native formatter ends at the assistant prefix; no
disable-thinking suffix is injected. The source OSS cap is reproduced as
min(4096, checkpoint context minus input tokens minus2). EOS IDs are the original
native generation union151643/151645. No undefined prompt truncation occurs:
all658 full canonical token sequences/decoded text/offsets were compared with
the checkpoint tokenizer, max394/819 input tokens versus context40960.

Source-copy eligibility requires a recognized `<tool_call>\n` envelope and a
partial JSON argument **object's string leaf**. Root argument arrays, keys,
names, XML, reasoning and unclear prefixes fall back. Unicode/whitespace,
quotes/escapes, special tokens and XML delimiters are guarded. The first value
content prefix maps to exact original character/UTF-8 byte positions and full
chat token indices. At most8 tokens/64 new characters are proposed.

The native proposer retains the actual old output IDs even if the model chose
a noncanonical token combination earlier. It accepts a new continuation only
when canonical tokenization has no token crossing the append character boundary
and decoding the unchanged old IDs plus proposed new IDs reproduces the exact
source suffix. Source token IDs/KV are not grafted. Tests cover prior token aliases
and ambiguous/root-array/wrapper fallbacks. This correction is independent of
BFCL model accuracy outcomes and fixed before generation.

Head27/0, threshold1.3354860129766166e-05 and margin.01 are inherited from the
pilot dev, without BFCL tuning. Source/gold length/fragment/type/presence strata
are frozen before generation; feature values use the first non-empty accepted
reference alternative, while scoring retains every accepted alternative. Empty
optional markers and undefined references are disclosed. Seven empty-alternative
parameters in two live cases remain in the full658-case AST denominator; they
were not relabelled or removed. The first failed preparation is preserved.

The verified arm has the separate [BF16 serial replay repair](BFCL_NUMERICAL_REPAIR.md).
Its parallel draft and every serial target replay cost count in generation time.
No acceleration of ordinary parallel PLD is claimed. Safety smoke must exercise
real proposals, compare native tokens/stops/EOS, reproduce all three preserved
pilot failures against original greedy, and measure real block/serial KV differences
before the single full658×5 evaluation. GPU index/UUID/name/clocks and shared
leases are recorded. Existing user authorization permits an equivalent genuinely
idle server9 GPU; other jobs are never co-occupied or stopped.

Primary official AST and exact parsed string-leaf EM are separate. AST string
normalization remains upstream behavior; raw-string scoring removes serialization
escaping only through JSON parsing and uses all accepted values. Scalar/invalid/
missing outputs, AST/parser failures and truncation stay in complete denominators.
No API functions or LLM judges are executed. Native formatting occurs during CPU
preparation; timed generation includes tokenization, matching/host sync, attention,
causal block/replay and final decode. CPU parser/AST time is recorded separately;
model initialization and disk IO are excluded.
