# Mid-implementation review (before pilot)
The supervisor created/pushed the bootstrap docs as [original private bootstrap revision]. Continue the current single agent; no second implementation job exists. Initial notes here are review observations, not measured failures.

- Preserve the actual fast greedy denominator: AttentionRow currently installs Python hooks on the model even while disabled. Benchmark truly unhooked SDPA greedy separately or quantify/disclose dormant-hook overhead before calling it original greedy.
- Engine currently hardcodes float32. Use the selected original model's normal inference dtype if exactness permits; otherwise record a pre-test rationale and clearly limit speed conclusions to FP32. Do not silently imply production BF16 speed.
- prompt() concatenates separately encoded pre/query/post. Unit-test equality to canonical full chat tokenization and exact decoded text/offset mapping for leading spaces, newline, quotes and Unicode. If boundary merging occurs, map full-prompt offsets correctly rather than assuming separately encoded token IDs are canonical.
- choose() currently calls float() per candidate on a GPU row. Prefer GPU candidate reductions/argmax and one small transfer where possible; record host-sync proposal cost.
- Independent synthetic verified_prefix tests must cover 0/partial/full acceptance, first and later mismatch, EOS, and rejected-cache crop; real greedy IDs/EOS equality remains the gate.
- User explicitly requested first Git commit/push, minimal environment, staged secret/data/log exclusion, dev/test provenance and raw-to-small-results correspondence. Bootstrap completes the first Git step only; keep final implementation/test/run completion claims separate.

Further read-only review of the first scorer/data draft:
- score(): enumerate(pred or []) is evaluated even when structure is false. A valid JSON scalar such as 123 raises TypeError; test scalars/null/dicts/strings so every failed model output stays in the denominator.
- gold_value_prefix_supported indexes gold by predicted call-list index. Call order is ignored by full_call_em, so align predicted-to-gold call identity first when evaluating per-trigger gold support; record uncertainty for duplicate same-name calls.
- Data grouping currently hashes the entire tool catalog. Different catalogs can share the same individual tool schema; audit overlapping normalized individual function/schema groups and normalized queries, not just identical complete lists. Document any unavoidable latent-template uncertainty without claiming exhaustive leakage freedom.
