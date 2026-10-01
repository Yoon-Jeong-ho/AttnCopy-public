# Initial bounded run plan
1. Confirm repository private ownership, host, authentication, source instructions/licenses, original evaluator and shared GPU allocation/locks.
2. Inspect cached model/environment. Prefer original Qwen3-0.6B instruction model if usable; record exact path/revision and template. Do not use trained RefCall reference-token checkpoints.
3. Freeze ~40 dev/~160 test real held-out tool calls with source/template grouping (adjust counts only for clean grouping), seed 42. Keep handcrafted boundary/duplicate-prefix fixtures separate.
4. Implement optimized greedy, attention-only greedy, prefix-match forced copy, attention+match forced copy, verified prompt lookup, sharing model/input/stops/scorer. Record backend differences.
5. Unit-test JSON states, first content-prefix, byte/character and contextual token alignment, whitespace/max length, quotes/escapes/Unicode/duplicates, causal block positions, PLD logit shift and rejected-KV trim.
6. Run CPU tests, then tiny model/GPU smoke. Measure attention-only overhead before copy pilot; retain all failed attempts. Gate verified PLD on full greedy token/EOS identity.
7. Select/freeze layer/head/threshold on dev only (bounded prespecified choice, no sweep); run fixed test once. Rotate arm order and CUDA-synchronize timing; disclose single-run uncertainty.
8. Write small result tables, complete raw-to-aggregate lineage and Korean go/no-go separating selection quality from total speed. Verify private commit/push and stop.

## Initial observations (historical, before execution)
The original development repository was created empty/private under the owner account. A separately authorized public file snapshot is published at https://github.com/Yoon-Jeong-ho/AttnCopy-public with fresh history.
Experiment host/account/SSH details withheld; existing owner authentication was verified.
Existing ChatGPT authentication was reused; older Codex CLI 0.153.4 was already installed. Private authentication-storage paths are omitted.
User explicitly requested implementation agent gpt-6.1-sol, distinct from evaluated model. Cached catalog did not yet list 6.1; confirm via actual supported CLI request, never substitute silently.
Initial selected GPU candidates were idle; busy or suspicious devices were excluded. User authorizes one currently idle allocated GPU; do not change other jobs or source lock files.
RefCall source: source/RefCall (read-only). Existing env recorded GPU environment, original model data/models/Qwen3-0.6B-c1899de, licenses/data eligibility not yet verified.

## Completed approved extension
The newer already installed CLI0.159.2 successfully used gpt-6.1-sol in the same
existing Sol session. Cached original Qwen3-0.6B/data licenses and runtime were verified.
The fixed40/160 implementation pilot completed over preserved interrupted/resumed
segments; its original PLD identity gate failed3 times and cannot support speed claims.
The user then approved complete official BFCL simple_python400+live_simple258.
Original pilot engine stayed unchanged; separate native adapter/serial-replay engine
passed preserved3-case regressions and all658 token/EOS gates. Fixed threshold/head/
margin/seed and data IDs stayed unchanged. GPU7 was genuinely idle and held existing
shared locks; no other job was changed. BFCL PID[private process identifiers withheld] naturally exited2026-10-01
00:47:53UTC. CPU official scores and independent recursive-type/protocol review passed.
See results/FINAL_REVIEW.md and BFCL_VALIDATION.json for results, limitations,
allIDs, hashes and separate revisions. The bounded experiment is complete; publication preparation is documented separately in docs/PUBLICATION.md.
No broader categories/sweep/training/extra GPU experiments are authorized.
