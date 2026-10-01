# Reproduce the bounded pilot

Use an explicitly authorized server and an environment matching docs/ENVIRONMENT.md.
Configure private operational paths with docs/PUBLICATION.md before running commands.
Do not install/authenticate packages, train, change RefCall files/jobs, use its
trained reference checkpoints, or transfer sealed final data into this pilot.
Only a genuinely idle user-allocated GPU whose shared locks are available may
run. `scripts/pilot.py` checks host, processes, utilization and existing locks,
pins one UUID, and uses a per-stage exclusive claim plus one controller lock.
An existing claim intentionally refuses duplicate launch. Never delete claims
to replay a completed test. Preserve this run; reproduce in a separate authorized
checkout with fresh ignored artifacts, or change the run ID/output path before
freezing a new run. Merely finding an idle GPU does not create a new allocation.

## Sources and eligibility

- Model: [Qwen3-0.6B pinned revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
  [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE).
  Cached original weights, no AttnCopy training. BF16 comes from its original
  config, with SDPA in all measured arms. No claim about FP32 or other backends.
- Dev40: cached RefCall xLAM dev, source [Salesforce/APIGen xLAM 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/blob/26d14ebfe18b1f7b524bd39b404b50af5dc97866/README.md),
  CC BY 4.0, gated; reuse the already authorized local copy, no download/login.
- Test160: cached RefCall **Hermes dev** pool, source [NousResearch Hermes](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/blob/dae3e1d28cfbcf4b915c04ea1e072030529b4bda/README.md),
  Apache-2.0. The original RefCall Hermes test/BFCL final remain sealed/unread.
- RefCall `docs/DATA.md`, split/convert code, current plain-v5/pointer protocols,
  scorer and prompt definitions were inspected read-only and hashed. No RefCall
  repository LICENSE was found, so its modules were not copied/imported. The
  implementation and strict sorted-call/typed-leaf scorer here are independent.
- These are project-held-out development pools, not certified unseen data for
  Qwen's original training. Dataset sources, normalized full catalogs, individual
  function schemas and normalized exact queries are disjoint across this pilot's
  dev/test. Unidentified latent language templates may overlap. Fixtures are
  separate, and no test-driven model/head/threshold selection occurs.

## Commands

The following sequence is for a fresh authorized checkout/artifact directory.
The existing run already executed it and must not be overwritten.

```bash
cd /path/to/AttnCopy
PY="${ATTNCOPY_GPU_PYTHON:?Set the authorized GPU environment Python}"
mkdir -p runs/pilot-20261001
python3 scripts/prepare.py
python3 scripts/amend_review.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  "$PY" -m unittest discover -s tests -v
"$PY" -u scripts/pilot.py --stage overhead
"$PY" -u scripts/pilot.py --stage smoke
"$PY" -u scripts/pilot.py --stage pilot
"$PY" scripts/analyze.py
```

The original failed smoke used `--stage smoke`; the preserved successful retry
used `--stage smoke --attempt 2` after the documented fence-eligibility repair.
The current implementation requires only a successful smoke; it does not replay
historical failures. Failed attempts use separate claims, lineage, logs and raw
files. There is no automatic experiment sweep or duplicate controller.

## Definitions and lineage

Greedy is truly unhooked. Attention arms attach Q-normalization/attention hooks
only for their generation; they compute one selected head's last-query GPU row
from the actual prefix K. Candidate masses/argmax stay on GPU with one small
host transfer; host synchronization costs count in proposal/total time.

The first JSON string content prefix maps to source character/UTF-8 byte and
canonical full-chat token offsets. Output-context retokenization must preserve
the existing output IDs and exact decoded suffix; uncertain merges, Unicode,
escapes, quotes and whitespace fall back. At most 8 tokens/64 new characters
are proposed, only from the prompt query. Recognized JSON fences permit argument
parsing but do not count as valid strict JSON. Match-only rejects divergent
duplicate continuations; attention uses its frozen threshold and margin.

Source KV is never copied. A draft is forwarded causally at fresh output
positions after the actual prefix cache. PLD uses the saved prefix-next logit,
shifted block logits, contiguous acceptance and exact rejected-cache crop, then
forwards the target mismatch. Forced arms include a full-block logit audit for
their conditional compatibility/wrong-token metrics; that audit's cost remains
in their measured latency. This bounded prompt-only string method is an
adaptation, not a reused upstream PLD implementation.

The fixed head is 27/0. Dev attention+match uses permissive threshold0/margin.01
once. Its trigger mass 25th percentile becomes the test threshold (or 1 if no
events); there is no sweep or test tuning. All five arms share weights, canonical
prompt, dtype, EOS IDs and 512-token cap. Order rotates per example, warmups are
separate, CUDA timings synchronize, and initialization/scorer/disk IO are
excluded. Total includes prompt tokenization through final output decode.

`runs/pilot-20261001/frozen.json` preserves the initial preparation;
`frozen-v2.json` records pre-inference review corrections with unchanged IDs.
Per-stage lineage contains code/config/frozen/runtime hashes, commands, PID and
session. `selection.json` is saved before test. Raw JSONL contains tokens/EOS,
text, costs, scalar-denominator scores, triggers, source offsets and cache traces.
The analysis asserts all 200×5 predictions, input IDs, score recomputation,
byte/character spans, block positions and trim lengths. The small result JSON
contains all raw hashes plus source/model/config/code hashes; raw examples,
weights, logs and failures remain ignored on server9.

Strict call EM ignores call order and distinguishes types; malformed/scalar
outputs stay in all denominators. Gold-support diagnostics identify unique call
names across output order; duplicate names remain unknown. Exactly delimited
fenced-payload diagnostics are separately labelled and never repair primary
accuracy. No tools are executed. One model/seed and one timing pass support a
bounded go/no-go decision, not a general superiority or robust speed claim.

The historical raw cached argument scorer uses the older RefCall sequential
same-name match. `attncopy/scoring_review.py` independently implements the current
definition (maximum strictly correct leaf assignment, then exact calls/key sets)
for the final table; it passes exhaustive-oracle and permutation tests. Both
argument scores are retained, with identical strict call/JSON definitions. This
postprocessing repair does not alter any model predictions or selected threshold.

## Actual interruption recovery and official native stage

The completed pilot has200dev/800test arm rows over three immutable test files:
snapshot272, first resume178, final segment350. Case55 stays in accuracy but is
excluded from within-segment paired timing (54/35/70 complete cases). Three
verified mismatches block the original implementation's speed conclusions.
Exact recovery/CPU commands used with the supplied read-only snapshot were:

```bash
PY="${ATTNCOPY_GPU_PYTHON:?Set the authorized GPU environment Python}"
"$PY" scripts/resume_pilot.py --dry-run
"$PY" -u scripts/resume_pilot.py --gpu 0
# Above preserved178 new rows and exited on identity failure; never append to it.
"$PY" -u scripts/finish_pilot.py
CUDA_VISIBLE_DEVICES='' "$PY" scripts/analyze.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  data/bfcl-cpu-env/bin/python scripts/argument_subgroups.py
```

The snapshot manifest/hash is part of this specific recovery record; it is not
an instruction to fabricate a snapshot or overwrite an existing run. A newly
authorized reproduction requires new run IDs/paths, fresh claims and equivalent
source identities. Analysis of the existing recorded artifacts is CPU-only.

The user extended scope to official simple_python400+live_simple258, with native
Qwen FC framing/response cleaning/AST. See [scope](BFCL_658_PLAN.md),
[adapter choices](BFCL_ADAPTER.md), and [numerical repair](BFCL_NUMERICAL_REPAIR.md).
The original GPU environment is unchanged; provided CPU env imports upstream
handler/AST and tokenizers. Preparation and checks were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  data/bfcl-cpu-env/bin/python scripts/prepare_bfcl.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  data/bfcl-cpu-env/bin/python -m unittest discover -s tests -p test_bfcl_official.py
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -p test_bfcl_tokenization.py
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -p test_bfcl_replay.py
python3 -m unittest discover -s tests -p test_bfcl_safety.py
```

All original658 IDs, raw source/gold checksums, canonical inputs, context and
gold strata are frozen in `runs/bfcl-658/frozen.json`. The first preparation
failure (empty gold alternatives) and pre-model resource/guard failures remain
separate artifacts. The observed GPU0/1 foreign allocations were preserved.
The reviewed native safety execution uses the available equivalent GPU7:

```bash
"$PY" -u scripts/bfcl_run.py --stage smoke --attempt 5 --gpu 7
# Execute only after safety-passed.json exists and the GPU/leases are rechecked:
"$PY" -u scripts/bfcl_run.py --stage evaluate --gpu 7
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  data/bfcl-cpu-env/bin/python scripts/analyze_bfcl.py
```

These commands document this run, not permission to duplicate an active or
completed controller. Smoke/evaluation source hashes, model/tokenizer hashes,
fixed threshold, same GPU, same warmup and every case/arm rotation are checked.
The main pass contains exactly3290 predictions. CPU scoring rechecks IDs,
denominators, byte spans, provisional trim/replay lengths and token/EOS identity,
then writes a separate ignored scored sidecar and small aggregate/strata tables.
Until the corresponding complete markers exist, submitted/running stages are
not completed results. All raw files and failures remain on server9.

## Completed official run and separate CPU review r1
BFCL GPU generation naturally finished at2026-10-01T00:47:53UTC,3290 rows.
The coordinator observed the completed shared stage and performed official CPU scoring
without signalling the existing Sol session. Exclusive scoring-directory creation prevented
a duplicate writer. Original generation code/config and legacy scored sidecar are preserved.
A separate CPU-only recursive-type/protocol review corrects shallow nested equality:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  data/bfcl-cpu-env/bin/python -m unittest discover -s tests -p test_bfcl_scoring_review.py -v
# One-time command used after legacy official scoring; it refuses existing review output:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  data/bfcl-cpu-env/bin/python scripts/validate_bfcl_results.py
```

Logs: runs/bfcl-658/coordinator-validation-20261001/{official-scoring.log,review-tests.log,independent-review.log}.
Reviewed sidecar: runs/bfcl-658/postprocess-review-r1/reviewed-predictions.jsonl.
Small results preserve both revisions: bfcl-658.json (original scorer) and
BFCL_VALIDATION.json (recursive strict types, unknown gold, native JSON protocol).
No post-run GPU inference, model/threshold change, new dataset or subset selection occurred.
The earlier instruction against installing packages applies to the existing GPU runtime;
the user separately approved the isolated official CPU dependency installation.
See FINAL_REVIEW.md for all accuracy/costs/denominators and limits. Do not rerun a
completed controller or delete its claim to reproduce; use fresh authorized artifact paths.

Publication revision public-redaction-r1 changes path/host configuration and historical Git identities, not scientific outputs. Recorded frozen hashes refer to privately archived original files, not these portable redacted files; see PUBLICATION.md.
