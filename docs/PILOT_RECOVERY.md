# Pilot interruption and authorized recovery
Supervisor SIGINT to exact Codex PID[private process identifiers withheld] also caused managed tool-session cleanup of pilot PID[private process identifiers withheld]. The pilot had a separate process group/session but was still managed by the CLI. This was a supervisor error, not a model/equality failure. Do not repeat agent interruption during GPU work.

## Preserved immutable evidence
Read-only snapshot: runs/pilot-recovery-snapshot-20260930T1620Z.
MANIFEST.json SHA256 5cfaf6986f34d7599b0da2819b1f4b20c545a3be0f37f8703246bc7f383749be.
24 files / 2219940 bytes; all original lineage code hashes unchanged. Dev200 rows (40*5), test272 rows (54*5 + 2 arms of case55). Preserve original raw files byte-for-byte; never append, truncate, delete or overwrite them.
GPU0 UUID [private GPU UUID withheld] was idle (1MiB/0%) and both original GPU0 source locks plus controller lock were available. Other GPU jobs were untouched. Recheck idle state, allocation and existing locks at actual launch.

## Resume rules
Implement a separate resume controller and output directory/file, importing the unchanged original Engine/safety/settings. Never edit pilot engine/safety/metrics/pilot controller or frozen source files for BFCL.
Before model load assert snapshot hashes, original code and freeze hashes, source/model/tokenizer identities, fixed ordered case IDs, exact original arm rotation, seed42, selected threshold/head/margin, BF16/SDPA, stop/cache/maxcopy/generation settings unchanged. Use original source locks/controller lock; a new resume claim and allocation/lineage record.
Reload original model and repeat the identical original warmup (fixture-long greedy max_new32); save new warmup separately. Validate old rows against case index/arm/order, no duplicates or malformed/truncated file line. Compute only missing (caseID,arm) pairs, retaining original arm order/rotation. No dev rerun or reselection. Newly computed rows belong only to the new segment file.
For each complete merged example, check verifiedPLD and attention-only full token/stop/EOS identity against greedy. Preserve any failure before raising. Final union must contain exactly40dev*5 +160test*5 with unique pairs and every original row unchanged.
Annotate every row with its source segment in the analysis layer, without rewriting historical rows. Case55 crosses the interruption boundary: exclude it from ordinary within-segment paired latency comparison and report that exclusion explicitly; it remains in all accuracy/identity denominators. Report segment1 54 complete cases, segment2 remaining complete cases, boundary separately, same warmup per segment, downtime/model reload excluded, and that summed times are interrupted-run descriptive totals. Do not claim an uninterrupted timing pass.
Keep interruption logs and both raw files with hashes, exact recovery command, code/selection verification, PID, GPU/lock evidence and final merged validation. Update analysis to ingest segments explicitly instead of requiring original raw test file to have800 rows.
After recovery is running, prepare BFCL only in separate modules on CPU. Launch BFCL GPU only after pilot natural completion/safety gates and resource recheck. No further interrupt of Codex/omx parents while GPU work runs.
