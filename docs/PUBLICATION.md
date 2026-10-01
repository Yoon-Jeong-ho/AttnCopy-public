# Publication preparation and evidence

Publication revision `public-redaction-r1` changes operational paths, host/lock
configuration and private metadata only. It does not generate new predictions,
change thresholds, tune test data or alter result numbers. All earlier scientific
revisions and failure evidence remain distinguished in their original reports.

Both historical Git snapshots were sanitized and authors/committers assigned the
repository owner's public GitHub noreply identity. Internal account/host paths,
agent session identifiers, process identifiers and GPU UUIDs are omitted. Foreign
process snapshots are omitted; the selected GPU's idle state, hardware/clocks,
resource checks, execution times and timing-segment distinctions are retained.
The original entire Git database, a bundle, tracked snapshot and recovery manifest
were preserved in an access-restricted private backup. A separate restored bundle
passed Git fsck and matched every reachable original object. Private originals,
ignored raw predictions/data/checkpoints and execution logs were not deleted.

## Scientific hashes versus portable publication files

Published run/config/source/runtime SHA256 values identify the **original frozen
experiment artifacts**. Redacting a path changes a public file's bytes/hash even
when its scientific settings are identical. Those historical hashes must not be
asserted to equal current portable file hashes. No historical raw/scored digest or
scientific result was recomputed to disguise this difference. Independent
publication validation checks unchanged result tables, subgroups, selection,
case IDs, raw/scored hashes and all non-path configuration values. Current source
hashes belong to this publication revision, not to a new inference run.

## Configure an independently authorized reproduction

`ATTNCOPY_SOURCE_DIR` points to a read-only, appropriately licensed source cache;
its default is `data/source/RefCall`. `ATTNCOPY_MODEL_DIR` points to the pinned
original checkpoint; its default is `data/models/Qwen3-0.6B-c1899de`.
`ATTNCOPY_GPU_PYTHON` identifies an environment matching the recorded GPU versions.
GPU controllers require an explicit `ATTNCOPY_ALLOWED_HOST` and existing shared
locks. `ATTNCOPY_GPU_LOCK_DIR` / `ATTNCOPY_GPU_LOCK_HOST` optionally locate those
existing locks. Configuring a hostname or finding an idle GPU does not grant an
allocation; obtain authorization and use fresh run IDs/claims before a new run.

The public repository contains no raw predictions, gated data, model weights,
authentication files or private backups. Some integration tests and historical
analysis require these ignored artifacts and exact licensed caches. The previous
run is complete; users can review reports and code without launching anything.
Further training, categories, sweeps or GPU runs require a separate decision.

## Separate public export

The original development repository remains private because GitHub retained old
sensitive SHA views after its authorized redaction. No old Git objects, refs,
reflogs, remote configuration or private backup are copied into this public
repository. The owner separately authorized AttnCopy-public, created from only
the sanitized tracked files with one fresh root commit. Code, scientific result
tables and failure evidence are unchanged by this export; repository-facing
links and publication explanations are updated. Anonymous access and the new
root history are verified separately. The public-export manifest records every
exported source file hash. No further experiment or original-repository rewrite
is part of this publication step.

## Publication CPU validation

- portable-runtime-cpu: 23 CPU tests passed.
- official-integration-cpu: 4 CPU tests passed.
- strict-scoring-cpu: 3 CPU tests passed.
- original-recovery-hash-gate: 1 CPU tests passed.

The full-suite attempt in the GPU environment could not import isolated official dependencies and correctly rejected changed original hashes. Checks were then split between the existing GPU environment with CUDA hidden, the existing official CPU environment and the unchanged original checkout. The public configuration was not substituted into old sealed run lineage. See PUBLICATION_VALIDATION.json; no GPU inference was run for publication.
