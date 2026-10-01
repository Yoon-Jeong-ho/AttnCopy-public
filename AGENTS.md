# AttnCopy pilot contract
- Use only an explicitly authorized experiment server; the client is SSH transport only. Configure operational paths as described in docs/PUBLICATION.md.
- Scope: preparation, unit tests, smoke and one ~200-example pilot, one existing instruction/tool model, one genuinely idle allocated GPU. No training, broad sweeps, paid experiment APIs or other servers.
- Preserve RefCall repositories/data/jobs. Read relevant .agents/skills and source licenses/evaluation definitions before reuse. Record source hashes; never edit source experiments.
- Reuse prefix KV only. Never append source KV; copied tokens require a causal block forward at new output positions.
- Copy only inside JSON argument string values. Align first content-prefix to original character and UTF-8 byte offsets; verify contextual tokenization and exact text, not token IDs alone.
- Stop copying before whitespace initially; max block 8 tokens/64 characters. Quotes, escapes, Unicode, duplicate prefixes and uncertain boundaries require tested handling or fallback.
- Measure attention-only overhead first. Prefer selected layer/head last-query row computed on GPU. Keep optimized greedy separate from instrumented/backend-degraded greedy.
- Freeze source/template-disjoint dev/test before selection; fixtures separate. No test tuning. Verified PLD must match greedy full token sequence including EOS, or no speed conclusion.
- Preserve code/config/data/model hashes, seed, commands, logs, raw predictions, aggregates and failures. Git excludes raw data/checkpoints/secrets/large artifacts; commit/push only source, small results and reproduction instructions.
- One controller/writer owns execution. Use gpt-6.1-sol for the Codex implementation agent; record actual backend/model/session. Do not install/authenticate tools or change persistent credentials.

- Explicit user extension completed: entire BFCL simple_python400+live_simple258 after preserving the pilot, same model/settings and any genuinely idle equivalent server9 GPU. Separate isolated official CPU dependencies were authorized; the existing GPU environment remains unchanged. Further GPU runs/sweeps/categories require a new user decision.
