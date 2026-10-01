# Recorded environment
Experiments used one authorized server; its private hostname/account are omitted. Experiments ran only on server9. The Mac was SSH transport.
The existing GPU environment was not installed into or modified; an isolated CPU evaluator environment was installed from official PyPI. No authentication configuration or persistent credentials changed.
- GPU Python: ${ATTNCOPY_GPU_PYTHON}, Python3.11.15/GCC14.3.
- torch2.11.0+cu128 (git70d99e998b4955e0049d13a98d77ae1b14db1f45), transformers5.3.0.dev0, tokenizers0.22.2, safetensors0.7.0, numpy2.4.3, pytest9.0.2.
- Qwen runtime SHA256 fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab; portable module labels and all CPU package versions in [JSON](ENVIRONMENT.json).
- Original cached Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca, BF16/SDPA, seed42. Model/tokenizer hashes are attached to pilot/BFCL result JSON and immutable run lineage. No training or reference checkpoint.
- Implementation agent: existing Codex CLI0.159.2, actual slug gpt-6.1-sol, existing ChatGPT authentication. The older installed CLI0.153.4 rejected this slug before any GPU inference; the already installed newer managed CLI supported it. No substitute model.
- GPU RTX A6000/49140MiB, NVIDIA driver580.159.04. Pilot GPU0 interrupted/reloaded segments and continuous official BFCL GPU7 are reported separately.
- CPU evaluator: data/bfcl-cpu-env, Python3.11.15, numpy1.26.4/tokenizers0.23.2/pandas3.0.6/tree-sitter0.21.3. [Full version inventory](bfcl-cpu-versions.txt). The imported upstream SDK dependencies were not used for API requests.
These version strings and source hashes are a provenance record; a development Transformers build is not guaranteed to rebuild bit-identically from versions alone. Use the existing installation for CPU review of preserved artifacts. Generation sources/config/frozen hashes are checked against the actual execution lineage.
