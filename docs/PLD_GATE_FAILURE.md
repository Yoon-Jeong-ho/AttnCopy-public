# VerifiedPLD gate failure: preserved evidence
Observed2026-09-30T16:58UTC. Resume PID[private process identifiers withheld] naturally exited on its identity assertion, not supervisor interruption.
Original test raw272 rows and resume segment178 rows are intact:90 complete test cases (450 arm rows). Dev40*5 also intact. CaseID hermes_86c8f17a-f149-4506-8c5a-472709263c55, example_index89.
Greedy/attention-only125 tokens; verifiedPLD130 tokens; same EOS151645. First token difference at output index106. Forced arms are allowed to differ; the failed identity comparison is specifically verifiedPLD vs greedy.
The identity-failure.json arm field says attention+match-forced because that was the last generated row when the merged case became complete. The actual exception names verifiedPLD. Preserve the original failure file unchanged and clarify this in analysis.

First verified proposal: output_start37, draft[15,15], block_length2, accepted2/rejected0, cache683->685, complete block compatibility passed.
Second proposal: output_start120, after the first mismatch; also accepted2. Thus the first mismatch was69 tokens after the first accepted block, not directly inside its verification. A block/serial BF16 cache or logit numerical difference is a hypothesis, not yet established. Do not label it an off-by-one bug without evidence.

No verified acceleration conclusion is permitted for the failed implementation. Do not launch official full BFCL GPU evaluation before investigating/repairing this correctness issue and passing bounded model checks. Preserve frozen original engine/safety/config/data/threshold/model hashes and all generated rows; any repair must be a separate, versioned implementation/controller with explicit lineage. No test threshold tuning or new datasets. Investigate on the already failing case as a correctness diagnostic, with idle GPU/lock checks and one controller only. No duplicated diagnostic workload from the supervisor.

Latest user clarification permits any genuinely idle allocated equivalent server9 GPU, with model/UUID/clocks/segment disclosure. Current Sol session remains[private process identifier withheld]; do not interrupt it or any GPU parents for steering. Root will review the diagnostic evidence and final merged-count/timing treatment.
