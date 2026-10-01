# Small results only
Commit small aggregates only after validation against ignored raw predictions.
Each table must identify the run, fixed split/model/config/code hashes, denominators, invalid/truncated rows, greedy identity gate and raw artifact paths/hashes.
Preserve negative outcomes and distinguish offline full-call exact match from actual tool execution.
No pilot results exist at the initial bootstrap commit.
