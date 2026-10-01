AttnCopy-public의 완료된 연구 파일럿과 고정 BFCL658 공개 기록을 검토하세요.
AGENTS.md, RUN_PLAN.md와 docs/IMPLEMENTATION_NOTES.md를 먼저 읽으세요.
구현 에이전트는 기존 Codex CLI의 gpt-6.1-sol이며 평가 모델은 원본 Qwen3-0.6B입니다.
results/FINAL_REVIEW.md와 results/BFCL_658.md에서 최종 공식 결과를 확인하세요.
공식 범위는 고정 upstream의 simple_python400+live_simple258 전체이며 전체BFCL이 아닙니다.
dev40/test160 xLAM/Hermes는 구현 파일럿, 수제 fixture는 unit/smoke 진단입니다.
다섯 arm은 동일 native prompt/parser를 사용하고 강제형 정확도와 검증형 동치를 분리합니다.
prefix KV만 유지하고 복사 토큰은 새 위치 causal forward; source KV를 붙이지 않습니다.
JSON 인자 문자열 경계·문자/byte/token 정렬을 검사하고 모호하면 fallback합니다.
head/threshold/seed는 고정했으며 test/BFCL에서 튜닝하지 않았습니다.
원시 데이터·예측·모델·로그는 서버 runs/data에 보존하며 Git에서는 제외합니다.
재현/CPU 검증은 docs/REPRODUCE.md를 따르고 완료된 controller를 중복 실행하지 마세요.
추가 GPU 실험·범위 확장·학습은 사용자 결정 전 실행하지 마세요.

공개 이력 정리와 재현 경로 설정은 docs/PUBLICATION.md를 확인하세요.
