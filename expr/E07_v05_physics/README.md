# E07_v05_physics

**질문/내용:** spec v0.5 검증 스위트 V5-1…V5-6 (지형·연직좌표·개방경계·EOS)

**보고서:** docs/24_v05_physics_results.md

**노드:** 로컬(검증)

**상태:** **완료** (2026-09-14 재실행). V5-1…6 전부 PASS. 보존본이 N15 수정 **이전**(9/11 04:42) 것이어서 V5-1 이 실제로는 실패 중이었다 — N15 가 코리올리 제거를 바꿨는데 스위트를 다시 안 돌렸다(docs/90 **N35**). 이제 V5-1 은 항등 다리(theta)만 게이트하고 split-explicit 의 설계상 차이(6.3e-5)는 측정해 보고한다.

**재현 명령:**
- `python3 tools/verify_v05.py --nx 48 --nz 20 --steps 100 --json output/v05_verification.json`

**데이터 (`data/`):** `v05_verification_20260923.json`(**현재 코드의 결과, 원고용** — N44 뒤 재실행, V5-1…6 전부 PASS), `v05_verification_20260914.json`(9/14, 비교용), `v05_verification.json`(9/11 04:42, N15 이전 — 비교용)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
