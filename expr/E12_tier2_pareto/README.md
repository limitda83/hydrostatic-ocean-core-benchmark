# E12_tier2_pareto

**질문/내용:** 2층 조각 S: 스킴×해법×CFL×케이스, 400²×30, CUDA, 오차 vs 소요시간 (파레토 평면)

**보고서:** docs/04_design_v2.md §3, docs/30 (예정)

**노드:** gpgpu (RTX 5090)

**상태:** **완료** (2026-09-12 06:14, 3 케이스 × 31 구성). 결과 문서 `docs/30_tier2_pareto.md`,
결과 요약은 그 문서에 있다.
주의: N14·N15 수정 이전 측정은 폐기(노드의 `output/old_N14/`), N18 발산 가드 이후 재개분 포함.

**재현 명령:**
- `BACKENDS_S=cuda nohup bash tools/run_tier2_gpgpu.sh`
- `python3 tools/tier2_collect.py output/collected/tier2_*.csv --out output/tier2_summary`
- `python3 tools/tier2_report.py`

**데이터 (`data/`):** tier2_gpgpu_S_current.csv

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** N14·N15·N16 — 14:13 이전 측정 폐기(노드 `output/old_N14/`)
