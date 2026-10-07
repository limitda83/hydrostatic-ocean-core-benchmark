# E06_performance_matrix_v02

**질문/내용:** v0.2~0.3 코어의 스킴×백엔드×격자 성능 매트릭스, split-explicit, 해법·메모리·dt 축

**보고서:** docs/20_performance_matrix.md, docs/21_p2_3d_backends.md, docs/22_split_explicit.md, docs/23_solver_memory_axes.md

**노드:** gpgpu · geo85

**상태:** done (v0.2 코어 — 지형 없음, N14 이전 fb 는 압력경사 절반이었음: fb 행은 폐기)

**재현 명령:**
- `bash tools/bench3d_matrix.sh`
- `bash tools/scheme_hw3d.sh`
- `bash tools/solver_hw_experiment.sh`

**데이터 (`data/`):** 18개 — `scheme_hw_<stamp>_results.csv`·`solver_hw_<stamp>_results.csv` (스윕 결과), 같은 접두사의 `_c_metrics.json`(마지막 run 의 지표)과 `_last.log`. 모두 gpgpu 에서 회수했다.

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** N14 — v0.2/v0.4 3D fb 는 압력경사 절반; θ·split 행만 유효
