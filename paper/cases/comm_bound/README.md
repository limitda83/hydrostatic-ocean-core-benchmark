# comm_bound

**뒷받침하는 주장:** paper/TARGET.md Appendix C (communication constants for the single-node-vs-single-card limitation)
**결과 문서:** docs/28_multigpu_communication.md
**출처 실험:** E11_multigpu_comm
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

2-rank halo/Allreduce/PCG-iteration/barotropic-substep costs on RTX 5090 (PCIe, one node) — the measured constants behind the statement that multi-device scaling is out of scope but bounded.

## 재생산

- `make -C libs/cuda comm`
- `mpirun -np 2 libs/cuda/build/comm_bench`

## 파일 (1)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E11_multigpu_comm/comm_bench_20260914.txt` | `expr/E11_multigpu_comm/data/comm_bench_20260914.txt` | 353968f62a9c |
