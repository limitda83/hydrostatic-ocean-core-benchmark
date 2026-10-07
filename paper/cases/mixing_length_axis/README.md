# mixing_length_axis

**뒷받침하는 주장:** paper/TARGET.md C5
**결과 문서:** docs/36_mixing_length_axis.md
**출처 실험:** E16_mixing_length
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

같은 물리, 15배 적은 산술(재귀 혼합길이) 의 값어치가 정밀도·장치·스레드·프레임워크에 따라 0.93~2.72배 — 한 구성의 이득은 일반화되지 않는다.

## 재생산

- `GRIDS="400 1000 2000" BACKENDS="cuda cuda_sp" bash tools/mxl_axis.sh`
- `python3 tools/shared_node_min.py paper/cases/mixing_length_axis/data/*/ktcloud*.csv`

## 파일 (12)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E16_mixing_length/geo85.csv` | `expr/E16_mixing_length/data/geo85.csv` | 5caf9cd1b9ab |
| `E16_mixing_length/gpgpu.csv` | `expr/E16_mixing_length/data/gpgpu.csv` | e4a760cedf8b |
| `E16_mixing_length/gpgpu_frameworks.csv` | `expr/E16_mixing_length/data/gpgpu_frameworks.csv` | f5143560e857 |
| `E16_mixing_length/gpgpu_nz.csv` | `expr/E16_mixing_length/data/gpgpu_nz.csv` | d3f5cf4a880d |
| `E16_mixing_length/ktcloud.csv` | `expr/E16_mixing_length/data/ktcloud.csv` | 5efe64567c56 |
| `E16_mixing_length/ktcloud_remeasure.csv` | `expr/E16_mixing_length/data/ktcloud_remeasure.csv` | a0cbe5c01697 |
| `E16_mixing_length/ktcloud_remeasure2.csv` | `expr/E16_mixing_length/data/ktcloud_remeasure2.csv` | ab18d0a552b0 |
| `E16_mixing_length/mixing_length_checks.json` | `expr/E16_mixing_length/data/mixing_length_checks.json` | 6ab0e67e3af8 |
| `E16_mixing_length/mxl_cost.json` | `expr/E16_mixing_length/data/mxl_cost.json` | d66fd21aace9 |
| `E16_mixing_length/v06_verification_recursive.json` | `expr/E16_mixing_length/data/v06_verification_recursive.json` | bbb18b51e227 |
| `E16_mixing_length/v06_verification_recursive_20260914.json` | `expr/E16_mixing_length/data/v06_verification_recursive_20260914.json` | bbb18b51e227 |
| `E16_mixing_length/v06_verification_recursive_20260923.json` | `expr/E16_mixing_length/data/v06_verification_recursive_20260923.json` | bbb18b51e227 |
