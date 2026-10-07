# precision_axis

**뒷받침하는 주장:** paper/TARGET.md C4
**결과 문서:** docs/34_precision_axis.md
**출처 실험:** E15_precision, E18_mixed_precision
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

소비자 GPU 의 물리 벌금은 fp64 벌금. fp64 코어 + fp32 폐쇄(mixed) 는 5090 에서 스텝 0.51~0.58×, H100 에서 0.89~0.91×, fp64 와 반복수 동일. 일괄 fp32 는 다중격자에서 수렴하지 않는다(그 행은 RESULT 로 격리, 여기 없음).

## 재생산

- `GRIDS="400 1000 2000" BACKENDS="cuda cuda_sp openacc openacc_sp" bash tools/precision_axis.sh`
- `python3 tools/make_mixed_cuda.py && (cd libs/cuda && make all3d5-mixed all3d5-mixedcheck)`
- `python3 tools/mixed_report.py --decompose paper/cases/physics_cost/data/E14_v06_representative/decompose_gpgpu_fb_20260915.csv paper/cases/precision_axis/data/E18_mixed_precision/decompose_gpgpu_mixed_fb_20260915.csv --ladder paper/cases/precision_axis/data/E18_mixed_precision/ladder_gpgpu_mixed_20260915.csv`

## 파일 (6)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E15_precision/gpgpu_rtol6.csv` | `expr/E15_precision/data/gpgpu_rtol6.csv` | 599e3358ae12 |
| `E15_precision/ktcloud_rtol6.csv` | `expr/E15_precision/data/ktcloud_rtol6.csv` | 3bb6b2dd30f1 |
| `E18_mixed_precision/decompose_gpgpu_mixed_fb_20260915.csv` | `expr/E18_mixed_precision/data/decompose_gpgpu_mixed_fb_20260915.csv` | 9f279238e166 |
| `E18_mixed_precision/decompose_ktcloud_mixed_fb_20260915.csv` | `expr/E18_mixed_precision/data/decompose_ktcloud_mixed_fb_20260915.csv` | 88c96d3a43fd |
| `E18_mixed_precision/ladder_gpgpu_mixed_20260915.csv` | `expr/E18_mixed_precision/data/ladder_gpgpu_mixed_20260915.csv` | 12dcd17ad6a5 |
| `E18_mixed_precision/ladder_ktcloud_mixed_20260915.csv` | `expr/E18_mixed_precision/data/ladder_ktcloud_mixed_20260915.csv` | ddf5c259f6d1 |
