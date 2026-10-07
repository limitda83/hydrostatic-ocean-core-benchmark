# pareto_plane

**뒷받침하는 주장:** paper/TARGET.md C1, C7
**결과 문서:** docs/30_tier2_pareto.md
**출처 실험:** E12_tier2_pareto, E13_tier2_ladder
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

허용오차가 스킴 승자를 바꾼다 — 10⁻¹ 에서 semi-음해/split 4~9배, 10⁻³ 에서 양해법; 거친 지형에서 다중격자 V-cycle 4→166 으로 PCG 에 진다 (RTX 5090 CUDA, 400²×30, 3 케이스 × 31 구성).

## 재생산

- `python3 tools/tier2_collect.py paper/cases/pareto_plane/data/*/*.csv --out output/tier2_summary`
- `python3 tools/tier2_report.py`

## 파일 (3)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E12_tier2_pareto/tier2_gpgpu_S_current.csv` | `expr/E12_tier2_pareto/data/tier2_gpgpu_S_current.csv` | 86a899ac662e |
| `E13_tier2_ladder/tier2_gpgpu_20260911-141315_localhost_3904676.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260911-141315_localhost_3904676.csv` | b960287f4ba9 |
| `E13_tier2_ladder/tier2_gpgpu_20260911-234805_localhost_2256997.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260911-234805_localhost_2256997.csv` | 3d9951f2bcde |
