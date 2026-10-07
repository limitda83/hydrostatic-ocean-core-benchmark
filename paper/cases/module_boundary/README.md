# module_boundary

**뒷받침하는 주장:** paper/TARGET.md C6
**결과 문서:** docs/38_module_boundary.md
**출처 실험:** E17_module_boundary
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

폐쇄의 실제 발자국(276 값/열)을 호스트로 왕복시키는 비용은 fp32·400² 이상에서 두 카드 모두 폐쇄의 2.1~2.7배(하한). pinned 필수(pageable 은 5090 2.1×, H100 4.9×).

## 재생산

- `python3 tools/module_boundary_report.py --decompose paper/cases/physics_cost/data/*/decompose_*.csv --boundary paper/cases/module_boundary/data/*/*_boundary.csv`

## 파일 (3)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E17_module_boundary/gpgpu_boundary.csv` | `expr/E17_module_boundary/data/gpgpu_boundary.csv` | f74c500eefb5 |
| `E17_module_boundary/ktcloud_boundary.csv` | `expr/E17_module_boundary/data/ktcloud_boundary.csv` | 79ff5adb3860 |
| `E17_module_boundary/gpgpu_boundary_static.csv` | `expr/E17_module_boundary/data/gpgpu_boundary_static.csv` | 2a2516b51a87 |
