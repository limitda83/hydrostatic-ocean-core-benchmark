# paper/ — 원고 작업 폴더

**원고는 이 폴더 안의 것만 쓴다.** `expr/` 은 실험 대장(모든 측정, 폐기·격리 포함), `docs/` 는 결과 문서,
`docs/90` 은 내부 작업 로그다. 그중 **원고에 실제로 쓸 케이스만** `paper/cases/<case>/` 로 승격한다.

## 승격 조건 (전부 만족해야 함)
1. `tools/checklist.py` 의 해당 실험 행이 13항목 전부 ✓ 또는 —
2. 그 케이스가 뒷받침하는 **주장이 `paper/TARGET.md` 의 빈 영역 안에** 있다
3. 데이터는 `expr/E##/data/` 에서 **복사**한다(격리 파일 `*_SUPERSEDED / *_RESULT / *_UNGATED / *_IRREPRODUCIBLE` 은 제외) — 이 폴더만으로 표가 재생산되어야 한다
4. `README.md` 에 출처 실험, 파일, 재생산 명령, 뒷받침하는 주장 번호를 적는다

## 구조
- `TARGET.md` — 유사 논문 지도, 빈 영역, 작성 방향, 투고 저널
- `cases/` — 승격된 케이스
- `../ref/` — 외부 문헌 (PDF 는 `ref/pdf/`)

## 승격 도구
`python3 tools/promote_cases.py` 가 `PLAN` 에 적힌 케이스를 `expr/E##/data/` 에서 복사하고 README 를 쓴다
(격리 태그 파일은 이름만으로 거부). `--check` 는 승격본이 출처와 바이트 일치하는지, 격리 파일이
섞이지 않았는지 검사한다(일부러 넣으면 실패함 — R4-1 확인 2026-09-16).

## 승격된 케이스 (2026-09-16)
| 케이스 | 출처 | 파일 | 주장(TARGET.md) |
|---|---|---|---|
| `pareto_plane` | E12 | 1 | C1, C7 |
| `hardware_ladder` | E13 + E12 조각 S + E16 | 83 | C2, C3 |
| `physics_cost` | E14 | 7 | C2, C6 |
| `precision_axis` | E15 + E18 | 6 | C4 |
| `mixing_length_axis` | E16 | 11 | C5 |
| `module_boundary` | E17 | 3 | C6 |
| `kernel_matrix` | E08 (r7 재측정본) + E03 | 8 | C7 |
| `verification` | E05 · E07 · E14/E16 검증 JSON · 게이트 로그 | 39 | M |

`hardware_ladder` 만으로 docs/32 의 8표가 재생산되어 `tools/audit_crossdevice.py` 0 실패(2026-09-16 확인);
`precision_axis`+`physics_cost` 로 `mixed_report.py`, `module_boundary`+`physics_cost` 로 `module_boundary_report.py` 가 그대로 돈다.

## 투고 (2026-09-16)
대상 저널 두 판: **Computers & Geosciences** → `paper/cageo/`, **Computers & Fluids** → `paper/caf/` (각각 `manuscript/main.tex`, 그림 스크립트, `NUMBERS.md`, `REVIEW.md`, `SUBMISSION_CHECKLIST.md`). 데이터는 둘 다 `paper/cases/`.
