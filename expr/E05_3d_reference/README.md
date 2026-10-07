# E05_3d_reference

**질문/내용:** 3D 정역학 레퍼런스(v0.2) · 완전 코어(v0.4) 검증 V3D-1…5

**보고서:** docs/14_p2_3d_reference.md, docs/16_p2c_full_core.md, docs/40_verification_summary.md

**노드:** 로컬(검증만)

**상태:** done

**재현 명령:**
- `python3 main.py verify3d --case baroclinic_igw --sweep 16,32,64`

**데이터 (`data/`):** `verify3d_baroclinic_igw_<stamp>_{manifest,metrics}.json` — **2026-09-23 재실행분(N44 뒤, N35)이 현재 코드의 결과**(수렴차수 1.97, 2.15 — 9/14 와 동일)이고 9/10·9/14 분은 비교용이다. 로컬 검증 전용이므로 시간 수치는 보고하지 않는다(R7-1).; `gates_20260915/`(2026-09-15 네 노드 R2 게이트 로그 — 로컬 24/24 · geo85 24/24 · gpgpu 48/48 · ktcloud CUDA 12/12 · ktcloud OpenACC 재빌드 후 12/12, 구판 이진의 12 실패 로그 포함 — docs/40, docs/90 N42)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
