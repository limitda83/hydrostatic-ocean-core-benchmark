# E15_precision

**질문/내용:** 정밀도 축(axis G)을 전체 v0.6 코어로 확장 — 같은 고정스텝 문제를 fp64·fp32 로
같은 rtol(1e-6)에서 풀어 (a) 카드별 fp32 가속비, (b) 물리 벌금이 정밀도에 어떻게 의존하는지,
(c) fp32 가 32 GB 카드의 격자 상한을 바꾸는지 측정한다.

**보고서:** `docs/34_precision_axis.md` (발견: 소비자 GPU 의 물리 벌금은 fp64 벌금이며, 정밀도는 용량 레버)

**노드:** gpgpu (RTX 5090, fp64=fp32/48) · ktcloud (H100, fp64=fp32/2)

**상태:** 완료 (2026-09-13). 두 노드 각 48행.

**재현 명령:**
- `python3 tools/make_sp_cuda.py` → `libs/cuda/src/cfd_exp3d5_cuda_sp.cu` 생성
- `make -C libs/cuda all3d5-spcheck && make -C libs/cuda all3d5-sp` (spcheck = 생성파일 fp64 빌드, 원본과 비트 동일해야 함)
- `make -C libs/fortran all3d5-sp acc3d5-sp serial3d5-sp FC=<nvfortran|gfortran>`
- `GRIDS="400 1000 2000" BACKENDS="cuda cuda_sp openacc openacc_sp" bash tools/precision_axis.sh`

**데이터 (`data/`):** `gpgpu_rtol6.csv`·`ktcloud_rtol6.csv` — 두 노드의 2층 스윕 수집본에서 rtol6 태그 행만 뽑아 보존한 것(원 수집본은 output 폴더의 collected 디렉터리, git 에 없음)

**코드 리비전:** 4f62993 이후

**주의:** rtol 이 1e-6 이므로 E13/E14(1e-12)의 행과 같은 표에 섞지 않는다(R5). fp32 결과는
fp64 결과와 **열을 분리**해 보고한다(R9). 2000² CUDA fp64 는 32 GB 초과로 OOM — 빈칸이 아니라
"OOM" 으로 적는다(N24).
