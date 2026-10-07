# E16_mixing_length

**질문/내용:** `docs/35` 커널 프로파일이 TKE 폐쇄(`k_closure`)를 한 스텝의 단일 최대 커널
(H100 1000²×30 에서 21.8 %)로, 그리고 그 원인을 병렬성 부족이 아니라 **열당 O(nz²) 산술**로
특정했다. 스펙 S11.2 에 혼합길이 축 `mxl = integral | recursive` 를 열어 같은 물리를 O(nz)
두 스윕으로 구현하고(NEMO `nn_mxl=2`, Blanke & Delecluse 1993), 산술 ~15배 절감이 **실제
시간으로 얼마나 바뀌는지가 장치의 fp64 처방에 어떻게 의존하는지** 측정한다.

**보고서:** `docs/36_mixing_length_axis.md`
(발견: 같은 알고리즘 최적화가 RTX 5090 fp64 에서 1.6~1.9배, 같은 카드 fp32 에서 1.01~1.09배,
H100 에서 1.1~1.2배 — 최적화의 가치가 장치·정밀도의 함수다. 이득은 nz 와 함께 자란다: 1.45배(15)
→ 1.90배(30) → 2.72배(60))

**노드:** gpgpu (RTX 5090 + Threadripper 9955WX) · ktcloud (H100) · geo85 (EPYC 9655, PBS 배타)

**상태:** **완료** (2026-09-13). 400²·1000² 전 축 — 2 카드 × 2 정밀도 × 구현 4종 × 스레드 4점 × nz 3점. 2000² 는 CUDA 만(JAX OOM/N21, 5090 fp64 OOM/N24). ktcloud 오염분 재측정 완료(N26).
· geo85 PBS `mxl_nx400` 실행 중

**게이트 (R2, 모두 통과):**
- 로컬 gfortran: `cfd_exp3d5_serial`, `cfd_exp3d5` 각 12/12 (신규 `v06.mxl_rec.*` 2건 포함, 오차 2e-16)
- 로컬 JAX: 12/12 · gpgpu: CUDA·fp32검증·OpenACC 36/36 · ktcloud: CUDA·fp32검증 24/24
- 물리: V6-3 Kato–Phillips `recursive` 비율 0.80~1.00 (`integral` 0.80~0.97)

**재현 명령:**
- 게이트: `bash tools/gate3d5.sh` (케이스 `v06.mxl_rec.tke.theta`, `v06.mxl_rec.tke.fb`)
- 물리: `python3 tools/verify_v06.py --only v6_3 --mxl recursive`
- 축: `GRIDS="400 1000 2000" BACKENDS="cuda cuda_sp" bash tools/mxl_axis.sh`
- nz 스윕: 위 명령을 `NZ=15 30 60`, `GRIDS=400`, `SCHEMES=fb` 로 반복
- CPU: `qsub -v NX=400,CASES="lock_exchange_v06 lock_exchange_v06r",TAG=".mxl",... tools/pbs_tier2.sh`
- 수집: `python3 tools/tier2_collect.py <csv...> --out <dir>` → `mxl_cost.json`

**데이터 (`data/`):** `v06_verification_recursive_20260923.json`(V6-3[recursive] PASS — N44 뒤 재실행, 현재 코드의 결과), `gpgpu.csv`(400²·1000²·2000², fp64+fp32), `ktcloud.csv`(오염행 포함 원본),
`ktcloud_remeasure.csv`(2026-09-13 재측정), `gpgpu_nz.csv`(nz 15/30/60),
`geo85.csv`(EPYC 9655 직렬/1/96/192 스레드), `gpgpu_frameworks.csv`(OpenACC·JAX·OpenMP), `ktcloud_remeasure2.csv`(마지막 한 칸 ×2), `mxl_cost.json`, `v06_verification_recursive.json`(V6-3 물리 검증).
ktcloud 행은 `python3 tools/shared_node_min.py data/ktcloud*.csv` 로 정리한다 —
반복 스윕의 **최소-중앙값**, MAD > 2 % 스윕은 탈락(N26).

**코드 리비전:** 스펙 S11.2 `mxl` 축 추가 + 다섯 백엔드 구현 (2026-09-13). CUDA 는
`CUDA_LAUNCH_OK` 발사오류 검사 추가분 포함(N25), 스윕은 구성별 `gpu_samples.log` 포함(N26).

**주의:**
- `rtol=1e-6` 이므로 E13/E14(1e-12) 행과 같은 표에 섞지 않는다(R5).
- 두 형태는 **답이 다르다**(u 상대 L2 1.6e-3) — 정밀도 축과 같은 성격의 점이며, fp64/fp32 처럼
  열을 분리한다.
- `ktcloud.csv` 의 2000² 및 fp32 θ0.5 1000² `integral` 행은 **경합 오염**이다. 반복수가 같고
  MAD 가 0.07~0.89 % 인데도 오염이었다 — 판정 근거는 docs/90 **N26**. 재측정본으로 대체한다.
