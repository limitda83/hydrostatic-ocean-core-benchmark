# E18_mixed_precision

**질문/내용:** 혼합정밀도 — **fp64 코어 + fp32 폐쇄**. docs/37 §2-3 이 학습 모듈의 자연스러운 구조로 예측한
것이고, docs/34 §5 가 "미시도"로 남겨 둔 축. 폐쇄(`k_closure`)만 fp32 로 계산했을 때 (1) 폐쇄 비용이
얼마나 줄고, (2) 해가 fp64 에서 얼마나 벗어나며, (3) 그것이 일괄 fp32 와 어떻게 다른가.

**구현:** `tools/make_mixed_cuda.py` 가 fp64 CUDA 소스에서 `k_closure` 의 **본문만** `closure_t` 로 다시
쓴다(커널 서명·모든 장치 배열은 double 유지; 적재 시 좁히고 저장 시 넓힘. 스칼라 인자·전역 배열 읽기·
`safe_inv` 반환값도 적재 시 좁혀 산술이 실제로 fp32 에서 돈다). `libs/cuda/Makefile` 의 `all3d5-mixed`
(`-DCLOSURE_T=float`) · `all3d5-mixedcheck` (`-DCLOSURE_T=double`). `tools/tier2_sweep.sh` 에 백엔드
`cuda_mixed` 추가.

**검증 (2026-09-15, gpgpu):**
- **생성의 충실성:** `mixedcheck`(폐쇄를 다시 double 로) 의 상태파일이 원본 fp64 이진과 **바이트 단위로 동일**
  (128²×20, v0.6, 20 스텝, θ·mg CFL 4).
- **편차 (별도 축, 게이트 아님 — R9):** 같은 문제에서 fp32 폐쇄 대 fp64, 상대 L2 —
  η 4.9e-7 · u 2.8e-5 · **v 3.1e-5** · b 2.6e-7 · T 3.3e-8 · S 2.4e-15. 일괄 fp32 (docs/34 §4, 400²·50스텝):
  v 2.6e-3. **폐쇄만 fp32 로 내리면 편차가 두 자릿수 작다** — 저정밀이 위험한 곳(전역 solver)을 fp64 에
  남겼기 때문.

**보고서:** docs/34_precision_axis.md §6 (결과), docs/37_ai_module_motivation.md §2-3 (동기), docs/90 N41 후속

**노드:** gpgpu (RTX 5090) · ktcloud (H100)

**상태:** **완료** (2026-09-16, 두 카드). docs/34 §6. 폐쇄 비용 mixed/fp64: 5090 0.11~0.13× · H100 0.67~0.70×. 스텝 전체 mixed/fp64: **5090 0.51~0.58× · H100 0.89~0.91×**. 두 카드 모두 fp32 다중격자는 CFL 2 에서 max_iter, mixed 는 fp64 와 반복수 동일(1000/1000, 934/934). 건전성(`.v05` mixed/fp64) 0.999~1.007. 생성 충실성(폐쇄 fp64 재빌드 = 원본 비트 동일)과 편차(v 3.07e-5)가 두 노드에서 일치.

**재현 명령:**
- 생성·빌드: `python3 tools/make_mixed_cuda.py && cd libs/cuda && make all3d5-mixed all3d5-mixedcheck`
- 비트 검사: 같은 namelist 로 `cfd_exp3d5_cuda` 와 `cfd_exp3d5_cuda_mixedcheck` 를 돌려 `cmp` 상태파일
- 편차: `python3 tools/state_error3d5.py mixed.bin orig.bin --nx 128 --nz 20 --ts`
- 분석: `python3 tools/mixed_report.py --decompose expr/E14_v06_representative/data/decompose_gpgpu_fb_20260915.csv expr/E18_mixed_precision/data/decompose_gpgpu_mixed_fb_20260915.csv --ladder expr/E18_mixed_precision/data/ladder_gpgpu_mixed_20260915.csv`
- 측정: `output/mixed_matrix.sh` (gpgpu) — `BACKENDS=cuda_mixed ... decompose_physics.sh` + `tier2_sweep.sh TAG=.mixed`

**데이터 (`data/`):** `decompose_{gpgpu,ktcloud}_mixed_fb_20260915.csv`(fb 분해 5변종 × 3격자, `cuda_mixed`), `ladder_{gpgpu,ktcloud}_mixed_20260915.csv`(`lock_exchange_v06` fb·θ·mg, 400²·1000², fp64/fp32/mixed), `ladder_{gpgpu,ktcloud}_fp32_maxiter_RESULT.csv`(일괄 fp32 다중격자 max_iter 행 — 결과, SUPERSEDED 참조)

**보존 상태:** 두 노드 원자료 회수 완료.
회수한다. `ktcloud` 는 세션 종료 시 `/home/work` 가 삭제되므로 측정 직후 회수(R13-2).

**코드 리비전:** 생성기 `tools/make_mixed_cuda.py`, 생성 파일 `libs/cuda/src/cfd_exp3d5_cuda_mixed.cu`

**주의:** fp32 폐쇄는 R2 게이트 대상이 아니다(1e-12 를 만족할 수 없고, 그것이 목적도 아니다). 게이트는
`mixedcheck` 의 비트 동일성이 대신한다 — 생성기가 fp64 의미를 바꾸지 않았음을 증명한다.
