# E11_multigpu_comm

**질문/내용:** 다중 GPU 통신 마이크로벤치 (halo/allreduce, RTX 5090 ×2)

**보고서:** docs/28_multigpu_communication.md

**노드:** gpgpu

**상태:** **완료** (2026-09-14 재측정). 2랭크 512²·1024²·2048² + 1랭크 대조. 유휴 GPU 판정을 '메모리만'에서 **'여유 메모리 + 사용률'**로 고친 뒤에야 측정됐다 — 첫 시도는 0 % 사용률로 노는 카드 두 장을 메모리 점유만 보고 4시간 거부했다(R7-7 과보수).

**재현 명령:**
- `make -C libs/cuda comm`
- `MPIDIR=$HOME/opt/nvhpc/Linux_x86_64/25.11/comm_libs/mpi; OPAL_PREFIX=$MPIDIR PATH=$MPIDIR/bin:$PATH CUDA_VISIBLE_DEVICES=<유휴 두 개> mpirun -np 2 libs/cuda/build/comm_bench`
- 대기·거부 포함 자동 실행: `bash ~/run_e11.sh` (gpgpu)

**데이터 (`data/`):** `comm_bench_20260914.txt` — 2랭크 3격자 + 1랭크 대조의 원출력과 측정 직전 GPU 상태(`nvidia-smi` CSV). 각 줄 `CSV,ranks,nx,ny,staging,matvec,halo,allreduce,dot,pcg_iter,substep` [µs].
**보존 상태:** 원자료는 한 번 유실됐고 **2026-09-14 에 재측정해 복구했다.** 구 보고서
(docs/28)의 값이 잘 재현된다 — PCG 반복 전체 512² 68.0→70.6, 1024² 103.9→107.1,
2048² 211.7→194.2 µs. Allreduce 도 0.38→0.40~0.66 µs 로 같은 자릿수다.

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
