# E19_multinode_scaling

**질문/내용:** 같은 코어를 노드/카드 수를 늘려 돌릴 때의 강·약 확장성 — EPYC 9655 1·2·4 노드(MPI+OpenMP, IB), RTX 5090 1·2·4장(PCIe, NVLink 없음), H100 1·N 노드. 동일 정확도(R2 게이트, 반복수 일치) 유지.

**보고서:** docs/41_multinode_design.md (설계), **docs/42_multigpu_scaling.md (결과, 2026-09-23)**

**노드:** gpgpu (RTX 5090 ×4, 장치 1·0·2·3; 보고 세션 2026-09-24 는 다른 사용자 GPU 프로세스 없음, 부하 ≈ 1.7) · ktcloud H100 ×2 (2노드 × 1장) · geo85 다중 노드는 이월

**상태:** **완료 — RTX 5090 ×2·×4(노드 안, 2026-09-24) + H100 ×2(2노드, MPI, 2026-09-23)**; 9/23 의 RTX 2장 세션(GPU 2·3 타 사용자 점유)은 9/24 의 유휴 노드 1·2·4장 세션으로 대체(값 1 % 이내 일치)하고 `data/SUPERSEDED_gpgpu_20260923/` 로 격리. 구현: 단일 프로세스·다중 장치 CUDA 드라이버 `libs/cuda/src/cfd_exp3d5_cuda_multi.cu`(y-슬랩, 깊은 halo H=16, 스텝당 1회 pack→peer copy→unpack; fb 스킴만; docs/03 §12). **게이트:** 1·2·4장 결과가 단일 장치 상태와 비트 단위 동일(max|diff| = 0, DC·DC+TKE, H=8·16; `NDS="1 2 4" DEVS=1,0,2,3 bash tools/e19_gate.sh`). 다중 노드(MPI)는 여전히 2편째 논문으로 이월.

**재현 명령:** `make -C libs/cuda multi ARCH=sm_120` → `bash tools/e19_gate.sh`(게이트) → `DEVSETS="1 1,0 1,0,2,3" bash tools/e19_run.sh; bash tools/e19_single.sh`(측정: fb, DC/DC+TKE, 400²·1000²·2000²×30, 장치 집합 `DEVSETS`, 50 스텝, warm-up 1, 반복 5; 체인 `data/gpgpu_20260924/e19_4card_chain.log`). 첫 시도(cudaMemcpy3DPeer 스트라이드 복사)는 halo 교환이 스텝당 수십 ms 로 느려 폐기(`output/e19_superseded_strided_*`, 원고 사용 금지).

**데이터 (`data/`):** gpgpu — `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_gpgpu_20260924-102645.csv`(24행: 3격자 × 2물리 × {드라이버 1장, 2장, 4장, 단일 이진}, 1장 2000² 은 OOM 행; 열 `max_abs_diff_vs_1dev` 전부 0), `expr/E19_multinode_scaling/data/gpgpu_20260924/*_metrics.json`(18개), `expr/E19_multinode_scaling/data/gpgpu_20260924/run.log`, `expr/E19_multinode_scaling/data/gpgpu_20260924/gate_gpgpu_4card_20260924.txt`(게이트 출력, max|diff|=0 ×10), `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_topo_20260924.txt`(PCIe 토폴로지), `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_4card_chain.log`, `expr/E19_multinode_scaling/data/gpgpu_20260924/nvidia_smi_before.txt`·`expr/E19_multinode_scaling/data/gpgpu_20260924/nvidia_smi_after.txt`(노드 점유 스냅샷). 격리(원고 사용 금지, 참고용): `expr/E19_multinode_scaling/data/SUPERSEDED_gpgpu_20260923/`(9/23 2장 세션 19파일). ktcloud — `expr/E19_multinode_scaling/data/ktcloud/e19_ktcloud_20260923-204200.csv`(24행: 3격자 × 2물리 × {드라이버 1장, 단일 이진, MPI 2노드 host-staged, MPI 2노드 CUDA-aware}), `expr/E19_multinode_scaling/data/ktcloud/*_metrics.json`(24개), `expr/E19_multinode_scaling/data/ktcloud/run.log`, `expr/E19_multinode_scaling/data/ktcloud/gate_ktcloud_20260923.txt`(1장 드라이버 게이트), `expr/E19_multinode_scaling/data/ktcloud/gate_mpi_20260923.txt`(2노드 MPI 게이트, max|diff|=0 ×11), `expr/E19_multinode_scaling/data/ktcloud/gate_mpi_tke_20260924.txt`(DC+TKE 의 CUDA-aware·halo 8 게이트 4행, max|diff|=0 ×4; 외부 감사 지적으로 추가), `expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_before.txt`·`expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_after.txt`, `expr/E19_multinode_scaling/data/ktcloud/kt_chain.log`·`expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_before_rerun.txt`·`expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_after_rerun.txt`(2026-09-24 재측정: DC+TKE 1000² host-staged 행 — 9/23 실행은 기록 후 종료 단계에서 rank 1 segfault → 재측정값 29.88 ms 로 CSV 행 대체, 원래 행은 `run.log` 에만).
**보존 상태:** 상태 이진(`*_state3d5.bin`)·번들(`*.nml`, `*_domain.bin`, `*_init.bin`)은 gpgpu `output/e19_20260923-183552/` 에만 있음(재생성 가능).

**코드 리비전:** `libs/cuda/src/cfd_exp3d5_cuda_multi.cu`(노드 안), `cfd_exp3d5_cuda_mpi.cu`(MPI) 2026-09-23, `tools/e19_{gate,run,single,run_mpi,cmp,table}.{sh,py}`

**주의 (docs/90 참조):** 다중 노드 Allreduce 비용은 E11 의 노드 내 값(0.38 µs)과 다르다 — 네트워크 지연 × log P.
