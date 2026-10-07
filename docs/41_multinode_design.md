# 41. E19 다중 노드 · 다중 GPU 확장 실험 — 설계 초안 (2026-09-16)

**상태:** 설계 초안 — **2026-09-23 부분 재개:** 노드 안 다중 GPU(단일 프로세스, 깊은 halo, fb 만)를 E19 로 구현·측정(docs/42). 다중 노드·MPI·다중격자 분산은 여전히 보류. 원래 메모: 현 원고(paper/caf)는 단일 노드 대 단일 카드로 마감하기로 결정; 다중 장치·분할 방식·통신 경로는 2편째 논문의 주제로 이월. 아래는 그때를 위한 초안이다. 코드 변경 전에 이 문서와 `docs/03` §12(신설 예정)를 먼저 고친다(R1).
**질문:** 같은 코어를 노드/카드 수를 늘려 돌리면 스텝당 시간이 어떻게 줄고, CPU 노드 N개 대 GPU M장의 동일 정확도 비교가 어떻게 바뀌는가.
**원칙:** 이번에도 어느 장치가 이겨야 할 이유는 없다. 강/약 확장성과 노드-환산치를 있는 그대로 싣는다.

## 0. 현황 조사 (2026-09-16 확인)

| 노드 | 장치·통신 | MPI | 확인 결과 |
|---|---|---|---|
| geo85 | EPYC 9655 ×4 노드 가용(node05 down), **InfiniBand(mlx5, ibs2)** | `/usr/mpi/gcc/openmpi-4.1.7a1` (gfortran, IB 지원), `/appl/mpi/mpich-4.3.1-{aocc,oneapi}` | 현재 4 노드 모두 `job-exclusive` — 사용자의 NEMO 잡(`hs22_tke_wind`, node02+node03, 160 mpiprocs ×2)과 held 잡 8건. **우리 잡은 대기해야 함** |
| gpgpu | RTX 5090 ×4, 전부 PCIe (`nvidia-smi topo`: NODE, **NVLink 없음**) | NVHPC 25.11 HPC-X(Open MPI, CUDA-aware 옵션), NCCL, NVSHMEM | GPU 0·1 은 타 사용자 점유(29.9 GB) — 지금은 2장(2·3)만 유휴 |
| ktcloud | 컨테이너당 H100 1장, NIC 4개(mlx5), MPI(HPC-X)·NCCL 있음 | 사용자가 다중 노드 할당 가능하다고 함 | **필요 정보:** 노드 수, 호스트명/IP, 노드 간 IB 여부, 컨테이너 간 ssh 또는 스케줄러(SLURM/PBS), 공유 파일시스템 |

**RTX 5090 다중 GPU 는 "소프트웨어식" 병렬이 맞다.** NVLink 가 없으므로 카드 간 데이터는 PCIe 를 거친다(P2P 가 켜지면 PCIe 직결, 아니면 호스트 메모리 경유). E11 에서 잰 값(docs/28): 2랭크 halo 24~55 µs, Allreduce 0.38 µs(노드 내 공유메모리). 다중 노드에서는 Allreduce 가 네트워크 지연 2~10 µs × log P 로 커진다.

## 1. 설계 선택 (확인 필요 ★)

| 항목 | 선택안 | 이유 |
|---|---|---|
| ★ 분할 | **1D 슬랩(y 방향)**, halo 폭 2 | 코드가 `ip/im/jp/jm` 인덱스 표(Fortran)·`wrap3`(CUDA)로 주기 인덱싱 → j 방향만 로컬화하면 x 방향 연산자는 손대지 않음. 3차 이류(i±2) 때문에 폭 2. 2D 분할은 랭크 수가 많아질 때 유리하나 구현량 2배 |
| ★ 솔버 | **fb 와 θ·PCG-Jacobi** 만 | PCG 는 Allreduce 2회/반복 + halo 1회/반복으로 분산이 단순. **다중격자는 제외**(분산 V-사이클은 별도 작업) — 한계로 명시 |
| 스킴 | fb, θ0.5·PCG, split(부분스텝마다 halo) | split 은 필터 없음 그대로 |
| 물리 | DC, DC+TKE | 폐쇄는 열 단위라 통신 없음 |
| 구현 | **Fortran+MPI(+OpenMP)** → 같은 소스로 OpenACC+MPI(GPU 노드에서 랭크당 1 GPU) · **CUDA+MPI**(랭크당 1 GPU, halo pack/unpack 커널, CUDA-aware 또는 호스트 스테이징 둘 다 측정) | JAX 다중 장치(sharding)는 2단계 |
| 게이트 | R2: N 랭크 결과를 gather → NumPy 오라클과 1e-12/1e-9. **반복수는 랭크 수와 무관하게 일치해야 함**(Allreduce 합산 순서로 마지막 자리 차이 가능 → 기록) | R4-2: 노드×백엔드×**랭크 수**마다 게이트 기록 |
| 문제 | 2000²×30(현 최대) + **4000²×30**(약 4.8×10⁸ 셀, 1노드 CPU 로 가능, 단일 H100 은 fp64 로 안 들어감 → 다중 GPU 의 존재 이유) | 강확장: 고정 문제, 1→2→4 노드/카드. 약확장: 랭크당 1000²×30 고정 |
| 측정 | R7 그대로(5회 median+MAD), 통신 시간 별도 열(halo·Allreduce 누적), 노드 배타 | 노드-환산치: "H100 M장 = EPYC 노드 N개" 를 같은 문제·같은 정확도에서 |

## 2. 코드 변경 범위 (추정)

| 파일 | 변경 | 규모 |
|---|---|---|
| `docs/03` §12 | 분할·halo·전역 합 정의, 결정적 합산 규칙 | 0.5일 |
| `libs/fortran/src/mod_comm.f90` (신설) | MPI 초기화, 슬랩 범위, halo 교환(2D·3D 배열), Allreduce 래퍼, gather | 1일 |
| `mod_grid.f90` | 로컬 ny·halo 를 포함한 `jp/jm` 표(경계 랭크만 wrap) | 0.5일 |
| `mod_model3d_v05.f90` | 스텐실 전 halo 호출 삽입(스텝당 약 20곳, PCG 반복당 1곳) | 1~2일 |
| `mod_solvers.f90`, `mod_helm_var.f90` | `red_*` 를 Allreduce | 0.5일 |
| `cfd_exp3d5.f90` | 도메인 파일을 읽어 슬랩 추출, 상태 gather 후 출력·검증 | 0.5일 |
| `libs/cuda/src/cfd_exp3d5_cuda.cu` | 동일 구조: 로컬 ny+2h, pack/unpack 커널, MPI 호출, PCG Allreduce | 1~2일 |
| `tools/gate3d5.sh`, `tools/tier2_sweep.sh` | `-np` 축, mpirun 래퍼, 게이트 gather | 0.5일 |
| 측정 | geo85 1·2·4 노드(PBS `select=N:ncpus=192`), gpgpu 1·2·4 GPU(0·1 이 비면), ktcloud 1·2·…N H100 | 대기시간 포함 2~4일 |

합계 약 **2주**(코딩 5~7일 + 게이트·측정). 다중격자까지 포함하면 +1주.

## 3. 사용자에게 필요한 것

1. ★ 1D 슬랩 + PCG-only 로 갈지(빠름), 2D + 다중격자까지 갈지(완전하지만 +1~2주).
2. ktcloud 다중 노드: 노드 수·호스트명·노드 간 IB·컨테이너 간 ssh/스케줄러·공유 FS.
3. geo85: 우리 확장 잡을 언제 넣을 수 있는지(현재 4 노드 점유 중). 4 노드 배타로 한 번에 2~3시간이면 강·약확장 한 세트가 끝난다.
4. gpgpu GPU 0·1 이 비는 시점(4장 측정용).

## 4. 원고에의 반영

E19 결과는 `paper/caf` 의 4.1 지도에 "노드/카드 수" 축을 더한 표(4.1b) 와 확장성 그림 1장으로 들어간다. 결론의 "카드 한 장 = EPYC 노드 1/4~여러 개" 문장이 "M장 = N노드" 로 바뀐다. 한계 절의 "single node" 문장은 삭제된다.
