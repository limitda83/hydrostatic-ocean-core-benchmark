# 실험 설계안 (Experiment Design)

## 1. 연구 질문

> **RQ0 (프레이밍).** "정밀도"와 "속도"는 별개 질문이 아니다.
> 음해법은 큰 `dt`를 쓸 수 있지만 스텝당 비싸고 시간 이산화 오차가 커진다.
> 따라서 유일하게 공정한 비교축은 **고정 오차에 도달하는 데 걸린 시간(time-to-solution at fixed error)**,
> 즉 `error × wallclock` 파레토 프론티어다. 이 실험의 모든 결론은 이 평면 위에서 진술한다.

- **RQ1 (수치).** 양해법(forward–backward) / semi-음해법(θ-method, Casulli형) / 완전음해법(θ=1)의
  파레토 프론티어는 어디서 교차하는가? SCHISM류가 관행적으로 쓰는 θ≈0.55–0.6의
  "강건성 대가로 지불하는 정확도"는 정량적으로 얼마인가?
- **RQ2 (알고리즘–하드웨어 상호작용).** semi-음해법의 비용은 2D 자유수면 Helmholtz(타원형) 해에
  지배된다. 이 전역 해(PCG의 전역 reduction)가 GPU에서 병목이 되어,
  CPU에서 이기는 스킴이 GPU에서 지는 **순위 역전**이 일어나는가?
- **RQ3 (구현).** 동일 알고리즘을 Fortran(CPU) / CUDA / OpenACC / Python-GPU 로 구현했을 때
  성능 차이는 얼마이며, **코드량·유지보수 비용**을 함께 놓으면 어떤 선택이 합리적인가?
- **RQ4 (정밀도).** 대역폭 지배(bandwidth-bound) 커널에서 fp64 → fp32/mixed 전환의
  실이득과 정확도 손실은? (특히 소비자용 GPU의 열악한 FP64 비율 하에서)
- **RQ5 (규모).** 목표 격자 100×100×30 = 3×10⁵ 셀은 GPU에게 **너무 작다.**
  CPU→GPU 손익분기 격자 크기는 어디인가?
- **RQ6 (지형·spec v0.5).** 실제 해저지형이 들어오면 자유표면 Helmholtz 는 변계수가 된다.
  반복수는 거칠기(`r_std`, `rx0`)와 시간간격의 어떤 함수인가? 그리고 그 변화가
  RQ2 의 CPU↔GPU 교차점을 어디로 옮기는가?
  → **측정 결과 docs/25 §4: 지형은 CFL ≤ 32 에서 반복수를 바꾸지 않고, CFL 128 에서만
  PCG 를 49~61% 늘린다. 그 증가는 거칠기의 함수가 아니라 지형 유무의 계단이고,
  다중격자는 거의 면역이다(+18%).**
- **RQ7 (하드웨어 세대).** 소비자용 RTX 5090 에서 얻은 결론이 데이터센터 H100 로 이전되는가?
  fp64 처리량 17배·대역폭 2.1배 차이가 어떤 커널에서 실제 이득으로 나타나는가?
  → **측정 결과 docs/25 §8: PCG 에서는 H100 이 진다(22.3 vs 14.4 ms). 그 구간은 지연시간
  지배이고 커널 발사 시간은 하드웨어로 사지 못한다. 커널이 크고 적은 다중격자에서만
  H100 이 앞선다(2.00 vs 2.73 ms).**
- **RQ8 (보고 방법론).** 같은 측정에서 "GPU 가속비"는 기준선 선택에 따라 얼마나 달라지는가?
  → **측정 결과 docs/25 §2: 512²·CFL128 에서 1.95배와 322.2배가 동시에 성립한다 —
  165배의 재량. R8-1 이 이 결과에서 나왔다.** (2026-09-13 안정화 재측정 수치 — docs/90 N28, N29.)

---

## 2. 측정된 하드웨어 현황 (2026-09-09/10 실측)

| | 로컬 (개발/검증) | **`gpgpu`** — GPU 기준 노드 | **`geo85`** — CPU 확장성 노드 |
|---|---|---|---|
| 접속 | 로컬 | `ssh <gpgpu-node>` | `ssh <geo85-node>` |
| 구성 | 단일 워크스테이션 | 단일 워크스테이션 | **PBS 클러스터** (master + node01–05) |
| CPU | Apple M4 Pro, 12C, arm64 | AMD Threadripper PRO 9955WX, 16C/32T | 로그인: 2× EPYC 9115 (32C) · **계산노드: 192C × 5대** |
| 메모리 | — | — | 로그인 377 GB · **계산노드 754 GB/대** |
| GPU | (Metal only) | **RTX 5090 32GB × 4**, driver 580.126 | **없음** |
| 저장소 | 로컬 | 1.3 TB | **262 TB 공유** (`/home`, 계산노드 공유) |
| OS | Darwin 25.3.0 | RHEL 9.7 | RHEL 9.4 |
| Fortran | gfortran 15.2 ✅ | **없음 ❌** | gfortran 11.4 ✅ |
| C/C++ | clang 17 | gcc ✅ | gcc/g++ 11.4 ✅ |
| CUDA | 없음 | nvcc **13.2** ✅ | 없음 |
| NVHPC SDK (OpenACC/CUDA Fortran) | 없음 | **없음 ❌** | 없음 (GPU가 없어 불필요) |
| MPI | 없음 | 미확인 | **OpenMPI 4.1.7** ✅ (`/usr/mpi/gcc/openmpi-4.1.7a1/bin`) |
| 스케줄러 | — | 없음 (직접 실행) | **PBS** (`workq`; node02/03 사용 중, node01/04 유휴, node05 down) |
| Python | 3.13 (numpy/scipy/jax/torch/netCDF4) | **3.9.25** (numpy 2.0.2, netCDF4) | **3.9.18** (numpy 2.0.2, netCDF4) |
| 배포 위치 | `/Volumes/workarea/work/cfd_exp` | `~/cfd_exp` ✅ 배포·검증 완료 | `~/cfd_exp` ✅ 배포·검증 완료 |

### 2.1 노드 역할 분리 (RULES.md R7-1)

> **`geo85`에는 GPU가 없다.** 따라서 CPU↔GPU 비교는 `gpgpu` 한 노드 안에서만 성립한다.
> `geo85`의 CPU 시간과 `gpgpu`의 GPU 시간을 나란히 놓으면 서로 다른 CPU를 섞은 무효 비교가 된다.

- **`gpgpu`** — RQ2/RQ3/RQ5의 1차 수치(CPU↔GPU 순위 역전, 백엔드 비교, 손익분기 격자크기).
  단, CPU 쪽이 16코어뿐이라 대규모 CPU 확장성은 볼 수 없다.
- **`geo85`** — 그 빈틈을 메운다: **192코어 OpenMP 스윕**, **MPI 다중노드**, **NUMA 효과**.
  `gpgpu`로는 불가능한 독립 축이며, 실제 해양모델 운영 환경(대규모 CPU 클러스터)에 더 가깝다.
- Python 3.9뿐이라 `tomllib`(3.11+)를 쓸 수 없다 → `libs/utils/config.py`가 `tomli` 로 폴백한다.

### 2.2 즉시 해결해야 할 환경 블로커

| # | 블로커 | 영향 | 노드 |
|---|---|---|---|
| B1 | **NVHPC SDK 미설치** (`nvfortran`, `nvc++`) | **OpenACC·CUDA Fortran 축 성립 불가** — Phase 4/5 선행조건 | `gpgpu` |
| B2 | gfortran 미설치 | GPU 노드에서 Fortran CPU 기준선을 못 만듦 → 동일노드 CPU↔GPU 비교 불가 | `gpgpu` |
| B3 | GPU Python 스택 부재 (`cupy`/`jax`/`numba`) | Phase 6 불가 | `gpgpu` |
| B4 | cmake 없음, MPI 미확인 | 빌드/병렬 축 제약 | `gpgpu` |
| B5 | Intel oneAPI가 `installer` 만 존재 (컴파일러 미설치) | `ifx` 대 `gfortran` 비교 불가 (선택 축) | `geo85` |

> B1·B2는 **`gpgpu` 노드에서 Fortran 계열 GPU 축 전체를 막고 있다.** 최우선 해소 대상.

### 2.3 RTX 5090 관련 전제

RTX 5090은 **Blackwell GB202 소비자칩** → FP64:FP32 처리율이 1:64로 알려져 있다.
Phase 3에서 **마이크로벤치마크로 실측 확인** 후 RQ4의 전제로 삼는다.
단, 해양모델 커널은 대역폭 지배이므로 FP64 페널티가 FLOP 비율만큼 크지 않을 수 있다 —
이것이 이 하드웨어에서만 물을 수 있는 고유한 질문이다.

---

## 3. 실험 축 (Factorial Matrix)

| 축 | 수준 | 비고 |
|---|---|---|
| **A. 시간적분** | `fb` (forward–backward, 양해) · `theta=0.5` (CN) · `theta=0.55` · `theta=0.6` (SCHISM류) · `theta=1.0` (완전음해) · `split_explicit` (ROMS/NEMO류) | Phase 1~ |
| **B. 타원형 해법** | `fft` (주기경계 + 평탄바닥 전용, 정확) · `pcg_jacobi` · `pcg_rbgs` · `rbgs` · `multigrid` | semi-음해법 전용. 지형이 있으면 `fft` 는 사용 불가(spec S10.5) |
| **C. 이류(advection)** | `centered2` · `upwind1` · `tvd_superbee` · `elm` (Eulerian–Lagrangian, SCHISM류) | Phase 2~ |
| **D. 백엔드** | `ref` (NumPy fp64) · `fortran_cpu` (+OpenMP) · `openacc` · `cuda` · `py_gpu` ∈ {CuPy, JAX, Numba-CUDA} | |
| **E. 격자** | 32² · 64² · **100²** · 200² · 400² · 800² (수평), 수직 30층 고정 | RQ5 크로스오버 |
| **F. dt** | 각 스킴의 안정 한계 대비 0.1 / 0.25 / 0.5 / 1 / 2 / 4 / 8 배 | 파레토 곡선 생성용 |
| **G. 정밀도** | `fp64` · `fp32` · `mixed` | fp64가 기준선 (R9). 문제는 항상 fp64 로 저장하고 **해법의 산술만** 바꾼다 |
| **H. 지형 (spec v0.5)** | `flat` · `slope` · `seamount` · `ridge` · `rough(r_std = 0.02…0.40)` | rx0 0.000→0.639. 격자 스윕에는 `spectrum_kmax` 로 대역제한한 지형을 쓴다 |
| **I. 연직좌표** | `zlevel`(부분셀) · `zstar` · `sigma` | 압력경사 보정 on/off 와 **직교하지 않으므로 반드시 함께 명시**(docs/24 §3) |
| **J. 상태방정식** | `linear` · `seos`(캐블링·열압축성) · `teos10`(55항) | 코어에서 유일한 연산집약 커널 |
| **K. 하드웨어** | `gpgpu`(RTX 5090) · `ktcloud`(H100) · `geo85`(EPYC 9655) | R7-1 의 역할 분리를 엄수 |

전 조합은 조합폭발이므로 **단계별 부분 factorial**로 진행한다 (§6 Phase 게이트).

---

## 4. 검증 케이스 스위트 (Verification Suite)

원칙: **해석해가 있거나 보존량이 정확히 알려진 케이스만** 검증에 쓴다.
"그림이 그럴듯하다"는 검증이 아니다. 각 케이스는 **격자 세분화 하 수렴차수**를 측정한다.
(Bishnu et al. 2024 JAMES 의 barotropic solver 검증 스위트 구성을 따름)

| # | 케이스 | 경계 | 해석해 | 무엇을 검증하나 | Phase |
|---|---|---|---|---|---|
| V1 | **관성중력파 (inertia–gravity wave)** | 이중주기 | ✅ 정확 | 공간·시간 수렴차수, 위상오차, 수치감쇠 | **0** |
| V2 | **지형류 평형 (geostrophic balance)** | 이중주기 | ✅ 정상해 | 정상상태 유지, 장시간 드리프트, 질량/에너지 보존 | **0** |
| V3 | 연안 켈빈파 (coastal Kelvin wave) | 벽+주기 | ✅ (비분산) | 벽 경계 처리, 비분산 전파 | 1 |
| V4 | 로스비파 (planetary / topographic) | β-plane | ✅ | 분산관계, 지형 항 | 1 |
| V5 | 조석 강제 (barotropic tide) | 개방경계 | ✅ | 개방경계·강제항 | 1 |
| V6 | **MMS (제작해)** | 이중주기 | ✅ 구성적 | 비선형 이류항 포함 전 항의 구현 정확성 | 1 |
| V7 | 가우시안 추적자 회전이류 | 이중주기 | ✅ | 이류 스킴 보존성·단조성·수치확산 | 2 |
| V8 | **Lock exchange (밀도류)** | 폐쇄 | ❌ (RPE 진단) | **가짜 등밀도혼합**(spurious diapycnal mixing), Re_Δ | 2 |
| V9 | 내부 정진동 (internal seiche) | 폐쇄 | ✅ (선형) | 성층·경압 모드, 모드 분리 | 2 |
| V10 | 바람강제 폐쇄분지 (Ekman) | 폐쇄 | ✅ (정상) | 연직 점성 음해 처리, 경계층 해상 | 2 |
| V11 | 경압 제트 불안정 (Soufflet형) | 채널 | ❌ (모델간 비교) | 중규모/준중규모 에너지, 유효 해상도 | 3 |

### 4.1 정확도 지표

| 지표 | 정의 | 용도 |
|---|---|---|
| `L2_rel`, `Linf_rel` | 해석해 대비 상대오차 | 주 지표 |
| `order_p` | `log2(E(dx) / E(dx/2))` | **수렴차수 — 구현 정확성의 유일한 증거** |
| `amp_ratio` | N주기 후 진폭 / 초기 진폭 | 수치감쇠 (θ>0.5의 대가) |
| `phase_err` | 수치 위상속도 / 해석 위상속도 − 1 | 분산오차 |
| `dM/M0` | 전질량 상대 드리프트 | 보존성 |
| `dE/E0` | 전에너지(운동+위치) 상대 드리프트 | 보존성 |
| `RPE(t)` | 기준위치에너지 증가율 | **가짜 등밀도혼합** (Ilıcak et al. 2012) |
| `Re_Δ` | `U Δx / A_h` (격자 레이놀즈수) | Ilıcak 기준 `Re_Δ < 10` 준수 여부 |

### 4.2 성능 지표

| 지표 | 정의 |
|---|---|
| `t_kernel` | 시간적분 루프만 (I/O·JIT 제외), median ± MAD, n≥5 |
| `t_e2e` | 프로세스 시작~종료 |
| `t_compile` | JIT/컴파일 시간 (Python-GPU 축에서 별도 보고 — 숨기지 않음) |
| `SDPD` | simulated days per day (throughput) |
| **`TTS@ε`** | **목표 오차 ε 달성까지의 wallclock — 최상위 지표 (R8)** |
| `AI`, roofline 위치 | 산술강도(FLOP/byte), 달성 대역폭 / 피크 대역폭 |
| `E_sol` | Joule/simulated-day (`nvidia-smi`/RAPL 적분) |
| `SLOC`, `Δ SLOC` | 구현 코드량 및 기준선 대비 증분 — **유지보수 비용 대리지표 (RQ3)** |

---

## 5. 지배방정식 범위

상세 정의는 `docs/03_discretization_spec.md`. 요약:

- **Phase 0–1: 선형 회전 천수방정식 (linear rotating shallow water, 2D)**
  → 순압(barotropic) 모드만. 해석해가 존재하므로 **모든 스킴·백엔드의 검증 하네스**가 된다.
- **Phase 2+: 3D 정역학 원시방정식 (hydrostatic primitive equations)**
  연속방정식 + 운동량방정식 + 추적자(온·염) 보존방정식 + 상태방정식,
  Arakawa C-grid, z\*/σ 연직좌표, 100×100×30.
- 비정역학(non-hydrostatic)은 **범위 밖** — 필요하면 별도 프로젝트.

---

## 6. Phase 게이트

| Phase | 내용 | 게이트 산출물 (없으면 다음 Phase 금지) |
|---|---|---|
| **P0** | 하네스 + NumPy 레퍼런스 2D 선형 RSW, V1/V2, θ-family + FB | `output/verify_*/` 에서 **V1 수렴차수 ≈ 2 (θ=0.5), ≈ 1 (θ=1)** 확인 |
| **P1** | 스킴 확장(split-explicit, PCG/MG), V3–V6, CPU 파레토 곡선 | `docs/10_pareto_2d.md` + 오차-시간 곡선 |
| **P2** | 3D 정역학 확장 (100×100×30), 연직 음해확산(배치 삼중대각), 추적자, V7–V10 | `ref` 3D가 V8 RPE 진단 통과 |
| **P3** | Fortran CPU 포팅 (+OpenMP), 마이크로벤치(FP64 비율, STREAM, roofline 상수) | `ref` 대조 `<1e-12` (1 step) |
| **P4** | OpenACC 포팅 (NVHPC 필요) | 동상, + roofline 위치 |
| **P5** | CUDA 포팅 (배치 삼중대각, PCG 전역 reduction 최적화) | 동상 |
| **P6** | Python-GPU (CuPy / JAX / Numba-CUDA) | 동상, `t_compile` 별도 보고 |
| **P7** | 전 축 factorial 벤치 + 파레토/roofline/에너지 리포트 | `docs/20_final_report.md` |

---

## 7. 위험요소와 대응 (Threats to Validity)

| # | 위험 | 왜 문제인가 | 대응 |
|---|---|---|---|
| T1 | **100×100×30 = 3×10⁵ 셀은 GPU에 너무 작다** | 커널 실행 지연(launch latency)과 PCG 전역 reduction이 지배 → "GPU가 느리다"는 **하드웨어가 아니라 문제크기의 결론**이 된다 | 격자 스윕(축 E)을 **필수**로 넣어 손익분기점을 찾는다. 단일 격자 결론 금지 |
| T2 | RTX 5090 FP64 1/64 | fp64 기준선이 부당하게 불리 | **해소됨** — 실측 FP32:FP64 = 47.8:1 이지만 대역폭은 1.00:1, fp64 ridge AI = 1.25. 우리 커널(AI≈0.2)은 fp64에서도 대역폭 지배 → 실이득은 48배가 아니라 **2배**. `docs/12` |
| T9 | **`gpgpu` 는 공유 노드** | GPU 0·1이 타 사용자에 점유된 상태를 발견. 기본값(GPU 0)으로 돌리면 오염된 수치가 나온다 | 측정 전 `nvidia-smi` 확인 → `CUDA_VISIBLE_DEVICES` 명시 고정 → manifest에 GPU 인덱스·점유 상태 기록 |
| T3 | 컴파일러/플래그 비대칭 | gfortran `-O2` vs nvcc `-O3` 비교는 무의미 | 플래그를 `config/backends.toml` 에 고정·기록, 각 컴파일러의 동등 최적화 수준 명시 |
| T4 | 스킴마다 dt가 다름 | "속도" 비교가 dt 선택에 좌우 | R8 — 파레토 곡선으로만 결론 |
| T5 | 이류/혼합 스킴 혼입 | 시간적분법 효과와 공간 스킴 효과가 교란 | R5 — 축 C를 고정한 상태에서 축 A 비교 |
| T6 | Python-GPU의 JIT 비용 은닉 | 짧은 run에서 유리하게 왜곡 | `t_compile` 별도 보고 의무화 (R7-5) |
| T7 | 4× RTX 5090 = 다중 GPU 유혹 | 통신 축이 추가되면 교란 | **P7까지 단일 GPU 고정.** 다중 GPU는 후속 연구 |
| T8 | 소비자 GPU에는 ECC 없음 | 장시간 run의 비트 플립 | 검증 run은 반복 실행하여 재현성 확인 |

---

## 8. 산출물 로드맵

1. `docs/10_pareto_2d.md` — 2D 스킴 파레토 (P1)
2. `docs/11_p3_fortran_cpu.md` — Fortran CPU 백엔드, R2 게이트, OpenMP 임계값 ✅
2b. `docs/12_gpu_microbench.md` — RTX 5090 FP64/FP32 처리율, 대역폭, roofline 상수 ✅
3. `docs/12_backend_matrix.md` — 백엔드별 성능 + SLOC (P4–P6)
4. `docs/20_final_report.md` — 종합 (P7)
5. `docs/90_negative_results.md` — 실패·역전 사례 (상시)
