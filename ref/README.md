# ref/ — 관련문헌과 우리 위치

> `refs.bib` 52 항목 · `ref/pdf/` 39 편(공개 접근본만; 유료는 서지만). 갱신 2026-09-16. 투고 저널·빈 영역은 `paper/TARGET.md`.

> 각 항목은 **그 논문이 한 것**과 **하지 않은 것**을 함께 적는다. 두 번째가 우리 기여의 자리다.
> PDF 는 공개접근(arXiv·EGU/Copernicus·JOSS·OSTI·기관 리포지토리)만 `ref/pdf/` 에 내려받았다.
> 유료 논문은 서지·DOI 만 적는다 (다운로드하지 않는다).

## 0. 우리 연구의 한 줄 위치

> 기존 문헌은 **(A) 스킴을 비교하거나**, **(B) 구현/프레임워크를 비교하거나**, **(C) 한 모델을 GPU 로
> 옮긴 성능을 보고한다.** 셋을 **같은 이산화 위에서 교차**시키고, 모든 구현이 **1e-12 안에서 같은 답**임을
> 증명한 뒤, **고정 오차에서의 소요시간**으로 순위를 매긴 연구는 없다. 우리가 그것을 한다.

---

## 1. 프레임워크·이식 (B, C)

| 문헌 | 한 것 | **하지 않은 것** (우리 자리) | PDF |
|---|---|---|---|
| **Häfner et al. 2021**, *Fast, Cheap, and Turbulent — Global Ocean Modeling With GPU Acceleration in Python* (JAMES, 10.1029/2021MS002717) | Veros 를 NumPy→JAX 로, Fortran 과 CPU/GPU·에너지 비교. "time to solution / power / kWh" 표 | **스킴은 하나**(그 모델의 시간적분 고정). 정확도 축 없음 — 같은 답을 전제하고 속도만 비교. GPU 메모리 한계·프레임워크가 **실행 가능성**을 가르는 지점은 다루지 않음 | `veros_jax_2021_haefner.pdf` |
| **Häfner et al. 2018**, *Veros v0.1* (GMD 11, 3299) | 순수 파이썬 해양모델, Fortran 대비 검증 | 상동 | `veros_v01_gmd_2018.pdf` |
| **Ramadhan et al. 2020** (JOSS), **Wagner et al. 2023/2025** (arXiv, JAMES) — Oceananigans.jl | Julia + KernelAbstractions 로 CPU/GPU 단일 소스, 대규모 성능(10 SYPD @8km, 64×A100) | **한 구현·한 언어.** 스킴 교차 없음. 다른 프레임워크와의 동일문제 대조 없음 | `oceananigans_joss_2020.pdf`, `oceananigans_2023_breakthrough.pdf`, `oceananigans_2025_allscales.pdf` |
| **Jendersie, Lessig, Richter 2025**, *A GPU parallelization of the neXtSIM-DG dynamical core* (GMD 18, 3017) | **CUDA·SYCL·Kokkos·PyTorch 네 프레임워크 비교** — 우리 D 축과 가장 가까움 | 해빙(sea-ice) DG 코어. **시간적분 스킴 축 없음**, 고정오차 비교 없음, 구현 간 답 일치 검증을 성능비교의 전제로 삼지 않음 | `nextsim_dg_gpu_2025.pdf` |
| **Dahm et al. 2023**, *Pace v0.2* (GMD 16, 2719) | 파이썬(GT4Py) 로 FV3 대기모델 이식, CPU/GPU 성능이식성 | 대기·한 스킴. 정확도-비용 평면 없음 | `pace_python_portable_atmos_2023.pdf` |
| **Norman et al. 2015**, *A case study of CUDA FORTRAN and OpenACC for an atmospheric climate kernel* (Parallel Computing / OSTI 1462913) | CAM-SE 이류 커널에서 OpenACC 가 CUDA 대비 ~1.5× 느림 | **커널 하나**. 전체 스텝·물리 복잡도·메모리 한계 없음 | `camse_cuda_vs_openacc_2015.pdf` |
| **swLICOM 2026** (GMD 19, 3317) | Sunway 다중코어 이식, roofline 기반 분할, 1 km | 특정 하드웨어. 프레임워크/스킴 교차 없음 | `swlicom_2026_sunway.pdf` |
| **Cros et al. 2025**, *GPU technologies for ocean forecasting* (State of the Planet 5-opsr) | 해양예보 GPU 기술 개관(PSyclone·OpenACC·프레임워크) | 개관. 통제 실험 아님 | `gpu_ocean_forecasting_2025.pdf` |
| **Silvestri et al. 2025**, *A GPU-Based Ocean Dynamical Core for Routine Mesoscale-Resolving Climate Simulations* (JAMES, 10.1029/2024MS004465) | GPU 전용 dycore, 순압 solver 설계가 성능의 핵심임을 보임 | 유료(다운로드 안 함). 한 구현. 우리는 이 논문의 **순압 solver 논점**을 스킴 축으로 삼아 정면으로 잰다 | — |
| **Brodtkorb & Holm 2021**, *Coastal ocean forecasting on the GPU* (Tellus A 73) | 2D 유한체적 천수 GPU | 2D·비정역학 아님, 스킴 축 없음 | `gpuocean_tellus_2021.pdf` |

## 1b. 하이브리드 물리-AI 파라미터화 (D) — 2026-09-16 추가

| 문헌 | 한 것 | **하지 않은 것** (우리 자리) | PDF |
|---|---|---|---|
| **Zanna et al. 2025**, *A Framework for Hybrid Physics-AI Coupled Ocean Models* (arXiv 2510.22676) | MOM6 에 여러 학습 모듈을 넣고 비용을 보고 — **같은 CNN 이 CPU 에서 코어의 ~10배, GPU 에서 ~0.1배** | 그 100배를 관찰로만 적음. 호스트를 GPU 로 옮기는 비용, 모듈 경계 비용은 재지 않음 | `zanna2025_hybrid_physics_ai_ocean.pdf` |
| **Zhang et al. 2023** (JAMES) / **Perezhogin et al. 2024** (JAMES) | MOM6 CNN 와동 파라미터화, 안정한 구현 | 추론 비용 때문에 망을 1층×20뉴런으로 줄여야 했다 — 하드웨어가 과학을 제약한 사례 | `zhang2023_ml_eddy_mom6.pdf` / 유료 |
| **Sane et al. 2023** (JAMES) / **2026** (GRL) | OSBL 혼합계수 신경망 → O(10) 계수 방정식으로 환원 | 환원의 동기가 "5~10 % 비용". 폐쇄가 스텝의 몇 % 인지는 재지 않음 | 유료 |
| **Christopoulos 2024**, **Han 2025** (JAMES), **CondensNet 2025** | 대기 폐쇄의 온라인 학습·물리 제약·+4 K 일반화 | 모듈 기량. 비용 구조 없음 | 유료 |
| **Yu et al. 2023**, *ClimSim-Online* (arXiv 2306.08754) | 하이브리드 ML-물리 에뮬레이션 프레임워크·데이터셋 | 대기 · 커플링 비용 미측정 | `climsim_online_2023.pdf` |
| **Partee et al. 2021**, *SmartSim* (arXiv 2104.09355) / **Atkinson et al. 2025**, *FTorch* (JOSS) | CPU 호스트 ↔ 외부 추론의 두 결합 방식 | 결합 방식의 비용을 같은 코어에서 배치별로 대조하지 않음 — **우리 E17 이 그 경계 비용을 잰다** | `smartsim_ocean_ml_2021.pdf`, `ftorch_joss_2025.pdf` |
| **Meunier et al. 2025**, *differentiable Veros* (arXiv 2511.17427) | JAX 로 완전 미분가능 해양모델 | 미분가능성의 비용(속도) 없음 | `veros_differentiable_2025.pdf` |
| **Bonev et al. 2025**, *FourCastNet 3* | 단일 GPU 60일 앙상블 예보 | 초기장은 수치모델·자료동화가 만든다 | `fourcastnet3_2025.pdf` |
| **Falasca 2026**, **Rucker 2026**, **Stanley-Clamp 2026** | 데이터 기반 에뮬레이터의 강제응답·OOD 한계 | — (우리 배경: 시뮬레이션은 수치모델의 몫) | `falasca2026_*.pdf`, `rucker2026_*.pdf`, `stanleyclamp2026_*.pdf` |

## 1c. GPU 이식·정밀도·성능이식성 (B, C) — 2026-09-16 추가

| 문헌 | 한 것 | **하지 않은 것** | PDF |
|---|---|---|---|
| **Bishnu et al. 2023** (GMD 16, 5539) | MPAS-Ocean 을 Julia 로, CPU/GPU·MPI 대 Fortran | 한 스킴, 정확도 축 없음, 답 일치를 전제 | `bishnu2023_mpas_julia_vs_fortran.pdf` |
| **Tintó Prims et al. 2019** (GMD 12, 3135) | NEMO/ROMS 의 변수별 fp32 허용 여부 탐색 | 정밀도를 **속도 축과 교차**하지 않음; 타원 솔버의 정밀도 붕괴(우리 N41)·mixed 의 카드별 이득 없음 | `tinto2019_mixed_precision_nemo_roms.pdf` |
| **Omega v0.1.0** (GMD 19, 3569, 2026) | E3SM 새 해양모델, YAKL/Kokkos 성능이식성 | 한 구현 | `omega_v010_gmd_2026.pdf` |
| **icon-exclaim** (GMD 19, 713) / **ICON-GPU 운용** (GMD 19, 755, 2026) | DSL 이식, 운용 NWP GPU 성능 | 대기 · 한 구현 · 한 카드 | `icon_exclaim_dsl_gmd_2026.pdf`, `icon_gpu_operational_gmd_2026.pdf` |
| **Koldunov et al. 2026**, FESOM2 LLM 이식 (arXiv 2606.11356) | Fortran→C→C++/Kokkos | 이식 경험. 스킴·정확도 축 없음 | `fesom2_llm_port_kokkos_2026.pdf` |
| **Shan et al. 2024** (arXiv 2404.04441) | 스텐실 커널의 프로그래밍 모델 비교 | 커널. 전체 스텝·물리 없음 | `stencil_programming_models_gpu_2024.pdf` |
| **Porter & Heimbach 2024** (SP preprint) | 해양예보 GPU 기술 개관 | 개관 | `porter_heimbach_gpu_ocean_forecasting_2024.pdf` |
| **László, Giles, Appleyard 2016** (TOMS 42) | 배치 삼중대각 GPU 해법 | 우리 연직 음해 확산의 해법 근거 | `laszlo_giles_tridiagonal_toms.pdf` |
| **Zhang et al. 2016**, SCHISM (OM 102) / **Delft3D** 기능명세 | semi-음해 비정형·ADI 의 실무 모델 | — (스킴 배경) | `schism_zhang2016.pdf`, `delft3d_functional_spec.pdf` |
| **Wei et al. 2024**, LICOM3-Kokkos (FGCS) | 성능이식성 | 유료 — 서지만 | — |

## 2. 시간적분·순압 모드 (A)

| 문헌 | 한 것 | **하지 않은 것** | PDF |
|---|---|---|---|
| **Shchepetkin & McWilliams 2005**, *ROMS: a split-explicit, free-surface…* (Ocean Modelling 9, 347) | split-explicit 의 표준. 순압 부분스텝 + **시간필터**가 본체임을 보임 | 단일 모델·CPU 시대. 하드웨어 축 없음 | `roms_shchepetkin_mcwilliams_2005.pdf` |
| **Shchepetkin & McWilliams 2004** (ECMWF) | mode-splitting 오차를 줄이는 새 시간적분 | 상동 | `roms_timestepping_ecmwf_2004.pdf` |
| **Kang et al. 2021**, *A Scalable Semi-Implicit Barotropic Mode Solver for MPAS-Ocean* (JAMES, 10.1029/2020MS002238) | **semi-음해 vs split-explicit 순압 solver 직접 비교** — 우리 A 축과 가장 가까움. "거의 같은 정확도, 더 나은 확장성" | **"거의 같은 정확도"를 전제로 속도를 비교**한다. 허용오차를 바꿔 가며 승자가 뒤집히는지는 묻지 않는다. GPU·프레임워크 축 없음 | `mpas_semiimplicit_barotropic_kang2021.pdf` |
| **Demange et al. 2019**, *Stability analysis of split-explicit free surface ocean models* (JCP 398) | 깊이무관 순압 모드 근사의 안정성 해석 | 해석. 비용-정확도 측정 아님 | 유료 |
| **Casulli & Cattani 1994**, *Stability, accuracy and efficiency of a semi-implicit method for 3D shallow water* | θ-method 의 원전 | 1994. 하드웨어 축 없음 | 유료 |

## 3. 물리 (폐쇄·이류·상태방정식)

| 문헌 | 역할 |
|---|---|
| **Gaspar, Grégoris & Lefevre 1990** (JGR 95, 16179) | 1-방정식 TKE 폐쇄 — 우리 v0.6 폐쇄의 원형 |
| **Wagner et al. 2023**, *Formulation and calibration of CATKE* (arXiv 2306.13204) | GPU 를 염두에 둔 1-방정식 폐쇄 설계. "공간 국소성 때문에 GPU 최적화에 적합"이라는 주장 — **우리는 그 주장을 다섯 구현에서 수치로 잰다** (`catke_2023_closure.pdf`) |
| **Large, McWilliams & Doney 1994** | KPP — 대안 폐쇄 (구현하지 않음, 한계로 명시) |
| **Roquet et al. 2015**, *Accurate polynomial expressions for the density of seawater* (Ocean Modelling 90) | polyTEOS10-bsq — 우리 EOS 축 |
| **Leonard 1979 (QUICK)**, **Sweby 1984 (TVD)** | 3차 상류·제한자 — 우리 이류 축 |
| **Beckmann & Haidvogel 1993** | 해산 위 가짜유속 시험 (V5-3) |
| **Adcroft, Hill & Marshall 1997** | 부분셀(partial cell) |
| **NEMO-SI3 수직혼합 민감도 2024** (GMD 17, 7445) | 폐쇄 선택이 결과를 바꾼다는 물리측 근거 (`nemo_si3_vertical_mixing_2024.pdf`) |

## 4. 방법론 (성능 보고 규약)

| 문헌 | 역할 |
|---|---|
| **Hoefler & Belli 2015**, *Scientific benchmarking of parallel computing systems* (SC15) | 반복·중앙값·불확실성 보고 규약 — 우리 R7 의 근거 |
| **Williams, Waterman & Patterson 2009** (roofline) | 보조지표의 해석 틀 — 우리 R8 |
| **Balaji et al. 2017**, *CPMIP* (GMD 10, 19) | 기후모델 성능 지표 표준(SYPD, coupled cost) — 우리 성능지표(docs/26)와 비교 대상 |

---

## 5. 우리만의 기여 — 다섯 가지 (근거는 `docs/33_contribution.md`)

1. **고정 오차에서의 순위 역전을 측정했다.** 허용오차 10⁻¹ 에서는 semi-음해/split 이 4~9배 빠르고,
   10⁻³ 에서는 세 케이스 모두 양해법이 이긴다. 문헌의 "거의 같은 정확도" 전제(Kang 2021)와
   "N배 빠르다"(GPU 이식 논문들)를 **하나의 평면에서 만나게 한 결과**.
2. **물리 완전성이 하드웨어 순위를 바꾼다.** 같은 코드에 TKE 폐쇄 + 3차 TVD 이류를 켜면 CPU 는
   스텝당 2~4배, GPU 는 1.1~2배 비싸진다 → **dycore 만으로 한 GPU 벤치는 이득을 구조적으로 과소평가한다.**
   (다섯 구현 × 두 물리 수준 × 세 장치에서 답이 비트 비교 가능해야 할 수 있는 실험.)
3. **대격자에서는 연산이 아니라 용량이 승자를 정한다.** 1.2×10⁸ 셀에서 32 GB 카드는 관리메모리
   (OpenACC)만 돌고 그마저 같은 노드 CPU 보다 느리다; CUDA·JAX 는 OOM; 80 GB H100 은 21배 빠르다.
4. **프레임워크가 실행 가능성의 상한을 만든다.** JAX 는 한 스텝 전체를 하나의 jit 함수로 컴파일하며
   중간버퍼로 28 GiB 를 요구해 80 GB 카드에서도 2000² 를 못 돈다 (같은 알고리즘 CUDA 는 45.9 GB).
5. **거친 지형이 타원 solver 의 상식을 뒤집는다.** 자유표면 Helmholtz 의 계수대비가 커지면 다중격자
   V-cycle 이 4→166 회로 폭증해 PCG 가 2.4배 빨라진다 — 매끈한 지형에서는 반대(다중격자 5.7배).

6. **경험식을 AI 모듈로 떼어내려는 관점의 비용 구조**(docs/37·38, docs/02 L14): 폐쇄가 스텝의 몇 %인가(21.8~43 %), 그것을 바꿀 때의 값어치(0.93~2.72배), 호스트를 거치는 모듈 경계가 폐쇄의 몇 배인가(fp32 에서 2.1~2.7배), 그리고 fp64 코어 + fp32 폐쇄가 카드에 따라 1.1~1.95배를 돌려준다는 것. 하이브리드 문헌은 모듈의 **기량**을, GPU 이식 문헌은 코어의 **속도**를 따로 보고하고, 이 교차를 잰 연구는 없다.
