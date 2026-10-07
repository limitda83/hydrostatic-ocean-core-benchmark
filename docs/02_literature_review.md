# 문헌 기반 연구 포인트 (Literature-Grounded Research Points)

> 조사일: 2026-09-09. 각 항목은 **"문헌이 말하는 것" → "이 프로젝트가 취해야 할 포인트"** 형식.
> ⚠️ 표시는 원문 전문을 아직 확보하지 못해 **1차 출처 확인이 필요한** 항목.

---

## L1. Semi-음해법의 정본 — Casulli 계열

**문헌이 말하는 것.**
Casulli (1990/1992) 및 Casulli & Cattani (1994)가 3D 천수방정식의 semi-음해 유한차분법을 정립했다.
핵심은 **운동량방정식의 압력경사항**과 **연직적분 연속방정식의 유속항**을 θ-method로 이산화하는 것.
그 결과 하나의 큰 선형계가 생기지만, 이는 **연직 방향 삼중대각계 다수 + 수평 5-대각계 하나**로
형식적으로 분해되며, **모든 부분계가 대칭 양정치(SPD)** 다. 저자들은 "안정성을 확보하는
**최소한의 음해성(minimal degree of implicitness)**"을 채택해 최소 비용으로 최대 효율을 얻었다고 기술한다.

**취해야 할 포인트.**
- 이 구조(**연직 삼중대각 × 수평 SPD 5-대각**)가 이 프로젝트 GPU 설계의 전부다.
  연직 삼중대각은 100×100 = 10,000개의 **독립 배치 문제** → GPU에 완벽히 맞는다.
  수평 5-대각(자유수면 Helmholtz)은 **전역 결합** → GPU의 진짜 병목. **RQ2가 여기서 나온다.**
- "최소한의 음해성"은 규범적 주장이다. 이 프로젝트는 이를 **파레토 곡선으로 검증**한다.

---

## L2. θ (implicitness factor) — 강건성 vs 정확도의 교환

**문헌이 말하는 것.**
θ-method의 유효 범위는 θ ∈ [1/2, 1]. θ = 0.5는 Crank–Nicolson으로 **2차 정확·무감쇠**지만
중립안정이라 격자스케일 잡음이 남고, θ = 1은 후방오일러로 **1차 정확·강한 감쇠**다.
SELFE/SCHISM 계열은 "2차 Crank–Nicolson, 즉 implicitness factor 0.5, **실무에서는 강건성을 위해
0.5보다 약간 큰 값을 사용**"이라고 명시한다. ⚠️ 0.6 권장값의 1차 출처(Zhang et al. 2016 본문) 확인 필요.

**취해야 할 포인트 — 이 프로젝트의 가장 명확한 기여 지점.**
> θ = 0.55 / 0.6이 "강건성을 위해" 지불하는 **정확도 대가를 정량화한 공개 자료가 사실상 없다.**
> V1(관성중력파 해석해)에서 θ를 0.50 → 1.00으로 스윕하며
> `order_p`, `amp_ratio`, `phase_err` 를 측정하면, θ 선택이
> **"dt를 얼마나 키울 수 있나" vs "주기당 몇 % 감쇠하나"** 의 명시적 교환곡선으로 나온다.
> 이건 통제된 해석해 위에서만 깨끗하게 얻어지고, 정확히 우리 하네스가 하는 일이다.

---

## L3. Split-explicit vs Semi-implicit — 결론이 갈린다

**문헌이 말하는 것.**
- 구조 차이: split-explicit은 빠른(순압)/느린(경압) 모드를 **다른 dt로 둘 다 양해적으로**,
  semi-implicit은 **같은 dt를 쓰되 빠른 파를 음해적으로** 다룬다.
- E3SM 해양모델 semi-implicit 순압 solver: 기존 split-explicit subcycling 대비
  **1.4–2배 우수한 계산성능**, 정확도 저하 없음.
- 반면 split-explicit solver가 **위상오차·소산이 더 낮고**, 전 워크로드에서 **런타임과 병렬확장성이 더 좋다**는
  보고도 존재한다.
- split-explicit은 조건부 안정이며 모드 분리로 인한 **불일치(inconsistency) 보정**이 필요하다.
  semi-implicit은 경압과 같은 dt를 쓸 수 있어 두 모드가 내적으로 일관된다.

**취해야 할 포인트.**
- **문헌이 서로 반대 방향을 가리킨다** → 하드웨어·문제크기·해법 선택에 따라 결론이 뒤집힌다는 뜻.
  이 프로젝트가 답할 가치가 있는 이유이자, "우리 결론은 우리 조건에서만 유효"하다고
  범위를 명시해야 하는 이유.
- split-explicit을 축 A의 정식 수준으로 반드시 포함할 것(P1). 빠진 비교는 무의미.
- 비교 시 **병렬확장성(전역 reduction 횟수)** 을 반드시 계측 — 여기가 갈림길이다.

---

## L4. Delft3D — ADI 계열의 함정

**문헌이 말하는 것.**
Delft3D-FLOW는 운동량·연속방정식에 **완전 음해 ADI(Alternating Direction Implicit)** 를 적용해
큰 dt를 허용하고, 강건한 wetting/drying을 갖는다. 그러나 **큰 dt에서 수치감쇠(numerical attenuation),
기생진동, 파 전파 재현 열화** ('ADI-effect')가 보고된다. 곡선격자 사용으로 이를 최소화한다.

**취해야 할 포인트.**
- ADI는 축 A의 선택적 수준(P1 후반). ADI의 '큰 dt = 좋다'는 주장이 **파레토 곡선 위에서
  실제로 어디에 놓이는지** 보이는 것이 바로 이 실험의 요지.
- 방향분할(directional splitting) 오차는 V1 위상오차 지표에 그대로 잡힌다.

---

## L5. 검증(Verification)은 "수렴차수 측정"이지 그림 비교가 아니다

**문헌이 말하는 것.**
- Bishnu, Petersen, Quaife, Schoonover (2024, *JAMES* 16(4), doi:10.1029/2022MS003545)는
  순압 solver 검증용 테스트 스위트를 제시: **비분산 연안 켈빈파, 분산성 관성중력파,
  분산성 행성파/지형성 로스비파, 순압 조석, 비선형 제작해(manufactured solution)**.
  각 케이스에 대해 **수렴 연구를 수행해 이론 예측 수렴률과 일치함을 확인**한다.
- COMODO 그룹(NEMO, ROMS_AGRIF, MARS, HYCOM 등 개발자 연합)은
  **해석적 초기조건 + 커널의 소수 요소만 겨냥**하는 테스트를 원칙으로 삼고,
  연속·운동량방정식 / 추적자 이류확산 / 연직좌표 / 시간적분을 **따로, 그 다음 함께** 시험한다.
  복잡도가 커지면 내부파·불안정 경압제트(Soufflet et al. 2016)로 확장한다.

**취해야 할 포인트.**
- `docs/01_experiment_design.md` §4의 V1–V11 스위트는 이 두 출처의 구성을 그대로 채택한 것이다.
- **원칙: 하나의 케이스는 하나의 커널 요소만 겨냥한다** (RULES.md R5의 근거).
- 백엔드 포팅 검증(R2)의 통과 기준도 "그림이 같다"가 아니라 **`ref` 대비 상대 L2 임계값**이다.

---

## L6. 가짜 등밀도혼합(spurious mixing) — 3D 정확도의 진짜 지표

**문헌이 말하는 것.**
Ilıcak et al. (2012, *Ocean Modelling*)은 GOLD/MITgcm/MOM/ROMS를 대상으로,
**기준위치에너지(Reference Potential Energy, RPE)** 의 변화로 가짜 등밀도수송을 정량화했다.
핵심 결과: 가짜 등밀도수송은 **수평 격자 레이놀즈수 Re_Δ 에 비례**하며 **Re_Δ < 10 에서 크게 감소**한다.
또한 기존 연구가 이류 스킴에 집중한 데 반해 **운동량 폐합(momentum closure)** 의 역할이 크다는 점을 밝혔다.
이후 RPE 기반 진단은 해양순환모델의 표준 진단이 되었다.

**취해야 할 포인트.**
- 3D(P2)로 가면 "해석해 L2 오차"만으로는 정확도를 말할 수 없다. **V8 lock exchange + RPE 증가율**을
  3D 정확도의 1차 지표로 삼는다.
- `Re_Δ` 를 모든 3D run의 metrics에 자동 기록하고, 스킴 비교 시 **Re_Δ 를 맞춰서** 비교한다
  (안 맞추면 이류 스킴 차이가 시간적분법 차이로 오독된다).
- 시간적분법(θ)이 RPE 드리프트에 미치는 영향은 덜 조사된 영역 → 부수적 기여 가능.

---

## L7. GPU 포팅: OpenACC vs CUDA — 성능만이 축이 아니다

**문헌이 말하는 것.**
- NVIDIA는 NEMO v4.2.0 GYRE_PISCES(ORCA 1/2) 벤치마크를 **OpenACC + Unified Memory**로 가속했다.
  이 벤치마크는 **메모리 대역폭 지배(memory bandwidth-bound)** 이며, 추적자 확산·이류가 대상.
  Unified Memory로 **명시적 데이터 이동 관리를 제거**한 것이 포팅 노력 감소의 핵심.
- NEMO GPU 포팅은 여러 기관이 시도했으나 **아직 공식 소스에 통합되지 않았다.**
  OpenACC가 선호되는 이유는 소스 변경 최소화 + 미컴파일 시 CPU 코드로 그대로 동작.
- 프로그래밍 모델 비교: **CUDA는 튜닝 시 성능 우위**(명령어·메모리 세밀 제어),
  **OpenACC는 훨씬 적은 코드량으로 이식성 우위**.
- MPAS는 OpenACC 지시문 기반 실험적 포팅이 존재.

**취해야 할 포인트.**
- RQ3의 종속변수는 **시간만이 아니라 `SLOC`(코드량)** 이다. 두 축을 함께 플롯해야
  "실제로 어떤 선택이 합리적인가"에 답할 수 있다. → `docs/12_backend_matrix.md` 형식 결정.
- **Unified Memory 사용/미사용**을 OpenACC 축의 하위 수준으로 넣는다 (포팅 노력 vs 성능 교환).
- 우리 커널도 대역폭 지배일 것 → roofline 없이 GFLOP/s만 보고하면 안 된다 (L9).

---

## L8. Python도 후보다 — Veros의 반례

**문헌이 말하는 것.**
- Häfner et al. (2018, *GMD* 11, 3299): Veros — 순수 Python 해양 시뮬레이터.
- Häfner et al. (2021, *JAMES*, doi:10.1029/2021MS002717): **JAX 백엔드**로
  **CPU에서 Fortran 수준 성능**(코어 수 중간 구간에서만 ~40% 격차), **GPU에서 2–5배 높은 에너지 효율**.
  NumPy → JAX 전환은 **JIT 데코레이터 추가와 슬라이싱 문법 변경 정도**로 최소.
  0.1° 전지구 설정이 A100 16장 단일 노드에서 1.2 model-years/day (≈ Fortran 2000 CPU 상당).
- `pyhpc-benchmarks` 및 후속 비교: **stencil 벤치에서는 Numba가 최상위**,
  **JAX는 CPU/GPU 양쪽에서 일관되게 상위권**, CuPy는 구현이 쉽지만 연산집약 작업에서 상대적으로 느림
  (최적화 모드에서는 큰 개선 가능하나 코드 수정 부담이 큼).

**취해야 할 포인트.**
- **"Python은 느리다"를 사전 가정으로 두지 말 것.** Python-GPU 축(D)은 진지한 경쟁자다.
- Python-GPU 하위 수준을 **CuPy / JAX / Numba-CUDA 3종으로 고정**한다(문헌이 서로 다른 승자를 가리키므로).
  우리 커널이 stencil + 배치 삼중대각 + PCG 혼합이라 **셋 중 누가 이길지 사전에 알 수 없다** → 측정 가치.
- **에너지 효율(Joule/simulated-day)** 을 지표에 포함하는 근거가 여기서 나온다 (설계문서 §4.2 `E_sol`).
- NumPy 레퍼런스를 처음부터 **JAX로 옮기기 쉬운 형태**(순수함수, in-place 갱신 회피)로 쓴다 → P6 비용 절감.
  ⇒ `libs/core/` 코딩 제약으로 반영.

---

## L9. 대역폭 지배 커널 — roofline 없이 성능을 말하지 말 것

**문헌이 말하는 것.**
Roofline 모델은 **산술강도(arithmetic intensity, FLOP/byte)** 를 축으로 피크 연산성능과
피크 대역폭이라는 두 천장으로 달성 가능 성능을 한정한다. **stencil 커널은 산술강도가 낮아
전형적으로 메모리 대역폭 지배**다. NVIDIA Nsight Compute는 프로파일 커널의 roofline을 자동 산출한다.

**취해야 할 포인트.**
- 각 커널(운동량 갱신 / Helmholtz 행렬-벡터 곱 / 배치 삼중대각 / 추적자 이류)의
  **AI를 손으로 계산**해 두고, 측정 성능이 대역폭 천장의 몇 %인지로 보고한다.
- 이것이 T2(FP64 1/64 페널티)를 판정하는 방법이다:
  **대역폭 지배라면 fp64 → fp32 이득은 ~2배(트래픽 절반)에 그치고 64배가 아니다.**
  RTX 5090에서 이를 실측하는 것 자체가 보고 가치가 있다.

---

## L10. 혼합정밀도 — 공짜 점심이 있는 곳

**문헌이 말하는 것.**
Tintó Prims et al. (2019, *GMD* 12, 3135): "How to use mixed precision in ocean models" —
NEMO 4.0과 ROMS 3.6 대상. **대부분의 과학코드는 정밀도가 과설계(overengineered)** 되어 있다.
제안 방법은 단순하다: **변수 그룹의 정밀도를 낮추고 출력 영향을 측정**하는 것을
**분할정복 알고리즘**으로 반복해, 고정밀이 꼭 필요한 부분과 저정밀 허용 변수 집합을 자동 식별한다.
연산·메모리 지배 코드 모두에서 적은 노력으로 상당한 속도 향상.

**취해야 할 포인트.**
- 축 G(정밀도)의 `mixed` 수준은 임의로 정하지 말고 **이 분할정복 절차를 그대로 적용**한다.
  우리 코드는 작아서 전수 탐색에 가깝게 할 수 있다 → 방법론적으로 깨끗한 결과.
- 저정밀이 특히 위험한 곳을 사전 가설로 명시: (a) 자유수면 η (작은 값 + 큰 배경),
  (b) PCG의 내적(전역 reduction 누적오차), (c) 압력경사항의 상쇄(pressure gradient cancellation).
  → **가설을 먼저 쓰고 측정**한다.

---

## L11. 배치 삼중대각 해법 — 연직 음해항의 GPU 매핑

**문헌이 말하는 것.**
cuSPARSE는 **cyclic reduction 계열의 `gtsv2StridedBatch`** 를 제공한다.
PDE의 균일격자처럼 잘 조건화된 문제에서는 CR/pivoting이 **표준 Thomas 알고리즘 대비 불필요한 오버헤드**를
가진다는 지적이 있으며, PCR(parallel cyclic reduction), PTA(parallel Thomas) 등
여러 GPU 구현이 비교 연구되어 왔다 (Giles et al.; Tridigpu 등).

**취해야 할 포인트.**
- 우리 문제는 **10,000개(=100×100) × 크기 30** 의 배치 삼중대각. 배치 수는 많고 각각은 매우 짧다.
  → **thread-per-system Thomas** 가 유리할 가능성이 높다(각 시스템이 32 미만이라 CR의 병렬성 이점이 작다).
  CUDA 축(P5)에서 **cuSPARSE gtsv2StridedBatch vs 자체 thread-per-column Thomas** 를 비교한다.
- 메모리 레이아웃이 결정적: 열(column)별 연속(`k` 최내측)이면 Thomas가 유리하지만
  수평 stencil은 `i` 최내측을 원한다 → **레이아웃 충돌**. 이 교환을 명시적으로 측정할 것.
  (RULES.md의 `[k,j,i]` 규약과 Fortran 전치 규칙이 여기서 나온다.)

---

## L12. Eulerian–Lagrangian 이류(ELM) — SCHISM의 교환

**문헌이 말하는 것.**
SCHISM/SELFE는 **semi-implicit 유한요소/유한체적 + Eulerian–Lagrangian 알고리즘**으로
정역학 Navier–Stokes를 푼다. ELM은 이류 CFL 제약을 제거해 큰 dt를 허용한다.
⚠️ ELM의 수치확산·비보존성에 대한 정량 비교는 1차 출처 확인 필요.

**취해야 할 포인트.**
- 축 C에 `elm` 을 포함하되, **V7(가우시안 회전이류)에서 질량보존 오차와 수치확산을 반드시 측정**한다.
  "큰 dt를 쓸 수 있다"는 이점이 보존성 손실로 상쇄되는 지점을 찾는 것이 목적.
- ELM은 back-tracking이 불규칙 메모리 접근을 유발 → **GPU에서 특히 불리할 것**이라는 가설을 세우고 측정.
  (RQ2의 두 번째 사례: 알고리즘–하드웨어 상호작용)

---

## L13. **선행연구 점검 — 같은 실험을 한 논문이 있는가?** (2026-09-11 조사)

"해양모델 코어를 하나 고정해 놓고 **시간적분 스킴 × 언어/프레임워크 × 하드웨어**를
교차 실험한 연구"가 이미 있는지 직접 확인했다. **부분적으로 있고, 전체는 없다.**

### L13.1 언어·프레임워크 축을 다룬 연구 (있음)

| 연구 | 무엇을 비교했나 | 우리와 겹치는 부분 / 다른 부분 |
|---|---|---|
| **Bishnu et al. 2023, GMD 16, 5539** — Julia vs Fortran-MPI, MPAS-Ocean | Julia(단일코어/GPU/MPI) vs Fortran-MPI. **Julia가 NumPy보다 13배 빠름**, Julia-MPI는 낮은 코어수에서 Fortran-MPI와 동일, 높은 코어수에서 2배 빠름~2배 느림 | **백엔드 축은 겹친다.** 단, **천수방정식(2D)** 이고 스킴은 하나. 스킴×하드웨어 상호작용 없음 |
| **Häfner et al. 2021, JAMES (Veros)** | NumPy vs JAX, CPU vs GPU, 전지구 원시방정식 | **3D 코어 + 백엔드 축.** 스킴 비교 없음. 우리 L8의 근거 |
| **Wei et al. 2024, FGCS** — LICOM3 + Kokkos | LICOM3의 **Fortran / OpenMP / OpenACC / HIP / CUDA / Kokkos** 6종 비교. 1° 해상도에서 Kokkos가 raw CUDA 대비 1.9배, HIP 1.2배, OpenMP 1.1배 | **프레임워크 축이 우리보다 넓다** (Kokkos·HIP 포함). 스킴은 고정 |
| **Omega v0.1.0, GMD 19, 3569 (2026)** | E3SM 차세대 해양모델, C++/Kokkos, CPU/GPU 성능 | 단일 코드베이스 이식성. 스킴 비교 아님 |
| **FESOM2 LLM 포팅 (arXiv 2606.11356)** | Fortran → C → C++/Kokkos 이식 경험 | 이식 비용 축 |

> **결론:** "언어/프레임워크 × 하드웨어" 축은 **이미 잘 다뤄져 있다.**
> 특히 LICOM3-Kokkos 연구는 우리보다 프레임워크가 많다(우리에겐 Kokkos·HIP이 없다).
> 우리 백엔드 결과(CUDA가 OpenACC보다 2.7~5.4배)는 **새롭지 않고, 재확인에 가깝다.**

### L13.2 스킴 축을 다룬 연구 (있음, 그러나 CPU만)

| 연구 | 내용 |
|---|---|
| **Kang et al. 2021, JAMES** — MPAS-Ocean 확장형 semi-음해 순압 solver | semi-음해 vs split-explicit subcycling **직접 비교**. semi-음해가 1.4~2배 우수하다고 보고 |
| **Stability analysis of split-explicit free surface ocean models, JCP (2019)** | split-explicit의 안정성·위상오차 분석 |

> 둘 다 **CPU 전용**이다. 우리가 측정한 "CPU에서는 거의 무승부, GPU에서는 2.5~9배 차이"라는
> 비대칭은 이 논문들이 볼 수 없는 축이다.

### L13.3 **우리 결론을 부분적으로 앞선 연구 — Silvestri et al. 2025**

**Silvestri et al. (2025), *JAMES* 17, e2024MS004465 — "A GPU-Based Ocean Dynamical Core
for Routine Mesoscale-Resolving Climate Simulations"** (Oceananigans.jl).

- A100 64장에서 8 km 준전지구 해양을 **하루에 10 모델년** 적분.
- 순압 solver가 **통신 집약적**이며 전형적 IPCC급 시뮬레이션 비용의 **40~60%** 를 차지한다고 지적.
- 핵심 문장: **"GPU에서는 순압 모드의 시간이산화를 정교하게 만들 이유가 없다"**,
  그리고 **"부분스텝 수가 성능에 사실상 무관하다"**.
- 기법: tendency 커널을 inner/outer로 분할, **순압 halo를 subcycle 수만큼 키워
  순압 통신을 연직 음해확산 뒤에 숨김**.

> **이것이 우리 `docs/22` 결론과 같은 방향이다.**
> 다만 그들은 **split-explicit을 선택한 뒤 "부분스텝이 싸다"고 보고**했고,
> 우리는 **두 스킴을 같은 코어에서 나란히 측정해 교차점(CFL 8, 스텝당 208 PCG 반복)을 수치화**했다.
> 즉 우리 기여는 "GPU에서 split-explicit이 유리하다"는 **주장**이 아니라
> **"얼마나, 어디서부터, 왜"** 에 대한 통제된 측정이다.

### L13.4 그래서 남는 빈틈 (이 프로젝트의 실제 기여)

1. **정확도를 고정한 채** 스킴을 바꿔 비용만 비교한 연구가 없다.
   우리 `docs/22` 표는 두 스킴의 L2가 5자리까지 같다 — 순수 비용 비교다.
2. **수렴차수 검증을 통과한 코어** 위에서 성능을 잰 연구가 드물다.
   Bishnu 2024는 검증 스위트를 주지만 성능이 없고, 성능 논문들은 수렴차수를 보고하지 않는다.
3. **θ(implicitness factor)의 정확도 대가**를 정량화한 공개 자료가 여전히 없다 (L2).
4. **스킴 순위가 하드웨어에 따라 뒤집히는 지점**을 수치로 제시한 사례가 없다.

> **정직한 자기평가:** 백엔드 비교(항목 L13.1)는 새롭지 않다.
> 새로운 것은 **스킴×하드웨어 상호작용을 검증된 코어 위에서 정확도 고정으로 측정한 것**이며,
> Silvestri 2025가 같은 방향을 이미 정성적으로 보고했다는 점을 명시해야 한다.

### L13.5 spec v0.5 이후 추가된 빈틈 (2026-09-11 갱신)

v0.5 실험(docs/24–26)이 끝난 뒤, 위 목록에 더해 선행연구에서 찾지 못한 것들:

5. **보고 방법론의 재량폭을 정량화한 사례가 없다.** docs/25 §2 는 같은 측정이
   **1.95배와 322.2배로 동시에 읽힘**을 보인다(165배; 2026-09-13 안정화 재측정). GPU 해양모델 논문들은 각자 하나의
   기준선을 고르지만, **그 선택이 결론을 얼마나 움직이는지 재어 보고한 곳은 없다.**
   이것만으로도 독립적인 방법론 기여가 된다.
6. **타원형 해법 축을 GPU 위에서 지형과 함께 스윕한 사례가 없다.** Kang 2021 은 CPU 에서
   semi-음해 vs split-explicit 를 비교했고 Silvestri 2025 는 GPU 에서 순압 부분스텝을
   다뤘지만, **해법(PCG/RBGS/다중격자) × 지형 거칠기 × CFL × 백엔드 × 하드웨어**를
   한 코어 위에서 교차한 행렬은 없다. docs/25 §3–4 가 그 행렬이고, 결론은
   **"큰 시간간격의 비용은 스킴이 아니라 해법이 결정한다"** 이다.
7. **동일 커널의 소비자 GPU vs 데이터센터 GPU 비교가 알고리즘별로 뒤집힌다는 보고가 없다.**
   docs/25 §8: H100 이 PCG 에서 RTX 5090 에 지고 다중격자에서 이긴다. 하드웨어 세대 비교를
   **단일 숫자로 말할 수 없다**는 것을 반례로 보인 셈이다.
8. **압력경사 스킴과 연직좌표가 교란(confound)된다는 지적이 명시적이지 않다.** 해산 시험은
   보통 "σ 좌표의 압력경사 오차"를 보이는 데 쓰이는데, docs/24 §3 은 **보정을 켜면 σ 가
   z-level 부분셀보다 낫다**는 것을 같은 코드·같은 격자에서 보인다. 즉 문헌의 통설은
   *보정 없는 σ* 에 대한 진술이다.
9. **fp32 가 Krylov 를 돕고 다중격자를 망가뜨린다는 대조 측정이 없다.** docs/25 §9.
   혼합정밀도 해양모델 논문(L10)은 이류·확산 커널을 다루지만 **타원형 해법별 감수성**은
   다루지 않는다. 다중격자가 fp64 최적이자 fp32 최악이라는 사실은 실무 결정에 직결된다.

> **여전히 정직하게:** 위 항목들은 **커널 수준**의 결과다. 완전한 해양모델의 실런에서
> 같은 순위가 유지되는지는 v0.5 물리를 컴파일 백엔드로 이식한 뒤에야 말할 수 있다.

---

## L14. **AI 모델 시대에 수치모델이 남는 자리 — 그리고 그것이 GPU 를 요구하는 이유** (2026-09-15 조사)

이 절은 이 프로젝트가 "왜 해양 원시방정식 코어를 GPU 로 옮기는 비용을 재는가"의 배경이다.
결론부터: **예보는 AI 가 가져가고 있지만, 시뮬레이션(재현 + 변화실험)은 가져가지 못하고 있다.
그리고 시뮬레이션을 AI 로 점진 전환하는 유일하게 실현된 경로 — 경험식(closure)의 모듈 교체 —
는 호스트 모델이 어디에 상주하느냐에 따라 비용이 100 배 달라진다.**

### L14.1 예보: AI 가 이미 이겼고, 그 입력은 수치모델이다

- **FourCastNet 3** (Bonev et al., NVIDIA, 2025): 구면 신경연산자(SFNO)의 확률적 앙상블판.
  1024+ GPU 로 학습하고 **단일 GPU 에서 0.25°·6시간 간격 60일 전지구 예보를 4분 미만**에 낸다.
  선도 수치 앙상블을 능가하고 확산모델 대비 8~60배 빠르다.
- 그러나 이 계열(FCN3·GraphCast·Pangu·AIFS)은 **ERA5 재분석으로 학습하고 ERA5 또는 운용
  해석장으로 초기화**한다. 초기장을 만드는 것은 자료동화이고, 자료동화의 배경장은 수치모델이다.
  AI 예보모델의 존재 자체가 수치모델의 존재를 전제한다(HealDA 2026 은 그 초기오차 민감도를
  다시 확인했다).

### L14.2 시뮬레이션: AI 가 아직 못 하는 것이 여기 있다

수치모델의 값어치는 "빠른 예보"가 아니라 **과거를 정밀 재현한 뒤 조건을 바꿔 보는 것**
(변화실험·민감도·귀인)이다. 데이터 기반 에뮬레이터는 여기서 구조적으로 막힌다.

- **강제응답(forced response)을 재현하지 못한다** — 데이터 기반 기후 에뮬레이터의 개념적
  한계로, 인과 연구에 쓰기 어렵다는 것이 정면으로 분석되었다(Phys. Rev. Research, 2026).
- **분포 밖(OOD)에서 학습 기후로 되돌아간다** — 더 따뜻한 상태로 초기화해도 자기회귀 롤아웃
  동안 학습 기후값으로 표류하며 부여한 열역학 아노말리를 소산시킨다(ACE2/NeuralGCM 벤치마크,
  2026). "No Epoch Like the Present"(2026)는 ML 에뮬레이터 연구의 대부분이 **OOD 평가 자체를
  하지 않는다**고 지적한다.
- 반사실(counterfactual) 실험은 시도되고 있으나, 초기조건의 각인을 완전히 지우지 못하고
  학습 분포를 정확히 재현하지 못하는 한계가 보고된다(Faranda 계열, 2024–2025).

> 즉 **예보 = AI, 시뮬레이션 = 수치모델**이라는 분업이 당분간 유지되며, 그렇다면 수치모델을
> 버리는 것이 아니라 **수치모델 안으로 AI 를 넣는 것**이 현실적 경로다.

### L14.3 실현된 경로는 하나 — 경험식의 모듈 교체

보존을 담당하는 역학 코어는 그대로 두고, **경험식(파라미터화)만 학습 모듈로 바꾸는** 하이브리드는
이미 여러 모델에서 온라인으로 돌고 있다.

- **해양 연직혼합**: Sane et al. (2023, JAMES)은 OSBL 와동확산계수를 신경망으로 예측했고,
  Sane et al. (2026, GRL)은 그 신경망과 같은 기량을 내는 **O(10) 계수의 간결한 방정식**을
  역으로 뽑아냈다 — 그 동기가 **"대형 신경망을 쓰는 전지구 적분은 5~10 % 더 비싸다"** 였다.
- **대기 대류·응결**: Christopoulos et al. (2024)은 EDMF 내부의 유입(entrainment) 폐쇄를
  1D 컬럼 GCM 안에서 **온라인 학습**했고, Han et al. (2025)의 NCAM 은 현재기후만으로 학습한
  신경망을 넣고도 **+4 K 기후에서 10년을 총에너지·가강수량 표류 없이** 돌렸다. CondensNet
  (2025)은 물리 제약을 적응적으로 걸어 장기 안정성을 얻었다.
- **해양 중규모 와동**: Zhang et al. (2023, JAMES)이 MOM6 에 CNN 파라미터화를 넣었고,
  Perezhogin et al. (2024)이 안정한 구현을 만들었다.
- 공통 교훈: **오프라인 학습 → 온라인 표류/불안정**이 이 분야의 표준 실패이고, 해법은
  온라인 학습 또는 물리 제약이다. 보존을 코어가 담당한다는 구조가 그 안전망이다.

### L14.4 **그 모듈이 어디서 도는가가 100 배를 가른다** (이 프로젝트의 직접 근거)

Zanna et al. (2025, *A Framework for Hybrid Physics-AI Coupled Ocean Models*)은 MOM6/OM4 에
여러 학습 모듈을 넣고 비용을 함께 보고했다. MOM6 는 **Fortran·CPU 상주이고 미분가능하지 않다.**

| 모듈 | CPU 상주 호스트에서의 비용 |
|---|---|
| 작은 ANN (다층 퍼셉트론) | 사실상 무시 가능 |
| 해빙 편의보정 CNN (~10⁵ 파라미터) | **0.3 % 감속** |
| 복잡한 CNN | **코어보다 "한 자릿수(order of magnitude) 느림"** |
| 같은 복잡한 CNN, **GPU 에서** | **코어 비용의 10 %** |

같은 모듈이 **CPU 에서 코어의 ~10배, GPU 에서 코어의 ~0.1배** — 배치 하나로 **약 100 배**가
갈린다. 이것이 실제로 과학을 제약한 사례도 있다: Perezhogin et al. 은 추론 비용을 모델
런타임의 10 % 아래로 누르려고 **은닉층 1개·뉴런 20개**로 신경망을 줄여야 했고, Sane et al.
(2026)은 아예 신경망을 방정식으로 환원했다. Zhang et al. (2023)의 측정으로는 CNN 추론이
격자점당 최소 **268,005 FLOP** 이고 CPU 에서 **역학코어의 약 10배** 시간이 든다.

> **모듈의 구조(얼마나 큰 망을 쓸 수 있는가)를 과학이 아니라 호스트 모델이 도는 하드웨어가
> 정하고 있다.** 이것이 "AI 모듈을 쓰려면 수치모델이 GPU 에 상주해야 한다"는 주장의 근거다.

### L14.5 두 갈래, 그리고 현장이 이미 가는 방향

1. **CPU 호스트 + 외부 GPU 추론** — FTorch(2025, Fortran↔PyTorch 바인딩), SmartSim(HPE,
   Redis 기반 필드 교환), Infero. 바인딩 방식은 간단하지만 **호스트와 같은 자원(대개 CPU)에서
   돈다.** 필드 교환 방식은 GPU 를 쓸 수 있으나 **데이터 전송 오버헤드와 두 자원 풀의 스케줄링**
   을 떠안는다(SmartSim 문헌이 명시).
2. **GPU 상주 호스트** — Oceananigans(Julia/KernelAbstractions, GPU 네이티브, Enzyme·Reactant
   로 **미분가능**), Veros(JAX, 2025년 말 **완전 미분가능 확장**), ICON-GPU(운용 NWP,
   소켓 대 소켓 **5.5배**), Omega(E3SM, Kokkos), NEMO(PSyclone·통합메모리). 이 흐름은 지금까지
   **성능 이식성** 때문에 진행됐지 AI 모듈 때문이 아니었다 — 그러나 결과적으로 AI 모듈을
   장치 내 커널 호출로 만들 수 있는 유일한 구조다.

### L14.6 남는 빈틈 = 이 프로젝트가 재는 것

문헌은 (a) AI 모듈의 **기량**과 (b) GPU 이식의 **속도**를 각각 보고한다. 아무도
**"경험식을 모듈로 떼어내려는 관점에서 본 수치모델의 GPU 비용 구조"** 를 재지 않았다. 구체적으로:

1. 떼어낼 대상(경험식 폐쇄)이 **한 스텝의 몇 %인가** — 즉 모듈화의 상한(암달)은 얼마인가.
2. 그 경험식을 **다른 경험식으로 바꾸면** 비용이 얼마나 변하는가 — 모듈 교체의 민감도.
3. 그 비용이 **장치·정밀도·격자**에 따라 어떻게 달라지는가 — 어떤 하드웨어에서 모듈화가
   이득인가.
4. 호스트가 GPU 에 상주하지 않을 때 치르는 **경계 비용**은 얼마인가.

이 프로젝트의 docs/31·34·35·36 이 1~3 을 측정했고, 4 는 `expr/E17`(docs/38)이 측정했다 — 학습 모듈이 쓸 정밀도에서 호스트 왕복이 폐쇄의 2.1~2.7배(하한).

## 요약: 이 연구가 실제로 겨냥해야 할 지점

1. **θ의 정확도 대가 정량화** (L2) — 가장 명확한 빈틈.
2. **알고리즘–하드웨어 순위 역전** (L1·L3·L11·L12) — semi-음해법의 전역 타원형 해와 ELM의
   불규칙 접근이 GPU에서 CPU와 다른 순위를 만드는가.
3. **대역폭 지배 하에서 소비자 GPU FP64 페널티의 실효 크기** (L9·L10) — RTX 5090이라는
   구체적 하드웨어에서만 물을 수 있는 질문.
4. **성능 × 코드량 동시 보고** (L7·L8) — Fortran/CUDA/OpenACC/Python-GPU 선택의 실무적 답.
5. **검증은 수렴차수로** (L5) — 위 모든 주장의 신뢰성 기반.
6. **선행연구 대비 위치** (L13) — 백엔드 비교는 이미 잘 다뤄져 있다(LICOM3-Kokkos, Veros, MPAS-Julia).
   남은 빈틈은 **정확도를 고정한 스킴×하드웨어 상호작용**이며, Silvestri 2025가 같은 방향을
   정성적으로 앞서 보고했다.
7. **AI 모듈화 관점의 비용 구조** (L14) — 경험식 폐쇄가 스텝의 몇 %이고, 그것을 교체할 때
   비용이 장치·정밀도에 따라 어떻게 변하며, 호스트가 GPU 에 상주하지 않을 때 경계 비용이
   얼마인가. 문헌은 AI 모듈의 **기량**과 GPU 이식의 **속도**를 따로 보고할 뿐 이 교차를 재지
   않았다. docs/37 이 이 관점을 정리한다.

---

## 출처

- [Casulli & Cattani (1994), *Stability, accuracy and efficiency of a semi-implicit method for three-dimensional shallow water flow*, Computers & Mathematics with Applications](https://www.sciencedirect.com/science/article/pii/0898122194900590)
- [Casulli (1992), *Semi-implicit finite difference methods for three-dimensional shallow water flow*, IJNMF 15(6)](https://onlinelibrary.wiley.com/doi/10.1002/fld.1650150602)
- [Casulli (2014), *A semi-implicit numerical method for the free-surface Navier–Stokes equations*, IJNMF](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.3867)
- [Zhang et al. (2016), *Seamless cross-scale modeling with SCHISM*, Ocean Modelling 102, 64–81 (PDF)](https://ccrm.vims.edu/yinglong/Courses/Marsh-2017/Zhang_etal_OM_2016-SCHISMpaper.pdf)
- [SCHISM modeling system (VIMS)](https://ccrm.vims.edu/schismweb/)
- [Zhang & Baptista (2008), *SELFE: a semi-implicit Eulerian–Lagrangian finite-element model*](https://www.researchgate.net/publication/223763257_SELFE_a_Semi-implicit_Eulerian-Lagrangian_Finite-Element_model_for_cross-scale_ocean_circulation)
- [Bishnu et al. (2024), *A Verification Suite of Test Cases for the Barotropic Solver of Ocean Models*, JAMES 16(4)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003545) · [open-access preprint](https://essopenarchive.org/doi/full/10.22541/essoar.167100170.03833124/v1)
- [Eos research spotlight: *Verifying the Mathematics Behind Ocean Modeling*](https://eos.org/research-spotlights/verifying-the-mathematics-behind-ocean-modeling)
- [COMODO test-case suite / pycomodo (EGU 2016 abstract)](https://ui.adsabs.harvard.edu/abs/2016EGUGA..1813482G/abstract) · [CROCO baroclinic jet test case](https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.test_cases.jet.html)
- [Ilıcak et al. (2012), *Spurious dianeutral mixing and the role of momentum closure*, Ocean Modelling](https://www.sciencedirect.com/science/article/abs/pii/S1463500311001685)
- [E3SM: *A Semi-Implicit Barotropic Solver for the E3SM Ocean Model*](https://e3sm.org/a-semi-implicit-barotropic-solver-for-the-e3sm-ocean-model/)
- [*Stability analysis of split-explicit free surface ocean models*, JCP (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0021999119305662)
- [*A communication-avoiding implicit–explicit method for a free-surface ocean model*, JCP (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0021999115007482)
- [NVIDIA: *Less Coding, More Science: Simplify Ocean Modeling on GPUs With OpenACC and Unified Memory* (NEMO GYRE_PISCES)](https://developer.nvidia.com/blog/less-coding-more-science-simplify-ocean-modeling-on-gpus-with-openacc-and-unified-memory)
- [*GPU technologies for Ocean Forecasting* (Copernicus preprint sp-2024-32)](https://sp.copernicus.org/preprints/sp-2024-32/sp-2024-32.pdf)
- [*Evaluation of Programming Models and Performance for Stencil Computation on Current GPU Architectures* (arXiv:2404.04441)](https://arxiv.org/pdf/2404.04441)
- [Häfner et al. (2021), *Fast, Cheap, and Turbulent — Global Ocean Modeling With GPU Acceleration in Python*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717) · [PDF](https://www.gfy.ku.dk/~mjochum/veros2.pdf)
- [Häfner et al. (2018), *Veros v0.1 – a fast and versatile ocean simulator in pure Python*, GMD 11, 3299](https://gmd.copernicus.org/articles/11/3299/2018/)
- [pyhpc-benchmarks (dionhaefner)](https://github.com/dionhaefner/pyhpc-benchmarks) · [team-ocean/veros](https://github.com/team-ocean/veros)
- [*Evaluation of Alternatives to Accelerate Scientific Numerical Calculations on GPUs Using Python*](https://link.springer.com/chapter/10.1007/978-3-031-52186-7_1)
- [Tintó Prims et al. (2019), *How to use mixed precision in ocean models: NEMO 4.0 and ROMS 3.6*, GMD 12, 3135](https://gmd.copernicus.org/articles/12/3135/2019/)
- [Giles et al., *Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers* (ACM TOMS)](https://people.maths.ox.ac.uk/gilesm/files/toms_16b.pdf) · [Tridigpu (ACM TOPC)](https://dl.acm.org/doi/full/10.1145/3580373)
- [*A quantitative roofline model for GPU kernel performance estimation*, JPDC](https://www.sciencedirect.com/science/article/abs/pii/S0743731517301247) · [Roofline model overview](https://modal.com/gpu-glossary/perf/roofline-model)
- [Delft3D-FLOW functional specifications (ADI, drying/flooding)](https://content.oss.deltares.nl/delft3d4/Delft3D-Functional_Specifications.pdf)
- [Bishnu et al. (2023), *Comparing the Performance of Julia on CPUs versus GPUs and Julia-MPI versus Fortran-MPI: a case study with MPAS-Ocean*, GMD 16, 5539](https://gmd.copernicus.org/articles/16/5539/2023/)
- [Silvestri et al. (2025), *A GPU-Based Ocean Dynamical Core for Routine Mesoscale-Resolving Climate Simulations*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004465) · [MIT open access PDF](http://dspace.mit.edu/bitstream/handle/1721.1/163174/J%20Adv%20Model%20Earth%20Syst%20-%202025%20-%20Silvestri%20-%20A%20GPU%E2%80%90Based%20Ocean%20Dynamical%20Core%20for%20Routine%20Mesoscale%E2%80%90Resolving%20Climate.pdf?sequence=2&isAllowed=y)
- [Kang et al. (2021), *A Scalable Semi-Implicit Barotropic Mode Solver for the MPAS-Ocean*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002238)
- [Wei et al. (2024), *Accelerating LASG/IAP climate system ocean model version 3 for performance portability using Kokkos*, Future Generation Computer Systems](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003285)
- [*The ocean model for E3SM global applications: Omega version 0.1.0*, GMD 19, 3569 (2026)](https://gmd.copernicus.org/articles/19/3569/2026/)
- [Oceananigans.jl (CliMA)](https://github.com/CliMA/Oceananigans.jl) · [*High-level, high-resolution ocean modeling at all scales with Oceananigans* (arXiv:2502.14148)](https://arxiv.org/pdf/2502.14148)
- [*An Ocean Model Ported by a Large Language Model: FESOM2 Fortran to C to C++/Kokkos* (arXiv:2606.11356)](https://arxiv.org/html/2606.11356v1)
- [Bonev et al. (2025), *FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale* (arXiv:2507.12144)](https://arxiv.org/abs/2507.12144) · [NVIDIA Research](https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather)
- [Zanna et al. (2025), *A Framework for Hybrid Physics-AI Coupled Ocean Models* (arXiv:2510.22676)](https://arxiv.org/html/2510.22676v1) — **CPU 대 GPU 배치가 같은 CNN 모듈의 비용을 코어의 10배 대 0.1배로 가른다**
- [Zhang et al. (2023), *Implementation and Evaluation of a Machine Learned Mesoscale Eddy Parameterization into a Numerical Ocean Circulation Model*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003697) · [arXiv:2303.00962](https://arxiv.org/pdf/2303.00962)
- [Perezhogin et al. (2024), *A Stable Implementation of a Data-Driven Scale-Aware Mesoscale Parameterization*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004104)
- [Sane et al. (2023), *Parameterizing Vertical Mixing Coefficients in the Ocean Surface Boundary Layer Using Neural Networks*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003890)
- [Sane et al. (2026), *Machine Learned Equations for Vertical Mixing Coefficients in the Ocean Surface Boundary Layer*, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026GL122106) — 대형 NN 전지구 적분은 5~10 % 더 비싸다
- [Christopoulos et al. (2024), *Online Learning of Entrainment Closures in a Hybrid Machine Learning Parameterization*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004485)
- [Han et al. (2025), *A Decadal Hybrid GCM Simulation Using Deep-Learning-Based Cloud and Convection Parameterization Generalized to a Warm Climate*, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005231)
- [*CondensNet: enabling stable long-term climate simulations via hybrid deep learning models with adaptive physical constraints*, npj Clim. Atmos. Sci. (2025)](https://www.nature.com/articles/s41612-025-01269-5)
- [Atkinson et al. (2025), *FTorch: a library for coupling PyTorch models to Fortran*, JOSS](https://www.theoj.org/joss-papers/joss.07602/10.21105.joss.07602.pdf)
- [Partee et al. (2021), *Using Machine Learning at Scale in HPC Simulations with SmartSim: An Application to Ocean Climate Modeling* (arXiv:2104.09355)](https://ar5iv.labs.arxiv.org/html/2104.09355)
- [*ClimSim-Online: A Large Multi-scale Dataset and Framework for Hybrid ML-physics Climate Emulation* (arXiv:2306.08754)](https://arxiv.org/pdf/2306.08754)
- [*Probing forced responses and causality in data-driven climate emulators: conceptual limitations and the role of reduced-order models*, Phys. Rev. Research (arXiv:2506.22552)](https://arxiv.org/html/2506.22552)
- [*No Epoch Like the Present: Robust Climate Emulation Requires Out-of-Distribution Generalisation* (arXiv:2605.22248)](https://arxiv.org/html/2605.22248v1)
- [*Benchmarking Regional Thermodynamic Trends in an AI emulator, ACE2, and a hybrid model, NeuralGCM* (arXiv:2511.00274)](https://arxiv.org/html/2511.00274)
- [*Towards fully differentiable neural ocean model with Veros* (arXiv:2511.17427)](https://arxiv.org/abs/2511.17427)
- [*Operational numerical weather prediction with ICON on GPUs (version 2024.10)*, GMD 19, 755 (2026)](https://gmd.copernicus.org/articles/19/755/2026/) — 혼합정밀도 포함 소켓 대 소켓 5.5배
- [*Toward exascale climate modelling: a python DSL approach to ICON's dynamical core (icon-exclaim v0.2.0)*, GMD 19, 713 (2026)](https://gmd.copernicus.org/articles/19/713/2026/)
