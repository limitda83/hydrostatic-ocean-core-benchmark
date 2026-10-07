# usage — cfd_exp

해양수치모델의 **시간적분법 × 구현 백엔드**를 통제 비교하는 실험 하네스.
읽는 순서: `RULES.md`(규칙) → `docs/01_experiment_design.md`(설계) →
`docs/03_discretization_spec.md`(이산화 스펙) → 이 문서(실행법).

## 1. 설치

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python >= 3.11 (tomllib)
pip install -r requirements.txt
```

## 2. 설정 (`config/*.toml`)

소스에 파라미터를 넣지 않는다 (RULES.md R3). 모든 값은 아래 파일에 있다.

| 파일 | 내용 |
|---|---|
| `settings.toml` | 경로, 로깅, 물리상수(`g`, `f0`, `H`), 정밀도, NetCDF 옵션 |
| `grid.toml` | `nx/ny/nz`, `Lx/Ly`, 경계조건, 격자 세분 스윕 목록 |
| `schemes.toml` | 시간적분(`fb`/`theta`), `theta`, 코리올리 Picard 설정, 타원형 해법 |
| `cases.toml` | 검증 케이스(`igw`, `geo_balance`), 적분 길이, dt/CFL 스윕 |
| `backends.toml` | 백엔드 레지스트리, 컴파일 플래그, 기준 벤치 노드 정보 |

임시 변경은 파일을 고치지 말고 `--set` 으로 덮어쓴다 (원본 config는 manifest에 그대로 기록됨):

```bash
python3 main.py --set scheme.theta=0.6 --set scheme.name=theta run
```

## 3. 실행

### 3.1 단일 실행

```bash
python3 main.py run --case igw --nx 100
python3 main.py run --case geo_balance --nx 100 --cfl 0.5
```
`output/run_<case>_<scheme>_<timestamp>/` 에 `state.nc`, `metrics.json`,
`manifest.json`, `run.log` 을 남긴다.

### 3.2 수렴차수 검증 (구현 정확성의 유일한 증거)

```bash
python3 main.py verify --study space --case igw      # 격자 세분, 고정 CFL
python3 main.py verify --study time  --case igw --nx 128   # dt 세분, 동일격자 기준해 대비
```
기대값: `theta=0.5` → 2차 · `theta=0.6/1.0` → 1차 · `fb` → 2차 (CFL 한계 내).
이 표를 통과하지 못한 코드의 **속도 수치는 보고 금지** (RULES.md R4).

### 3.3 오차-비용 프론티어

```bash
python3 main.py bench --case igw --nx 100
```
CFL 스윕을 돌며 `L2 오차`와 `wallclock`을 함께 기록한다.
**`gpgpu` 노드가 아니면 경고가 뜨고, 그 타이밍은 리포트에 쓸 수 없다** (R7-1).

### 3.4 스킴 비교 예 (θ의 정확도 대가 재현)

```bash
for t in 0.5 0.6 1.0; do
  python3 main.py --set scheme.theta=$t verify --study space --case igw
done
```

## 4. 출력

```
output/<run_id>/
├── manifest.json   # git SHA/dirty, config 전문 + SHA256, 호스트/툴체인, UTC 시각
├── metrics.json    # 정확도 지표 + 보존량 + 타이밍
├── state.nc        # CF-1.10 NetCDF4 (eta/u/v + *_exact + *_error)
└── run.log
```

`state.nc` 확인:
```bash
ncdump -h output/<run_id>/state.nc
python3 -c "from netCDF4 import Dataset; d=Dataset('output/<run_id>/state.nc'); print(d.variables.keys())"
```

## 5. 지표 읽는 법

| 키 | 의미 |
|---|---|
| `l2_rel_eta` | 해석해 대비 상대 L2 오차 — 주 정확도 지표 |
| `observed_order` | 관측 수렴차수. 이론값과 다르면 **구현 버그** |
| `amp_ratio` | 1.0 = 무감쇠. `θ>0.5` 의 대가가 여기 나타난다 |
| `phase_err_rad` | 분산(위상) 오차 |
| `mass_drift` | `rms(η)·도메인면적` 으로 정규화 (M0≈0이라 M0 정규화 불가) |
| `energy_drift` | `θ=0.5` 는 ~0, `θ>0.5` 는 단조 감소 |
| `solver_iterations` | PCG 총 반복수 — GPU 확장성 병목 지표 (RQ2) |
| `timing_reportable` | `false` 면 개발기 측정 → 리포트 금지 |

## 6. Fortran CPU 백엔드 (`fortran_cpu`)

```bash
make -C libs/fortran                    # -> libs/fortran/build/cfd_exp
make -C libs/fortran debug              # 배열 경계검사 빌드
make -C libs/fortran OMP=                # OpenMP 없이 (소규모 문제에 유리, docs/11 S2)
```

### 6.1 실행

설정은 여전히 `config/*.toml` 하나뿐이다. namelist는 생성물이며 손으로 고치지 않는다.

```bash
python3 -m tools.toml2nml --set solver.kind=pcg_jacobi --case igw --nx 512 --cfl 0.5         --out output/fortran/run.nml --prefix output/fortran/run
./libs/fortran/build/cfd_exp output/fortran/run.nml
```
`dt` 와 `n_steps` 는 Python이 결정해 namelist로 넘긴다 —
두 백엔드가 서로 다른 문제를 푸는 일이 원천적으로 불가능하다.

### 6.2 R2 게이트 (백엔드 머지 전 필수)

```bash
python3 -m tools.compare_backends --set solver.kind=pcg_jacobi --case igw --nx 64 --cfl 0.5
```
1 step `<1e-12`, 전체 적분 `<1e-9` 를 통과해야 한다. 통과 전에는 **속도 수치 보고 금지**(R4).

### 6.3 OpenMP 주의 — 작은 격자에서는 켜지 마라

100×100 격자(10,000점)는 **병렬화 이득 구간에 들어오지 못한다.**
64×64 에서 12스레드가 직렬보다 **22배 느렸다**(docs/11 §2).
`omp_min_points`(기본 65,536) 미만이면 자동으로 직렬 폴백하며, 노드마다 재보정해야 한다.

### 6.4 geo85 192코어 확장성 (PBS)

```bash
ssh <geo85-node>
cd ~/cfd_exp && qsub -v NX=512,CASE=igw,CFL=0.5 tools/pbs_omp_sweep.sh
qstat -u $USER
```
**geo85에는 GPU가 없다.** 여기 수치는 CPU 확장성 축 전용이며,
`gpgpu` 의 GPU 수치와 나란히 놓으면 안 된다 (R7-1).

## 7. spec v0.5 — 실지형·연직좌표·실경계·비선형 EOS

### 7.1 검증 스위트 (V5-1 … V5-6)

```bash
python3 tools/verify_v05.py --nx 48 --nz 20 --steps 100 \
        --json output/v05_verification.json
# 특정 케이스만:  --only v5_1,v5_4
```

`V5-1` 은 v0.5 스테퍼가 검증된 v0.4 스테퍼를 기계정밀도로 재현하는지 본다. 물리 코드에
손대면 **가장 먼저 깨지는 것이 이것**이므로 항상 먼저 돌린다.

### 7.2 설정 축 (`config/settings.toml`)

```toml
[domain]      vcoord = "zlevel" | "zstar" | "sigma"
              bc_x / bc_y = "periodic" | "closed" | "open"
[bathymetry]  kind = "flat" | "slope" | "seamount" | "ridge" | "rough"
              r_target        # rough 의 std(H)/mean(H) 목표
              spectrum_kmax   # 0 이 아니면 대역제한 → 격자 스윕용
[physics3d_v05]
              eos = "linear" | "seos" | "teos10"
              pgf = "remove" | "keep"      # 깊이평균 제거 여부
              pgf_correction = true        # 공통깊이 보정 (끄면 30~110배 나빠진다)
```

> **주의:** 연직좌표와 압력경사 보정은 **직교하지 않는다.** 한쪽만 바꾼 비교는
> 다른 쪽의 효과를 오독하게 만든다 (docs/24 §3).

## 8. 커널 벤치마크 (변계수 Helmholtz · 상태방정식)

전체 v0.5 물리를 세 언어로 이식하는 대신, 비용을 지배하는 두 커널만 이식해 측정한다.

### 8.1 문제 만들기 (Python 한 곳에서만)

```bash
python3 tools/write_helmholtz_case.py --nx 512 --topo rough --r-target 0.10 \
        --cfl 32 --Lx 4.0e5 --out output/hcase --reference --skip-numpy-solvers
```

`ku.bin` · `kv.bin` · `mask.bin` · `rhs.bin` · `eta_ref.bin` · `case.nml` 을 쓴다.
**모든 백엔드가 이 이진 파일을 읽는다** — 지형 생성기를 세 언어로 다시 짜면 R2 게이트가
비교할 기준을 잃기 때문이다.

### 8.2 백엔드 빌드

```bash
make -C libs/fortran helm            # OpenMP CPU
make -C libs/fortran helm-serial     # 직렬 (확장성 보고의 유일한 기준선)
make -C libs/fortran helm-sp         # fp32 solver (축 G)
make -C libs/fortran helm-acc        # OpenACC GPU  (nvfortran 필요)
make -C libs/cuda    helm    ARCH=sm_90   # 네이티브 CUDA (H100=sm_90, RTX 5090=sm_120)
make -C libs/cuda    helm-sp ARCH=sm_90   # fp32 solver
make -C libs/fortran eos ; make -C libs/cuda eos ARCH=sm_90
```

### 8.3 실행과 R2 게이트

```bash
libs/fortran/build/helmholtz_bench output/hcase/case.nml pcg_jacobi
libs/cuda/build/helmholtz_bench_cuda output/hcase/case.nml multigrid
```

**모든 백엔드의 `iterations` 가 같아야 한다.** 다르면 튜닝 차이가 아니라 버그다.
`l2_rel_vs_reference` 도 자리마다 일치해야 한다.

### 8.4 행렬 스윕

```bash
# 한 노드 전체 (BACKENDS 로 역할 제한: R7-1)
PY=./.venv/bin/python SIZES="128 256 512" CFLS="2 8 32 128" \
  TOPOS="flat rough0.10 seamount" SOLVERS="pcg_jacobi pcg_rbgs multigrid" \
  THREADS="1 16 32" BACKENDS="cpu gpu" bash tools/helm_matrix.sh

# GPU 전용 노드 (ktcloud)
BACKENDS=gpu bash tools/helm_matrix.sh

# geo85 스레드 스윕 (PBS)
qsub -v NX=1024,CFL=32,TOPO=rough,RT=0.10 tools/pbs_helm_sweep.sh

# 상태방정식 (RQ7)
N=30000000 REPEAT=20 bash tools/eos_matrix.sh
```

### 8.5 3D 모델 R2 게이트

```bash
bash tools/gate3d.sh          # 모든 케이스 × 이 노드에 빌드된 모든 3D 백엔드
NX=48 STEPS=40 bash tools/gate3d.sh
```

## 9. 종합 성능지표

```bash
python3 tools/performance_index.py output/collected/*.csv \
        --hardware output/collected/gpgpu_helm.csv=rtx5090 \
        --hardware output/collected/h100_helm.csv=h100 \
        --validate --json output/performance_index.json
```

시간 = **반복수(알고리즘)** × **반복당 비용(구현·하드웨어)** 로 분해하고, 두 인자를
곱해 아무도 돌려보지 않은 조합을 예측한다. `--validate` 는 그 예측을 실측 전체와
대조한다 (측정: 중앙값 1.1 %, GPU 백엔드는 0.5~1.2 %). 읽는 법은 docs/26.

## 10. 에너지

```bash
tools/gpu_energy.sh mg512 libs/cuda/build/helmholtz_bench_cuda output/hcase/case.nml multigrid
```

100 ms 간격으로 `power.draw` 를 적분하고, **유휴 전력을 따로 보고**한다 (짧은 커널은
벽시계의 대부분을 유휴 전력으로 보내므로 그것까지 커널에 물리면 긴 실행이 유리해 보인다).

## 11. 현재 구현 범위

- ✅ Phase 0: 2D 선형 회전 천수, 이중주기, `fb` / `theta`, `fft` / `pcg_jacobi`, V1·V2
- ✅ Phase 1: `split_explicit` (spec v0.3)
- ✅ Phase 2: 3D 정역학 (100×100×30), 연직 음해확산, 추적자, V3D-1…5
- ✅ Phase 3: `fortran_cpu` (gfortran/nvfortran + OpenMP) — R2 게이트 통과
- ✅ Phase 4–5: OpenACC / 네이티브 CUDA — R2 게이트 통과
- ✅ Phase 8: spec v0.5 물리 — V5-1…V5-6 통과 (**NumPy 기준구현**)
- ✅ v0.5 **전체 코어**의 컴파일 백엔드: Fortran(직렬/OpenMP/OpenACC 한 소스,
  `cfd_exp3d5*`)과 네이티브 CUDA(`cfd_exp3d5_cuda`) — `tools/gate3d5.sh` 7 구성 전부 통과
  (gpgpu · ktcloud · geo85). z-level 만; z\*/σ·개방경계는 NumPy 결과(docs/24)로 대신한다.
- ✅ Phase 6: Python-GPU 백엔드 = **JAX** (`libs/jax/`, 같은 번들·같은 게이트, RTX 5090 7/7)
- ✅ Phase 7: **spec v0.6** — TKE 연직 난류 폐쇄 + κ=1/3 3차 이류(+Superbee TVD), 다섯 구현 전부
  (게이트 10 구성: gpgpu 30/30, geo85 20/20)
- ⬜ 다중 GPU / MPI

미구현 케이스·스킴을 요청하면 명시적 예외를 던진다 (조용히 다른 것을 계산하지 않는다).

## 12. 원격 벤치 노드

| 노드 | 접속 | 역할 | 툴체인 |
|---|---|---|---|
| `gpgpu` | `ssh <gpgpu-node>` | **CPU↔GPU 비교 전부** (RTX 5090 ×4) | nvfortran 25.11, nvcc sm_120. **gfortran 없음** → `FC=nvfortran` |
| `ktcloud` | `ssh ktcloud` (`~/.ssh/config`) | **GPU 전용** (H100 80GB). 호스트 CPU 공유 → CPU 시간 사용 금지 | gfortran 13.3, nvcc 13.0 sm_90 |
| `geo85` | `ssh <geo85-node>` | **CPU 확장성 전용** (EPYC 9655 96C/192T ×5, PBS, GPU 없음) | gfortran |

세 노드의 수치를 섞으면 무효다 (R7-1). 배포는 tarball + `scp`(gpgpu 에는 `rsync` 없음).
`ktcloud` 의 `/home/work` 는 세션 종료 시 삭제되므로 영속 자산은 `/home/work/cello` 아래에 둔다.

## 13. 2층(모델 수준) 실험 — 오케스트레이션 (docs/04)

한 노드의 스윕은 `tools/tier2_sweep.sh` 하나가 돈다. 두 모드:

| 모드 | 환경 | 하는 일 |
|---|---|---|
| 지평(horizon) — 조각 S | `STEPS` 없음 | 케이스마다 수렴 기준해(`fb`, CFL 0.1, 가장 빠른 게이트 통과 백엔드)를 먼저 만들고, 스킴×해법×CFL 을 물리적 지평까지 적분해 **오차(L2, 기준해 대비)와 소요시간**을 함께 기록 |
| 고정 스텝 — 조각 H | `STEPS=50` | 기준해 없음. 모든 노드가 **바이트 동일한 문제**를 50 스텝 풀어 스텝당 비용만 비교 |

```bash
# gpgpu: 조각 S(400²×30, CUDA) → 조각 H(격자 사다리, CPU+GPU); JAX 는 2차 패스(조각 H 만)
nohup bash tools/run_tier2_gpgpu.sh > output/tier2_gpgpu.log 2>&1 &
nohup bash tools/run_tier2_jax_pass.sh gpgpu > output/tier2_gpgpu_jax.log 2>&1 &
# ktcloud: 조각 H 만 (GPU 전용 노드), 이어서 JAX 패스
nohup bash tools/run_tier2_ktcloud.sh > output/tier2_ktcloud.log 2>&1 &
nohup bash tools/run_tier2_jax_pass.sh ktcloud > output/tier2_ktcloud_jax.log 2>&1 &
# geo85: 격자마다 배타 노드 하나 (place=excl)
qsub -v NX=400,STEPS=50,CASES=baroclinic_igw tools/pbs_tier2.sh
# 세 노드의 CSV 를 모아 파레토 평면·표를 만든다
python3 tools/tier2_collect.py output/collected/tier2_*.csv --out output/tier2_summary
```

`BACKENDS` 는 그룹(`cpu`, `gpu`)이나 개별 이름(`fortran_serial fortran_omp openacc cuda jax`)을
받는다. JAX 는 `JAX_PY` 로 인터프리터를 고른다(gpgpu `./.venv-jax/bin/python`, ktcloud
`/home/work/cello/.venv/bin/python`). 출력 디렉터리는 `output/tier2_<stamp>_<node>_<pid>` —
같은 초에 시작한 두 PBS 잡이 한 디렉터리를 나눠 쓴 사고(docs/90 N12) 뒤의 형식이다.

**왜 조각 S 는 CUDA 한 백엔드로만 도나.** 오차는 스킴·dt 의 성질이지 백엔드의 성질이 아니다(R2 로
모든 백엔드가 1e-9 안에서 같다). 그러므로 오차는 가장 빠른 백엔드로 한 번만 재고, 다른 백엔드의
"고정 오차에서의 소요시간"은 조각 H 의 스텝당 비용 × 조각 S 의 스텝수로 얻는다(docs/26 의
성능지표, 검증 1.1 %). 400²×30 을 전 지평 적분하면 OpenACC 로도 구성당 수십 분, CPU 16 스레드로는
수 시간이라 5회 반복(R7-2)을 모든 백엔드에 적용할 수 없다.

## 14. spec v0.6 — 연직 난류 폐쇄 · 3차 이류 (대표 코어)

```toml
[physics3d_v06]
closure = "tke"          # none | tke   (Gaspar 1990 1-방정식, 적분 혼합길이; docs/03 §11.2)
[scheme]
advection = "up3_tvd"    # none | centered2 | upwind1 | up3 | up3_tvd  (κ=1/3, Superbee; §11.3)
```

```bash
python3 tools/verify_v06.py            # V6-1 이류 차수(up3 = 3.00) · V6-2 제한자 단조성 · V6-3 Kato-Phillips 혼합층
bash tools/gate3d5.sh                  # 10 구성 = v0.5 7 + v0.6 3 (up3·mg, tvd+tke+θ, tke+split), 5 백엔드
CASES=lock_exchange_v06 bash tools/tier2_sweep.sh   # 케이스 ③ 를 폐쇄 + TVD 3차 이류로
```

폐쇄를 켜면 연직 확산계수가 매 스텝 3D 배열이 되어 삼중대각 계수·`q`·`K_u/K_v`·Helmholtz 연산자
(다중격자 전 레벨)를 **매 스텝 재구성**한다 — 그것이 실제 모델의 비용이고 `solver_rebuilds` 로 센다.
`closure='none'` 은 v0.5 와 비트단위 동일(V6-4 = R2 게이트).

