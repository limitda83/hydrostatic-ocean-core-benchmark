# E13_tier2_ladder

**질문/내용:** 2층 조각 H: 백엔드×하드웨어×격자(100²…2000²×30), 바이트 동일 50스텝, 세 노드

**보고서:** docs/04_design_v2.md §3, docs/30_tier2_pareto.md, docs/32_hardware_ladder.md, docs/39_speedup_generation_framework.md

**노드:** gpgpu · ktcloud · geo85(PBS excl)

**상태:** **완료** (2026-09-15 정정). 조각 H 는 세 노드 원본 CSV 71개(953 행)가 `data/` 에 보존되어 있고, docs/32 의 모든 칸이 그 파일들만으로 재생산된다(`tools/audit_crossdevice.py` 6표 0실패). 2026-09-14 재측정(`geo85_r2.csv` 65행, `ktcloud_jax_stab.csv` 18행)이 이제 실제로 구판 행을 **대체**한다 — 재측정이 케이스 토큰에 `.r2`/`.stab` 꼬리표를 달아 수집기가 별도 그룹으로 취급하는 바람에 초판 docs/32 는 오염이 확인된 구판 값을 싣고 있었다(docs/90 **N39**). 1차 시도는 PBS 부하 가드 결함으로 폐기(N30).

**재현 명령:**
- `nohup bash tools/run_tier2_ktcloud.sh`
- `nohup bash tools/run_tier2_jax_pass.sh <node>`
- `qsub output/pbs/tier2_nx*.pbs` (tools/pbs_tier2.sh)

**데이터 (`data/`):** `tier2_geo85_*.csv`(16), `tier2_gpgpu_*.csv`(23), `tier2_ktcloud_*.csv`(32) — 세 노드의 조각 S/H 원본; `geo85_r2.csv`(**재측정 확정본**, 65행), `ktcloud_jax_stab.csv`(재측정 완료, 18행), `ktcloud_openacc_r4_20260915.csv`(**R4 재측정** 1차, 48행)·`ktcloud_openacc_r4b_20260915.csv`(2차, 28행: `baroclinic_igw` 전 스킴 + `split`) — 게이트를 통과한 이진으로 H100 OpenACC 열 전체를 다시 잼(docs/90 **N42**); `split2000_lock_exchange_IRREPRODUCIBLE.csv`(Sep-11 CUDA·JAX 의 `lock_exchange*` split 2000² 3행 — 현재 코드는 이 구성에서 모든 백엔드가 발산, **N45**), `ktcloud_openacc_split2000_SUBSTEPS_MISMATCH.csv`(재측정에서 발산한 같은 구성 2행, l2=inf), `geo85_stab_INVALID.csv`(가드 결함으로 폐기), `ktcloud_openacc_UNGATED.csv`(게이트 없는 이진, 폐기), `diverged_cfl4_2000_RESULT.csv`·`gpgpu_dev_20step_SUPERSEDED.csv` — 전부 SUPERSEDED 참조. 조각 S 의 gpgpu 원본은 `expr/E12_tier2_pareto/data/` 에 있으며 여기에 중복 보존하지 않는다.

**재생산:** `python3 tools/tier2_collect.py expr/E13_tier2_ladder/data/tier2_*.csv expr/E13_tier2_ladder/data/geo85_r2.csv expr/E13_tier2_ladder/data/ktcloud_jax_stab.csv expr/E13_tier2_ladder/data/ktcloud_openacc_r4_20260915.csv expr/E12_tier2_pareto/data/tier2_gpgpu_S_current.csv expr/E16_mixing_length/data/*.csv --out output/tier2_summary` (인자 순서와 무관하게 같은 결과가 나온다 — docs/90 N43) → `python3 tools/audit_crossdevice.py docs/32_hardware_ladder.md --rows output/tier2_summary/rows.json`

**2026-09-15 추가 측정 (완료):** `geo85` 2000²×30 θ=0.5 · CFL 2, 두 솔버 × 두 케이스(`lock_exchange`, `lock_exchange_v06`), 스레드 64/96/128/192, 5회 반복 → `tier2_geo85_20260915-004338_node01_1528903.csv`(16행). CFL 4 는 두 솔버 모두 max_iter 로 발산하므로 2000² θ 의 유일한 수렴 설정인데 이 격자의 CPU 열이 비어 있었다(docs/32 §2-2). PBS `4536.master`, node01 배타, 2 h. 케이스 토큰에 **꼬리표를 달지 않아** 기존 2000² CFL 2 행과 한 그룹으로 합쳐졌다(R13-4). 측정 전 그 노드의 Fortran 드라이버를 최신본(n_done 통계 수정)으로 재빌드하고 R2 게이트 24/24 PASS 를 확인했다(R4). 네 솔버-케이스 조합 모두 **반복수가 네 장치에서 일치**(다중격자 928, PCG-Jacobi 2474). **남은 미측정 칸은 없다.** 2026-09-15/16: H100 OpenACC 열 전체를 게이트 통과 이진으로 재측정(1차 48행 + 2차 28행; 구판과 최대 6 % — 구판 값은 옳았으나 게이트 기록이 없었다, docs/90 **N42**). 2차에서 `lock_exchange*` split 2000² 는 **발산**(l2=inf, 20스텝에서 |η|>1e6) — 같은 번들로 CUDA 도 발산하므로 백엔드 결함이 아니라 현재 코드의 결과이며, Sep-11 의 같은 구성 3행은 옛 이진이라 재현 불가로 격리(**N45**). `baroclinic_igw` 와 `split` 스킴의 ktcloud OpenACC 행(원본 35행)은 구판 이진 것이라 `ktcloud_openacc_UNGATED.csv` 로 **분리해 SUPERSEDED 에 등록**했다 — 수집기와 감사가 기계적으로 제외한다. 쓰려면 게이트 통과 이진으로 재측정해야 한다.

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** N12·N13 (PBS 충돌·부하 꼬리), 100²/200² 는 omp_min_points=0 재측정본
