# expr/ — 실험 대장 (experiment ledger)

실험 하나 = 디렉터리 하나. 각 디렉터리에 `README.md`(질문·설계·명령·노드·날짜·상태·주의), `data/`(수집된 CSV/JSON — `output/` 은 git 에 없으므로 **여기가 보존본**), 결과 문서는 `docs/NN`. 규칙은 RULES.md R13.

> 상태·보고서·보존본은 **손으로 쓰지 않는다** — 아래 자동 생성 감사 표가 각
> `expr/E##/README.md` 를 읽어 채운다. 이 표는 무엇을 한 실험인지만 적는다.

| ID | 실험 | 노드 |
|---|---|---|
| E01_phase0_2d_schemes | 2D 선형 회전 천수 · fb/theta · fft/pcg 게이트와 오차-비용 곡선 (DEV-ONLY 시간) | 로컬(dev) |
| E02_fortran_cpu | Fortran CPU 백엔드 R2 게이트, OpenMP 임계값(omp_min_points) 보정, -fopenmp 1스레드 ≠ 직렬 | gpgpu · geo85 |
| E03_gpu_microbench | RTX 5090 roofline 상수 실측 (fp64/fp32 대역폭·연산) | gpgpu |
| E04_openacc | OpenACC 백엔드 R2 게이트, reduction 절 침묵 버그 | gpgpu |
| E05_3d_reference | 3D 정역학 레퍼런스(v0.2) · 완전 코어(v0.4) 검증 V3D-1…5 | 로컬(검증만) |
| E06_performance_matrix_v02 | v0.2~0.3 코어의 스킴×백엔드×격자 성능 매트릭스, split-explicit, 해법·메모리·dt 축 | gpgpu · geo85 |
| E07_v05_physics | spec v0.5 검증 스위트 V5-1…V5-6 (지형·연직좌표·개방경계·EOS) | 로컬(검증) |
| E08_kernel_matrix | 1층: 변계수 Helmholtz 해법 × 정밀도 × 하드웨어(RTX 5090/H100/EPYC), EOS 커널 roofline, 에너지 | gpgpu · ktcloud · geo85 |
| E09_performance_index | 종합 성능지표 = 반복수(알고리즘) × 반복당 비용(구현·하드웨어), 예측 검증 1.1 % | —(E08 데이터) |
| E10_lock_exchange_rpe | 락 익스체인지 RPE (예비, 이류 없음) | 로컬 |
| E11_multigpu_comm | 다중 GPU 통신 마이크로벤치 (halo/allreduce, RTX 5090 ×2) | gpgpu |
| E12_tier2_pareto | 2층 조각 S: 스킴×해법×CFL×케이스, 400²×30, CUDA, 오차 vs 소요시간 (파레토 평면) | gpgpu (RTX 5090) |
| E13_tier2_ladder | 2층 조각 H: 백엔드×하드웨어×격자(100²…2000²×30), 바이트 동일 50스텝, 세 노드 | gpgpu · ktcloud · geo85(PBS excl) |
| E14_v06_representative | spec v0.6: TKE 연직 난류 폐쇄 + 3차 이류(UP3/TVD) — 대표 코어의 비용 | 로컬(검증) → 세 노드 |
| E15_precision | 정밀도 축(axis G) 전체 코어 fp32: 카드별 가속비, 물리 벌금의 정밀도 의존, 용량 레버 | gpgpu · ktcloud |
| E16_mixing_length | 혼합길이 축 `mxl`: O(nz²) 적분형 대 O(nz) 재귀형 — 같은 최적화의 가치가 장치·정밀도·nz 에 어떻게 의존하는가 | gpgpu · ktcloud · geo85 |

## 어디에 무엇이 있나

- 설계: `docs/01`(1차), **`docs/04`(현행)**; 이산화 스펙 `docs/03`; 부정결과 `docs/90`
- 실행 도구: `tools/` (스윕·게이트·수집·보고서); 노드 경로·툴체인: `usage.md` §12
- 논문 계획: `expr/paper/PAPER_PLAN.md`

<!-- BEGIN expr_audit -->

## 대장 감사 (`python3 tools/expr_audit.py --write` 가 생성)

> 상태의 단일 출처는 각 `expr/E##/README.md` 의 `**상태:**` 필드다. 이 표는 그것을 읽어 쓴다.

| 실험 | 상태 | 보고서 | `data/` | 감사 |
|---|---|---|---:|---|
| `E01_phase0_2d_schemes` | done | docs/09_phase0_results.md | 29 파일 | OK |
| `E02_fortran_cpu` | done | docs/11_p3_fortran_cpu.md | 27 파일 | OK |
| `E03_gpu_microbench` | done | docs/12_gpu_microbench.md | 1 파일 | OK |
| `E04_openacc` | done | docs/13_p4_openacc.md | 1 파일 | OK |
| `E05_3d_reference` | done | docs/14_p2_3d_reference.md, docs/16_p2c_full_core.md, docs/40_verification_summary.md | 30 파일 | OK |
| `E06_performance_matrix_v02` | done (v0.2 코어 — 지형 없음, N14 이전 fb 는 압력경사 절반이었음: fb 행은 | docs/20_performance_matrix.md, docs/21_p2_3d_backends.md, docs/22_split_explicit.md, docs/23_solver_memory_axes.md | 18 파일 | OK |
| `E07_v05_physics` | **완료** (2026-09-14 재실행). V5-1…6 전부 PASS. 보존본이 N15 수정 | docs/24_v05_physics_results.md | 4 파일 | OK |
| `E08_kernel_matrix` | **완료** (2026-09-13, 두 차례 재측정). ① R7-2 재측정(5회 median+ | docs/25_kernel_matrix.md | 16 파일 | OK |
| `E09_performance_index` | 완료. 2026-09-13 E08 안정화 재측정본으로 **재계산** — 결론 유지(GPU·직렬 | docs/26_performance_index.md | 2 파일 | OK |
| `E10_lock_exchange_rpe` | **완료** (2026-09-14). docs/27 §4 가 열어 둔 세 확인을 측정해 종결했 | docs/27_lock_exchange_rpe.md | 5 파일 | OK |
| `E11_multigpu_comm` | **완료** (2026-09-14 재측정). 2랭크 512²·1024²·2048² + 1랭크  | docs/28_multigpu_communication.md | 1 파일 | OK |
| `E12_tier2_pareto` | **완료** (2026-09-12 06:14, 3 케이스 × 31 구성). 결과 문서 `doc | docs/04_design_v2.md | 1 파일 | OK |
| `E13_tier2_ladder` | **완료** (2026-09-15 정정). 조각 H 는 세 노드 원본 CSV 71개(953 행 | docs/04_design_v2.md, docs/30_tier2_pareto.md, docs/32_hardware_ladder.md, docs/39_speedup_generation_framework.md | 83 파일 | OK |
| `E14_v06_representative` | 완료 (2026-09-15). 측정 223행 + 결과 문서 docs/31. **세부 분해를 세 | docs/03_discretization_spec.md, docs/31_physics_cost.md | 12 파일 | OK |
| `E15_precision` | 완료 (2026-09-13). 두 노드 각 48행. | docs/34_precision_axis.md | 2 파일 | OK |
| `E16_mixing_length` | **완료** (2026-09-13). 400²·1000² 전 축 — 2 카드 × 2 정밀도 × | docs/36_mixing_length_axis.md | 13 파일 | OK |
| `E17_module_boundary` | **완료 · 2회차** (2026-09-15). 외부 적대적 검사(docs/90 **N44** | docs/38_module_boundary.md, docs/37_ai_module_motivation.md | 7 파일 | OK |
| `E18_mixed_precision` | **완료** (2026-09-16, 두 카드). docs/34 §6. 폐쇄 비용 mixed/f | docs/34_precision_axis.md, docs/37_ai_module_motivation.md | 7 파일 | OK |
| `E19_multinode_scaling` | **완료 — RTX 5090 ×2·×4(노드 안, 2026-09-24) + H100 ×2(2노 | docs/41_multinode_design.md, docs/42_multigpu_scaling.md | 3 파일 | OK |

<!-- END expr_audit -->
