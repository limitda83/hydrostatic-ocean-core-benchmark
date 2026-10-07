# E14_v06_representative

**질문/내용:** spec v0.6: TKE 연직 난류 폐쇄 + 3차 이류(UP3/TVD) — 대표 코어의 비용

**보고서:** docs/03_discretization_spec.md §11, docs/31_physics_cost.md

**노드:** 로컬(검증) → 세 노드 (geo85 · gpgpu · ktcloud)

**상태:** 완료 (2026-09-15). 측정 223행 + 결과 문서 docs/31. **세부 분해를 세 격자에서 실제로 측정해 docs/31 §3b 에 실었다** — 그 전까지 docs/34 가 인용하던 분해 수치는 측정된 적이 없는 값이었다(docs/90 **N40**). 실측(RTX 5090 CUDA fp64, θ0.5·다중격자 CFL 4): 추가 물리 중 폐쇄가 100² 89.1 % · 400² 75.5 % · 1000² 75.5 %, 격자 안에서 다섯 변종의 솔버 반복수 일치(2950/3012/2856), 400² 독립 3회 0.2 % 이내 재현. **fp32 로는 같은 구성을 잴 수 없다** — 다중격자가 rtol 1e-6 에 도달하지 못하고 매 스텝 max_iter 에 걸린다(docs/90 **N41**); 정밀도 교차 분해는 `fb` 로 따로 잰다.
측정: geo85 잡 `t2v6_nx*` (lock_exchange vs lock_exchange_v06, 50스텝 사다리) 2026-09-11 18:55 제출;
gpgpu 는 v0.5 체인 뒤 `CASES_S=lock_exchange_v06`(조각 S) + `CASES_H="lock_exchange lock_exchange_v06"`
(조각 H) + JAX 패스 자동 실행 (`output/tier2_gpgpu_v06.log`); ktcloud 는 게이트 뒤 동일 H 실행

**재현 명령:**
- 분해: `CUDA_VISIBLE_DEVICES=<유휴> BACKENDS=cuda NX=400 NZ=30 STEPS=50 REPEAT=5 bash tools/decompose_physics.sh` (gpgpu). 측정 전 CUDA 재빌드 + `gate3d5.sh` 12/12 PASS 확인(R4).
- `python3 tools/verify_v06.py`
- `bash tools/gate3d5.sh` (v06.* 세 구성)
- `CASES_H="lock_exchange lock_exchange_v06" nohup bash tools/run_tier2_ktcloud.sh` / `qsub output/pbs/tier2v06_nx*.pbs`
- 비교 지표: 같은 케이스·격자·스킴에서 v0.6 스텝당 ms ÷ v0.5 스텝당 ms (백엔드·장치별) → PAPER_PLAN F6

**데이터 (`data/`):** `v06_verification.json`, `v06_verification_20260914.json`, `v06_verification_20260923.json`(**현재 코드의 결과** — N44 뒤 재실행, V6 suite PASS), `advection_checks.json`, `physics_cost.json`(수집기 재생성본), `decompose_gpgpu_theta_20260915.csv`(θ0.5·다중격자 CFL 4, fp64 3격자 + fp32 max_iter 3행), `decompose_gpgpu_fb_20260915.csv`(fb CFL 0.5, fp64·fp32 각 3격자), `decompose_ktcloud_theta_20260915_CONTAMINATED.csv`(H100 θ — 물리를 더했는데 빨라진 행이 있어 **폐기**, SUPERSEDED 참조; H100 은 `_fb_` 분해만 쓴다)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
