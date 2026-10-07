# E17_module_boundary

**질문/내용:** 경험식 폐쇄를 AI 모듈로 떼어낼 때 치르는 **장치 경계 비용**은 그 폐쇄 커널
자체의 몇 배인가. v0.6 TKE 폐쇄가 스텝마다 실제로 읽고 쓰는 배열(`libs/core/closure.py::tke_step`
서명 그대로: 입력 `u,v,b,e,K_m_old,K_h_old`, 출력 `e_new,K_m,K_h`)을 격자 경계 너머로 넘기고
돌려받는 시간을 재고, 제자리 커널 시간(docs/35 의 `k_closure`)과 비교한다. 손익분기 격자를
장치별로 특정하는 것이 목표다.

**배경:** docs/37 §4·§6. 문헌 근거는 docs/02 **L14.4** — 같은 CNN 모듈이 CPU 상주 호스트에서
코어의 ~10배, GPU 에서 코어의 ~0.1배(Zanna et al. 2025). 이 실험은 그 100배가 어디서 오는지의
**하한**을 우리 자료 발자국으로 측정한다.

**보고서:** docs/38_module_boundary.md (결과), docs/37_ai_module_motivation.md §4·§6 (설계)

**노드:** gpgpu (RTX 5090, PCIe 5.0) · ktcloud (H100)

**R7 주의:** 이 실험이 재는 것은 **호스트를 거치는 전송**이다. `ktcloud` 는 호스트 CPU·PCIe 를
다른 테넌트와 공유하므로(R7-1) 그 노드의 B·C·E 값은 **상한이자 오염 가능값**이다 — MAD 를
반드시 병기하고, 장치 내 D 값과 `gpgpu` 의 값으로 교차 확인한다.

**상태:** **완료 · 2회차** (2026-09-15). 외부 적대적 검사(docs/90 **N44**) 뒤 케이스 D(장치 내 복사)·E(두 스트림)를 **폐기**하고 도구에서 제거, 두 노드에서 B·C(호스트 왕복)만 재측정. 결합 도구를 다시 썼다(장치는 CSV 의 `gpu` 열에서, 모든 축을 키에, 차이가 MAD 의 2배를 못 넘으면 폐기 표시). 결과: fp32·400² 이상에서 두 카드 모두 호스트 왕복이 폐쇄의 **2.1~2.7배**(하한). 제자리 읽기가 가능한 장치 상주 모듈의 경계 비용은 0 — 그것은 측정할 것이 없다.

**측정 도구가 CuPy 가 아니라 CUDA C 인 이유:** `ktcloud` 에 CuPy 가 없다. 두 노드에서 같은
이진 의미를 가지려면 구현이 하나여야 하므로(측정 도구가 둘이면 결과가 갈릴 때 어느 쪽이
맞는지 말할 수 없다) `nvcc` 로 통일했다.

**재현 명령:**
- 빌드: `cd libs/cuda && [ARCH=sm_90] make boundary`
- 측정: `CUDA_VISIBLE_DEVICES=<유휴> build/module_boundary --nx <N> --nz 30 --itemsize <8|4> --n-repeat 5`
  (CSV 를 stdout 으로. `--static-every-step` 은 `dz3`·`mask3` 까지 매 스텝 보내는 순진한 구현)
- 분모: `BACKENDS=cuda|cuda_sp NX=<N> STEPS=50 REPEAT=5 bash tools/decompose_physics.sh`
- 결합: `python3 tools/module_boundary_report.py --decompose expr/E14_v06_representative/data/decompose_*.csv --boundary expr/E17_module_boundary/data/*_boundary.csv`

**데이터 (`data/`):** `gpgpu_boundary.csv`, `ktcloud_boundary.csv`(2회차, B·C 만), `gpgpu_boundary_static.csv`(정적 배열까지 매 스텝 보내는 순진한 구현, 2회차), `*_DISCARDED_DE.csv`(1회차 — D·E 포함, SUPERSEDED 참조). 분모는 `expr/E14_v06_representative/data/decompose_*_fb_20260915.csv`.

**보존 상태:** 두 노드 원자료 회수 완료. `ktcloud` 는 세션 종료 시 `/home/work` 가 삭제되므로 측정 직후 회수했다(R13-2).

**코드 리비전:** 측정 시 `libs/cuda/src/module_boundary.cu`, gpgpu sm_120 / ktcloud sm_90 빌드. 같은 패스에서 CUDA 코어를 재빌드하고 R2 게이트 12/12 PASS 확인(R4).

**주의:** 이 실험은 성능 주장을 만들지 않고 **경계의 하한**을 준다. 실제 모듈 구현은 이보다
느릴 수만 있으므로, 측정값은 "CPU 호스트 + GPU 모듈" 구성에 **유리한 쪽으로 치우친** 추정치다.
사전 예측(docs/37 §6): 1000²×30 fp64 에서 왕복 약 2.2 GB/스텝 → PCIe 5.0 에서 약 40 ms,
같은 격자 `k_closure` 12.16 ms 대비 **3배 이상**. 예측이 틀리면 그것이 결과다(R12).
