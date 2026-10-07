# E04_openacc

**질문/내용:** OpenACC 백엔드 R2 게이트, reduction 절 침묵 버그

**보고서:** docs/13_p4_openacc.md

**노드:** gpgpu

**상태:** done

**재현 명령:**
- `make -C libs/fortran acc`

**데이터 (`data/`):** `openacc_gate_metrics.txt` (gpgpu 의 OpenACC 게이트 run 7개 metrics 합본)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
