# E08_kernel_matrix

**질문/내용:** 1층: 변계수 Helmholtz 해법 × 정밀도 × 하드웨어(RTX 5090/H100/EPYC), EOS 커널 roofline, 에너지

**보고서:** docs/25_kernel_matrix.md

**노드:** gpgpu · ktcloud · geo85

**상태:** **완료** (2026-09-13, 두 차례 재측정). ① R7-2 재측정(5회 median+MAD) → ② 감사에서 288행 중 69행 MAD>2 %, 최선 CPU 16칸 중 8칸이 그런 행 → **안정화 재측정** (반복 9회, MAD>2 %면 최대 4회 재실행, 시도별 중앙값의 최소 채택, 시도 횟수 CSV 기록). 남은 불안정 48행은 전부 다중스레드 CPU 이며 원인은 경합이 아니라 **거대페이지 폴백**(N29). 직렬·OpenACC·CUDA 는 불안정 0건. geo85 는 MAD 중앙값 0.11 %.

**재현 명령:**
- `bash tools/helm_matrix.sh`
- `bash tools/eos_matrix.sh`
- `qsub tools/pbs_helm_sweep.sh`
- `tools/gpu_energy.sh`

**데이터 (`data/`):** **재측정본(원고용)** — `gpgpu_helm_r7.csv`(960행), `gpgpu_helm_r7_rough010_stable.csv`(288행, **docs/25 §2 의 출처**), `gpgpu_helm_r7_rough010.csv`(같은 구성의 5회판, 비교용), `gpgpu_eos_r7.csv`, `ktcloud_helm_r7.csv`(480행), `ktcloud_eos_r7.csv`, `geo85_helm_r7.csv`(81행). ⚠ 이하는 **구 min-of-3 데이터**로 보존 목적이며 원고 사용 금지: geo85_helm.txt, gpgpu_helm.csv, gpgpu_helm_1024.csv, gpgpu_helm_1024_check.csv, gpgpu_helm_2048.csv, h100_helm.csv, h100_helm_2048.csv, h100_helm_big.csv

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** R7-2 (min-of-3) — 재측정 필요 항목
