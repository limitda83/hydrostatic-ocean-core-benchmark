# kernel_matrix

**뒷받침하는 주장:** paper/TARGET.md C7, C2(대역폭 지배)
**결과 문서:** docs/25_kernel_matrix.md, docs/12_gpu_microbench.md
**출처 실험:** E03_gpu_microbench, E08_kernel_matrix
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

변계수 Helmholtz 해법 × 정밀도 × 하드웨어(반복 9회 안정화 재측정, MAD 병기), EOS 커널 roofline, 5090 roofline 상수(ridge fp64 1.25 / fp32 59.8 FLOP/byte).

## 재생산

- `bash tools/helm_matrix.sh`
- `bash tools/eos_matrix.sh`
- `python3 tools/gpu_microbench.py`

## 파일 (8)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E08_kernel_matrix/geo85_helm_r7.csv` | `expr/E08_kernel_matrix/data/geo85_helm_r7.csv` | 9a51a87d4332 |
| `E08_kernel_matrix/gpgpu_eos_r7.csv` | `expr/E08_kernel_matrix/data/gpgpu_eos_r7.csv` | f63c58a40aa9 |
| `E08_kernel_matrix/gpgpu_helm_r7.csv` | `expr/E08_kernel_matrix/data/gpgpu_helm_r7.csv` | 6d1c4b03aafa |
| `E08_kernel_matrix/gpgpu_helm_r7_rough010.csv` | `expr/E08_kernel_matrix/data/gpgpu_helm_r7_rough010.csv` | f42809d13d96 |
| `E08_kernel_matrix/gpgpu_helm_r7_rough010_stable.csv` | `expr/E08_kernel_matrix/data/gpgpu_helm_r7_rough010_stable.csv` | fc3c43fc8950 |
| `E08_kernel_matrix/ktcloud_eos_r7.csv` | `expr/E08_kernel_matrix/data/ktcloud_eos_r7.csv` | 6342280b8082 |
| `E08_kernel_matrix/ktcloud_helm_r7.csv` | `expr/E08_kernel_matrix/data/ktcloud_helm_r7.csv` | dda66426a5e4 |
| `E03_gpu_microbench/gpu_microbench.json` | `expr/E03_gpu_microbench/data/gpu_microbench.json` | b5179d140c4a |
