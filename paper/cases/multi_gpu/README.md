# multi_gpu

**뒷받침하는 주장:** paper/TARGET.md E19 (strong scaling: fb, DC and DC+TKE; 2 and 4 RTX 5090 in one node by peer copy, 2 H100 across two nodes by MPI; bit-identical gate vs 1 device)
**결과 문서:** docs/42_multigpu_scaling.md, docs/41_multinode_design.md, docs/03 S12
**출처 실험:** E19_multinode_scaling
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

Single-process multi-GPU (y-slab, deep halo H=16) strong scaling of the fb scheme on 1, 2 and 4 RTX 5090 (idle node, 2026-09-24) and MPI across two H100 nodes; each run's state is bit-identical to the 1-device state (max_abs_diff_vs_1dev = 0).

## 재생산

- `make -C libs/cuda multi`
- `bash tools/e19_run.sh`

## 파일 (17)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E19_multinode_scaling/gpgpu_20260924/e19_gpgpu_20260924-102645.csv` | `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_gpgpu_20260924-102645.csv` | c5d887cccc1c |
| `E19_multinode_scaling/gpgpu_20260924/e19_4card_chain.log` | `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_4card_chain.log` | a834604c2503 |
| `E19_multinode_scaling/gpgpu_20260924/run.log` | `expr/E19_multinode_scaling/data/gpgpu_20260924/run.log` | 69e4a6b43c1c |
| `E19_multinode_scaling/gpgpu_20260924/e19_topo_20260924.txt` | `expr/E19_multinode_scaling/data/gpgpu_20260924/e19_topo_20260924.txt` | b097704ad9d1 |
| `E19_multinode_scaling/gpgpu_20260924/gate_gpgpu_4card_20260924.txt` | `expr/E19_multinode_scaling/data/gpgpu_20260924/gate_gpgpu_4card_20260924.txt` | 69b377d9ceac |
| `E19_multinode_scaling/gpgpu_20260924/nvidia_smi_after.txt` | `expr/E19_multinode_scaling/data/gpgpu_20260924/nvidia_smi_after.txt` | ace900d66463 |
| `E19_multinode_scaling/gpgpu_20260924/nvidia_smi_before.txt` | `expr/E19_multinode_scaling/data/gpgpu_20260924/nvidia_smi_before.txt` | de0e04132772 |
| `E19_multinode_scaling/ktcloud/e19_ktcloud_20260923-204200.csv` | `expr/E19_multinode_scaling/data/ktcloud/e19_ktcloud_20260923-204200.csv` | ddab58e989af |
| `E19_multinode_scaling/ktcloud/kt_chain.log` | `expr/E19_multinode_scaling/data/ktcloud/kt_chain.log` | 853839e7e0ce |
| `E19_multinode_scaling/ktcloud/run.log` | `expr/E19_multinode_scaling/data/ktcloud/run.log` | 3236db313a1f |
| `E19_multinode_scaling/ktcloud/gate_ktcloud_20260923.txt` | `expr/E19_multinode_scaling/data/ktcloud/gate_ktcloud_20260923.txt` | 1d6e978a64c8 |
| `E19_multinode_scaling/ktcloud/gate_mpi_20260923.txt` | `expr/E19_multinode_scaling/data/ktcloud/gate_mpi_20260923.txt` | 9b079897de7c |
| `E19_multinode_scaling/ktcloud/gate_mpi_tke_20260924.txt` | `expr/E19_multinode_scaling/data/ktcloud/gate_mpi_tke_20260924.txt` | 7e4a99ffeabd |
| `E19_multinode_scaling/ktcloud/nvidia_smi_after.txt` | `expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_after.txt` | 766e78592622 |
| `E19_multinode_scaling/ktcloud/nvidia_smi_after_rerun.txt` | `expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_after_rerun.txt` | 766e78592622 |
| `E19_multinode_scaling/ktcloud/nvidia_smi_before.txt` | `expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_before.txt` | 766e78592622 |
| `E19_multinode_scaling/ktcloud/nvidia_smi_before_rerun.txt` | `expr/E19_multinode_scaling/data/ktcloud/nvidia_smi_before_rerun.txt` | 44ec524ca5e0 |
