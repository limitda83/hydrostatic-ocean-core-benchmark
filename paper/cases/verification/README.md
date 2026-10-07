# verification

**뒷받침하는 주장:** paper/TARGET.md M
**결과 문서:** docs/40_verification_summary.md
**출처 실험:** E05_3d_reference, E07_v05_physics, E14_v06_representative, E16_mixing_length
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

다섯 구현 × 세 노드가 1e-12(1스텝)/1e-9(전체) 안에서 같은 답을 낸다는 게이트 기록, 수렴차수 V3D-1…5 · V5-1…6 · V6, 반복수 자리까지 일치.

## 재생산

- `python3 main.py verify`
- `bash tools/gate3d5.sh`
- `python3 tools/verify_v05.py --nx 48 --nz 20 --steps 100`
- `python3 tools/verify_v06.py`

## 파일 (44)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E05_3d_reference/verify3d_baroclinic_igw_20260910-095515_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260910-095515_manifest.json` | 9aabf16d37a5 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260910-095515_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260910-095515_metrics.json` | a447d5ff2a29 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260910-234155_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260910-234155_manifest.json` | 80693a7b106f |
| `E05_3d_reference/verify3d_baroclinic_igw_20260910-234155_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260910-234155_metrics.json` | 1c12e1cc7bdc |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-012818_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-012818_manifest.json` | 6cbeabb84d03 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-012818_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-012818_metrics.json` | 6a480033b279 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015116_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015116_manifest.json` | c7e6770fe662 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015116_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015116_metrics.json` | ed4cf9b8ad3b |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015132_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015132_manifest.json` | 576b0446ecc4 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015132_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015132_metrics.json` | abc737b2df51 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015150_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015150_manifest.json` | 54dcb6f69c6b |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015150_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015150_metrics.json` | 1e4f873843d2 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015212_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015212_manifest.json` | 76e287bd2a30 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-015212_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-015212_metrics.json` | 584dec8037a2 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-051222_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-051222_manifest.json` | 6787132af747 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-051222_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-051222_metrics.json` | 899a5ea81b36 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-135549_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-135549_manifest.json` | b65d53f2ec3a |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-135549_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-135549_metrics.json` | df2a2df79aa1 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-135633_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-135633_manifest.json` | e8bdfd032828 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-135633_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-135633_metrics.json` | 0d98078d3e21 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-140226_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-140226_manifest.json` | 24a1b103c31f |
| `E05_3d_reference/verify3d_baroclinic_igw_20260911-140226_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260911-140226_metrics.json` | cf0676a3d9c9 |
| `E05_3d_reference/verify3d_baroclinic_igw_20260914-130717_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260914-130717_manifest.json` | 806dbce983ba |
| `E05_3d_reference/verify3d_baroclinic_igw_20260914-130717_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260914-130717_metrics.json` | 9ed7ccb58fda |
| `E05_3d_reference/verify3d_baroclinic_igw_20260914-160138_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260914-160138_manifest.json` | 4cfea5dc2bbb |
| `E05_3d_reference/verify3d_baroclinic_igw_20260914-160138_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260914-160138_metrics.json` | 0c84c9fb65fd |
| `E05_3d_reference/verify3d_baroclinic_igw_20260923-224147_manifest.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260923-224147_manifest.json` | ad2cf98c9f8c |
| `E05_3d_reference/verify3d_baroclinic_igw_20260923-224147_metrics.json` | `expr/E05_3d_reference/data/verify3d_baroclinic_igw_20260923-224147_metrics.json` | c7414fc1be86 |
| `E05_3d_reference/gates_20260915/geo85_gate3d5.log` | `expr/E05_3d_reference/data/gates_20260915/geo85_gate3d5.log` | e50c506b044a |
| `E05_3d_reference/gates_20260915/gpgpu_gate3d5.log` | `expr/E05_3d_reference/data/gates_20260915/gpgpu_gate3d5.log` | 3c17f23726ca |
| `E05_3d_reference/gates_20260915/gpgpu_gate3d5_jax.log` | `expr/E05_3d_reference/data/gates_20260915/gpgpu_gate3d5_jax.log` | 24b0dc284161 |
| `E05_3d_reference/gates_20260915/ktcloud_gate3d5_acc_rebuilt.log` | `expr/E05_3d_reference/data/gates_20260915/ktcloud_gate3d5_acc_rebuilt.log` | 708bf5e26c44 |
| `E05_3d_reference/gates_20260915/ktcloud_gate3d5_cuda_acc_stale.log` | `expr/E05_3d_reference/data/gates_20260915/ktcloud_gate3d5_cuda_acc_stale.log` | 7fd06a79b036 |
| `E05_3d_reference/gates_20260915/ktcloud_gate3d5_jax.log` | `expr/E05_3d_reference/data/gates_20260915/ktcloud_gate3d5_jax.log` | 20f4e046efc2 |
| `E05_3d_reference/gates_20260915/local_gate3d5.log` | `expr/E05_3d_reference/data/gates_20260915/local_gate3d5.log` | 02778da48c16 |
| `E07_v05_physics/v05_verification.json` | `expr/E07_v05_physics/data/v05_verification.json` | 5f5aefad2a83 |
| `E07_v05_physics/v05_verification_20260914.json` | `expr/E07_v05_physics/data/v05_verification_20260914.json` | 15ed2231263e |
| `E07_v05_physics/v05_verification_20260923.json` | `expr/E07_v05_physics/data/v05_verification_20260923.json` | 15ed2231263e |
| `E14_v06_representative/v06_verification.json` | `expr/E14_v06_representative/data/v06_verification.json` | 9fa02bf5475c |
| `E14_v06_representative/v06_verification_20260914.json` | `expr/E14_v06_representative/data/v06_verification_20260914.json` | 776c4a740de3 |
| `E14_v06_representative/v06_verification_20260923.json` | `expr/E14_v06_representative/data/v06_verification_20260923.json` | 776c4a740de3 |
| `E16_mixing_length/v06_verification_recursive.json` | `expr/E16_mixing_length/data/v06_verification_recursive.json` | bbb18b51e227 |
| `E16_mixing_length/v06_verification_recursive_20260914.json` | `expr/E16_mixing_length/data/v06_verification_recursive_20260914.json` | bbb18b51e227 |
| `E16_mixing_length/v06_verification_recursive_20260923.json` | `expr/E16_mixing_length/data/v06_verification_recursive_20260923.json` | bbb18b51e227 |
