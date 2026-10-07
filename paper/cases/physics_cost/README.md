# physics_cost

**뒷받침하는 주장:** paper/TARGET.md C2, C6
**결과 문서:** docs/31_physics_cost.md
**출처 실험:** E14_v06_representative
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

TKE 폐쇄 + 3차 이류가 스텝에 더하는 비용과 그 분해: 추가 물리의 75~96 % 가 폐쇄(세 격자·두 정밀도·두 시간 스킴). 모듈화의 Amdahl 상한 1.28~3.59배.

## 재생산

- `CUDA_VISIBLE_DEVICES=<idle> BACKENDS=cuda NX=400 NZ=30 STEPS=50 REPEAT=5 bash tools/decompose_physics.sh`
- `python3 tools/verify_v06.py`

## 파일 (8)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E14_v06_representative/decompose_gpgpu_fb_20260915.csv` | `expr/E14_v06_representative/data/decompose_gpgpu_fb_20260915.csv` | 5f55985871dc |
| `E14_v06_representative/decompose_gpgpu_theta_20260915.csv` | `expr/E14_v06_representative/data/decompose_gpgpu_theta_20260915.csv` | f7a6b20a8412 |
| `E14_v06_representative/decompose_ktcloud_fb_20260915.csv` | `expr/E14_v06_representative/data/decompose_ktcloud_fb_20260915.csv` | 17f0bba6222f |
| `E14_v06_representative/advection_checks.json` | `expr/E14_v06_representative/data/advection_checks.json` | 07e1d1dc40e2 |
| `E14_v06_representative/physics_cost.json` | `expr/E14_v06_representative/data/physics_cost.json` | d933b355faab |
| `E14_v06_representative/v06_verification.json` | `expr/E14_v06_representative/data/v06_verification.json` | 9fa02bf5475c |
| `E14_v06_representative/v06_verification_20260914.json` | `expr/E14_v06_representative/data/v06_verification_20260914.json` | 776c4a740de3 |
| `E14_v06_representative/v06_verification_20260923.json` | `expr/E14_v06_representative/data/v06_verification_20260923.json` | 776c4a740de3 |
