# hardware_ladder

**뒷받침하는 주장:** paper/TARGET.md C2, C3
**결과 문서:** docs/32_hardware_ladder.md, docs/39_speedup_generation_framework.md
**출처 실험:** E12_tier2_pareto, E13_tier2_ladder, E16_mixing_length
**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 SUPERSEDED, _RESULT, _UNGATED, _IRREPRODUCIBLE, _INVALID, _CONTAMINATED, _DISCARDED, _MISMATCH 파일 제외

## 주장

백엔드 × 하드웨어 × 격자(100²…2000²×30) 사다리, 바이트 동일 50스텝. 물리 완전성이 하드웨어 순위를 바꾸고(C2), 32 GB 카드는 2000² 에서 같은 노드 CPU 에 진다(C3). 세 노드 전부 R4 게이트 통과 이진.

## 재생산

- `python3 tools/tier2_collect.py paper/cases/hardware_ladder/data/*/*.csv --out output/tier2_summary`
- `python3 tools/audit_crossdevice.py docs/32_hardware_ladder.md --rows output/tier2_summary/rows.json`

## 파일 (83)

| 여기 | 출처 | sha256[:12] |
|---|---|---|
| `E13_tier2_ladder/tier2_geo85_20260911-141202_node03_1757748.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-141202_node03_1757748.csv` | ad5f242ad9e3 |
| `E13_tier2_ladder/tier2_geo85_20260911-141203_node04_2082722.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-141203_node04_2082722.csv` | 3f3aa6718507 |
| `E13_tier2_ladder/tier2_geo85_20260911-141610_node03_1759995.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-141610_node03_1759995.csv` | b24f09703814 |
| `E13_tier2_ladder/tier2_geo85_20260911-142122_node04_2086115.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-142122_node04_2086115.csv` | 92e21bbf075f |
| `E13_tier2_ladder/tier2_geo85_20260911-144155_node04_2090545.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-144155_node04_2090545.csv` | 70c654fccb55 |
| `E13_tier2_ladder/tier2_geo85_20260911-185226_node02_903797.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-185226_node02_903797.csv` | 95e7f268c44d |
| `E13_tier2_ladder/tier2_geo85_20260911-185226_node03_1774516.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-185226_node03_1774516.csv` | 2ac64c284a1d |
| `E13_tier2_ladder/tier2_geo85_20260911-185226_node04_2106654.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-185226_node04_2106654.csv` | 5fbb627d3d9d |
| `E13_tier2_ladder/tier2_geo85_20260911-185831_node02_909049.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-185831_node02_909049.csv` | d02041af16b0 |
| `E13_tier2_ladder/tier2_geo85_20260911-190540_node03_1778122.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-190540_node03_1778122.csv` | 418bfc241a09 |
| `E13_tier2_ladder/tier2_geo85_20260911-193356_node02_914433.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-193356_node02_914433.csv` | ec6b503adf29 |
| `E13_tier2_ladder/tier2_geo85_20260911-194448_node02_917900.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-194448_node02_917900.csv` | b8ce3c450110 |
| `E13_tier2_ladder/tier2_geo85_20260911-194454_node04_2113614.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-194454_node04_2113614.csv` | e0b1ad7e079f |
| `E13_tier2_ladder/tier2_geo85_20260911-201225_node02_922705.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-201225_node02_922705.csv` | 7374f323f3f1 |
| `E13_tier2_ladder/tier2_geo85_20260911-211100_node02_929023.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260911-211100_node02_929023.csv` | 20ad3af56f46 |
| `E13_tier2_ladder/tier2_geo85_20260915-004338_node01_1528903.csv` | `expr/E13_tier2_ladder/data/tier2_geo85_20260915-004338_node01_1528903.csv` | e4b4f0f4f9ac |
| `E13_tier2_ladder/tier2_gpgpu_20260911-141315_localhost_3904676.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260911-141315_localhost_3904676.csv` | b960287f4ba9 |
| `E13_tier2_ladder/tier2_gpgpu_20260911-213658_localhost_2164066.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260911-213658_localhost_2164066.csv` | 5d33247a1caa |
| `E13_tier2_ladder/tier2_gpgpu_20260911-234805_localhost_2256997.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260911-234805_localhost_2256997.csv` | 3d9951f2bcde |
| `E13_tier2_ladder/tier2_gpgpu_20260912-061425_localhost_2532216.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-061425_localhost_2532216.csv` | c9fdd095eb67 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-061610_localhost_2533913.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-061610_localhost_2533913.csv` | dd0a6224f255 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-062150_localhost_2537908.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-062150_localhost_2537908.csv` | b6758c08b41a |
| `E13_tier2_ladder/tier2_gpgpu_20260912-063937_localhost_2549188.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-063937_localhost_2549188.csv` | e157a9bbe0b3 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-065956_localhost_2567586.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-065956_localhost_2567586.csv` | bfacf799d5e6 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-070012_localhost_2569048.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-070012_localhost_2569048.csv` | 243c30580d75 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-070031_localhost_2570568.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-070031_localhost_2570568.csv` | ca23fec40ae0 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-070056_localhost_2572187.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-070056_localhost_2572187.csv` | 919b1861e2de |
| `E13_tier2_ladder/tier2_gpgpu_20260912-070212_localhost_2574469.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-070212_localhost_2574469.csv` | b927f2487f8e |
| `E13_tier2_ladder/tier2_gpgpu_20260912-070609_localhost_2578368.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-070609_localhost_2578368.csv` | 714f0efb770d |
| `E13_tier2_ladder/tier2_gpgpu_20260912-093146_localhost_2695496.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-093146_localhost_2695496.csv` | f2422784792f |
| `E13_tier2_ladder/tier2_gpgpu_20260912-095417_localhost_2710799.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-095417_localhost_2710799.csv` | 2916c8b0e319 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-121812_localhost_2799802.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-121812_localhost_2799802.csv` | 939daeb0ab29 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-121831_localhost_2801322.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-121831_localhost_2801322.csv` | 2fcd6a587958 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-121853_localhost_2802931.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-121853_localhost_2802931.csv` | 168c9f980393 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-122011_localhost_2805164.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-122011_localhost_2805164.csv` | d88963e5f37e |
| `E13_tier2_ladder/tier2_gpgpu_20260912-122555_localhost_2811445.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-122555_localhost_2811445.csv` | ff42d832cd0e |
| `E13_tier2_ladder/tier2_gpgpu_20260912-124359_localhost_2825544.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-124359_localhost_2825544.csv` | 2b4e2fac5df9 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-135159_localhost_2869887.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-135159_localhost_2869887.csv` | 7c748f758633 |
| `E13_tier2_ladder/tier2_gpgpu_20260912-145945_localhost_2912132.csv` | `expr/E13_tier2_ladder/data/tier2_gpgpu_20260912-145945_localhost_2912132.csv` | 2f310a4fd5b4 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-141703_main1_2389861.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-141703_main1_2389861.csv` | 5131b0981e6a |
| `E13_tier2_ladder/tier2_ktcloud_20260911-141755_main1_2390682.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-141755_main1_2390682.csv` | 2465c769d42f |
| `E13_tier2_ladder/tier2_ktcloud_20260911-141857_main1_2391466.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-141857_main1_2391466.csv` | 94bd224a6a66 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-142026_main1_2392284.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-142026_main1_2392284.csv` | 1ab01f2b6e56 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-142448_main1_2393346.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-142448_main1_2393346.csv` | 4760d0a265bc |
| `E13_tier2_ladder/tier2_ktcloud_20260911-144404_main1_2401898.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-144404_main1_2401898.csv` | 42c05ff971ef |
| `E13_tier2_ladder/tier2_ktcloud_20260911-144517_main1_2403913.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-144517_main1_2403913.csv` | f9236676c1ce |
| `E13_tier2_ladder/tier2_ktcloud_20260911-144634_main1_2405969.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-144634_main1_2405969.csv` | 6a594a7463ff |
| `E13_tier2_ladder/tier2_ktcloud_20260911-144821_main1_2408067.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-144821_main1_2408067.csv` | e0aff4cdbccb |
| `E13_tier2_ladder/tier2_ktcloud_20260911-145119_main1_2410257.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-145119_main1_2410257.csv` | b00999736fa6 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-185711_main1_2441320.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-185711_main1_2441320.csv` | 5244422155a1 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-190015_main1_2443308.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-190015_main1_2443308.csv` | d2674a32f753 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-190530_main1_2445045.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-190530_main1_2445045.csv` | 36a346cf5b6a |
| `E13_tier2_ladder/tier2_ktcloud_20260911-191740_main1_2447311.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-191740_main1_2447311.csv` | cfd93375e10e |
| `E13_tier2_ladder/tier2_ktcloud_20260911-200707_main1_2452913.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-200707_main1_2452913.csv` | 29e121dbb99d |
| `E13_tier2_ladder/tier2_ktcloud_20260911-235346_main1_2482222.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-235346_main1_2482222.csv` | 1f4651e83b02 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-235624_main1_2486517.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-235624_main1_2486517.csv` | af09c38752e1 |
| `E13_tier2_ladder/tier2_ktcloud_20260911-235912_main1_2490987.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260911-235912_main1_2490987.csv` | a078d7a4770c |
| `E13_tier2_ladder/tier2_ktcloud_20260912-000258_main1_2495507.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-000258_main1_2495507.csv` | 4f4ae5d0b922 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-001414_main1_2500903.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-001414_main1_2500903.csv` | 2ba31546cfd3 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-004236_main1_2507222.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-004236_main1_2507222.csv` | c672a0d46428 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-004344_main1_2507672.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-004344_main1_2507672.csv` | 64bad4186b70 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-004504_main1_2508140.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-004504_main1_2508140.csv` | 5ac176cb95ca |
| `E13_tier2_ladder/tier2_ktcloud_20260912-004701_main1_2508644.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-004701_main1_2508644.csv` | 6d2d9be7d997 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-005221_main1_2509686.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-005221_main1_2509686.csv` | f1b823558bf6 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-014203_main1_2514991.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-014203_main1_2514991.csv` | e8f25f374599 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-020950_main1_2514972.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-020950_main1_2514972.csv` | b6541af182ee |
| `E13_tier2_ladder/tier2_ktcloud_20260912-025056_main1_2525072.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-025056_main1_2525072.csv` | 76b9c05960e2 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-045540_main1_2536923.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-045540_main1_2536923.csv` | bbb7943ea443 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-045700_main1_2538471.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-045700_main1_2538471.csv` | a0c2abba4ff0 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-045819_main1_2540040.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-045819_main1_2540040.csv` | 345730f62ca2 |
| `E13_tier2_ladder/tier2_ktcloud_20260912-050106_main1_2541698.csv` | `expr/E13_tier2_ladder/data/tier2_ktcloud_20260912-050106_main1_2541698.csv` | 06168ae12837 |
| `E13_tier2_ladder/geo85_r2.csv` | `expr/E13_tier2_ladder/data/geo85_r2.csv` | f467ae712f94 |
| `E13_tier2_ladder/ktcloud_jax_stab.csv` | `expr/E13_tier2_ladder/data/ktcloud_jax_stab.csv` | 76dfa7ac3875 |
| `E13_tier2_ladder/ktcloud_openacc_r4_20260915.csv` | `expr/E13_tier2_ladder/data/ktcloud_openacc_r4_20260915.csv` | 953442f6aa64 |
| `E13_tier2_ladder/ktcloud_openacc_r4b_20260915.csv` | `expr/E13_tier2_ladder/data/ktcloud_openacc_r4b_20260915.csv` | eeb5d21e0522 |
| `E12_tier2_pareto/tier2_gpgpu_S_current.csv` | `expr/E12_tier2_pareto/data/tier2_gpgpu_S_current.csv` | 86a899ac662e |
| `E16_mixing_length/geo85.csv` | `expr/E16_mixing_length/data/geo85.csv` | 5caf9cd1b9ab |
| `E16_mixing_length/gpgpu.csv` | `expr/E16_mixing_length/data/gpgpu.csv` | e4a760cedf8b |
| `E16_mixing_length/gpgpu_frameworks.csv` | `expr/E16_mixing_length/data/gpgpu_frameworks.csv` | f5143560e857 |
| `E16_mixing_length/gpgpu_nz.csv` | `expr/E16_mixing_length/data/gpgpu_nz.csv` | d3f5cf4a880d |
| `E16_mixing_length/ktcloud.csv` | `expr/E16_mixing_length/data/ktcloud.csv` | 5efe64567c56 |
| `E16_mixing_length/ktcloud_remeasure.csv` | `expr/E16_mixing_length/data/ktcloud_remeasure.csv` | a0cbe5c01697 |
| `E16_mixing_length/ktcloud_remeasure2.csv` | `expr/E16_mixing_length/data/ktcloud_remeasure2.csv` | ab18d0a552b0 |
