# E02_fortran_cpu

**질문/내용:** Fortran CPU 백엔드 R2 게이트, OpenMP 임계값(omp_min_points) 보정, -fopenmp 1스레드 ≠ 직렬

**보고서:** docs/11_p3_fortran_cpu.md

**노드:** gpgpu · geo85

**상태:** done

**재현 명령:**
- `python3 -m tools.compare_backends --set solver.kind=pcg_jacobi --case igw --nx 64 --cfl 0.5`
- `qsub tools/pbs_omp_sweep.sh`

**데이터 (`data/`):** `omp_sweep_igw*_t{1..192}_metrics.json` 27개 (geo85 node01, 512², 1~192 스레드)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
