# E01_phase0_2d_schemes

**질문/내용:** 2D 선형 회전 천수 · fb/theta · fft/pcg 게이트와 오차-비용 곡선 (DEV-ONLY 시간)

**보고서:** docs/09_phase0_results.md

**노드:** 로컬(dev)

**상태:** done

**재현 명령:**
- `python3 main.py verify --study time --case igw`
- `python3 main.py bench --case igw`

**데이터 (`data/`):** `verify_{space,time}_igw_<stamp>_{manifest,metrics}.json` — **2026-09-23 재실행분(N44 hypsometry 벡터화 뒤, N35)이 현재 코드의 결과**(공간 수렴차수 2.00)이고 9/9·9/14 분은 비교용이다. 로컬 검증이므로 시간 수치는 보고하지 않는다(R7-1).

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
