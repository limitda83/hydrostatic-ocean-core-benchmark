# E09_performance_index

**질문/내용:** 종합 성능지표 = 반복수(알고리즘) × 반복당 비용(구현·하드웨어), 예측 검증 1.1 %

**보고서:** docs/26_performance_index.md

**노드:** —(E08 데이터)

**상태:** 완료. 2026-09-13 E08 안정화 재측정본으로 **재계산** — 결론 유지(GPU·직렬 중앙값 ≤ 0.6 %, 다중스레드 CPU 3.8~7.4 %). 구판이 그 붕괴를 '캐시 점유'로 적은 것을 **거대페이지 폴백**으로 정정했다(docs/90 N29).

**재현 명령:**
- `python3 tools/performance_index.py output/collected/*.csv --validate --json output/performance_index.json`

**데이터 (`data/`):** performance_index.json (구판), performance_index_stable.json (재측정본)

**코드 리비전:** e79be85 (이 대장을 만든 시점; 실험 시점의 SHA 는 보고서/manifest 참조)

**주의 (docs/90 참조):** —
