#########################################################################
#  Module: promote_cases                                                 #
#  Description: Copy the usable data of each promoted experiment into    #
#               paper/cases/<case>/, refusing quarantined files, and     #
#               write a per-case README with source / claim / command.   #
#  Pipeline: expr/E##/data  ->  paper/cases/<case>/  (manuscript input)  #
#########################################################################
"""Promote experiment data into paper/cases/.

Quarantine tags are refused unconditionally: a file whose name carries one
of QUARANTINE never enters paper/cases/, and a SUPERSEDED directory is never
descended into.  --check verifies an existing paper/cases/ instead of writing.
"""
from __future__ import annotations
import argparse, hashlib, logging, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPR = ROOT / "expr"
CASES = ROOT / "paper" / "cases"
QUARANTINE = ("SUPERSEDED", "_RESULT", "_UNGATED", "_IRREPRODUCIBLE", "_INVALID",
              "_CONTAMINATED", "_DISCARDED", "_MISMATCH")
# old min-of-3 helmholtz files (E08 README: "원고 사용 금지")
E08_OLD = {"geo85_helm.txt", "gpgpu_helm.csv", "gpgpu_helm_1024.csv", "gpgpu_helm_1024_check.csv",
           "gpgpu_helm_2048.csv", "h100_helm.csv", "h100_helm_2048.csv", "h100_helm_big.csv"}

# case -> (claims, sources[(expr dir, glob, exclude-set)], reproduce cmds, report docs, one-line claim)
PLAN = {
 "pareto_plane": dict(
    claims="C1, C7",
    sources=[("E12_tier2_pareto", "*.csv", set()),
             # lock_exchange S-slice rows (physical-horizon runs, 400²) live in two E13 gpgpu files
             ("E13_tier2_ladder", "tier2_gpgpu_20260911-141315_localhost_3904676.csv", set()),
             ("E13_tier2_ladder", "tier2_gpgpu_20260911-234805_localhost_2256997.csv", set())],
    docs="docs/30_tier2_pareto.md",
    repro=["python3 tools/tier2_collect.py paper/cases/pareto_plane/data/*/*.csv --out output/tier2_summary",
           "python3 tools/tier2_report.py"],
    claim="허용오차가 스킴 승자를 바꾼다 — 10⁻¹ 에서 semi-음해/split 4~9배, 10⁻³ 에서 양해법; 거친 지형에서 다중격자 V-cycle 4→166 으로 PCG 에 진다 (RTX 5090 CUDA, 400²×30, 3 케이스 × 31 구성)."),
 "hardware_ladder": dict(
    claims="C2, C3",
    sources=[("E13_tier2_ladder", "tier2_*.csv", set()),
             ("E13_tier2_ladder", "geo85_r2.csv", set()),
             ("E13_tier2_ladder", "ktcloud_jax_stab.csv", set()),
             ("E13_tier2_ladder", "ktcloud_openacc_r4*.csv", set()),
             ("E12_tier2_pareto", "tier2_gpgpu_S_current.csv", set()),
             ("E16_mixing_length", "*.csv", set())],
    docs="docs/32_hardware_ladder.md, docs/39_speedup_generation_framework.md",
    repro=["python3 tools/tier2_collect.py paper/cases/hardware_ladder/data/*/*.csv --out output/tier2_summary",
           "python3 tools/audit_crossdevice.py docs/32_hardware_ladder.md --rows output/tier2_summary/rows.json"],
    claim="백엔드 × 하드웨어 × 격자(100²…2000²×30) 사다리, 바이트 동일 50스텝. 물리 완전성이 하드웨어 순위를 바꾸고(C2), 32 GB 카드는 2000² 에서 같은 노드 CPU 에 진다(C3). 세 노드 전부 R4 게이트 통과 이진."),
 "physics_cost": dict(
    claims="C2, C6",
    sources=[("E14_v06_representative", "*.csv", set()), ("E14_v06_representative", "*.json", set())],
    docs="docs/31_physics_cost.md",
    repro=["CUDA_VISIBLE_DEVICES=<idle> BACKENDS=cuda NX=400 NZ=30 STEPS=50 REPEAT=5 bash tools/decompose_physics.sh",
           "python3 tools/verify_v06.py"],
    claim="TKE 폐쇄 + 3차 이류가 스텝에 더하는 비용과 그 분해: 추가 물리의 75~96 % 가 폐쇄(세 격자·두 정밀도·두 시간 스킴). 모듈화의 Amdahl 상한 1.28~3.59배."),
 "precision_axis": dict(
    claims="C4",
    sources=[("E15_precision", "*.csv", set()), ("E18_mixed_precision", "*.csv", set())],
    docs="docs/34_precision_axis.md",
    repro=["GRIDS=\"400 1000 2000\" BACKENDS=\"cuda cuda_sp openacc openacc_sp\" bash tools/precision_axis.sh",
           "python3 tools/make_mixed_cuda.py && (cd libs/cuda && make all3d5-mixed all3d5-mixedcheck)",
           "python3 tools/mixed_report.py --decompose paper/cases/physics_cost/data/E14_v06_representative/decompose_gpgpu_fb_20260915.csv paper/cases/precision_axis/data/E18_mixed_precision/decompose_gpgpu_mixed_fb_20260915.csv --ladder paper/cases/precision_axis/data/E18_mixed_precision/ladder_gpgpu_mixed_20260915.csv"],
    claim="소비자 GPU 의 물리 벌금은 fp64 벌금. fp64 코어 + fp32 폐쇄(mixed) 는 5090 에서 스텝 0.51~0.58×, H100 에서 0.89~0.91×, fp64 와 반복수 동일. 일괄 fp32 는 다중격자에서 수렴하지 않는다(그 행은 RESULT 로 격리, 여기 없음)."),
 "mixing_length_axis": dict(
    claims="C5",
    sources=[("E16_mixing_length", "*.csv", set()), ("E16_mixing_length", "*.json", set())],
    docs="docs/36_mixing_length_axis.md",
    repro=["GRIDS=\"400 1000 2000\" BACKENDS=\"cuda cuda_sp\" bash tools/mxl_axis.sh",
           "python3 tools/shared_node_min.py paper/cases/mixing_length_axis/data/*/ktcloud*.csv"],
    claim="같은 물리, 15배 적은 산술(재귀 혼합길이) 의 값어치가 정밀도·장치·스레드·프레임워크에 따라 0.93~2.72배 — 한 구성의 이득은 일반화되지 않는다."),
 "module_boundary": dict(
    claims="C6",
    sources=[("E17_module_boundary", "*_boundary.csv", set()), ("E17_module_boundary", "*_boundary_static.csv", set())],
    docs="docs/38_module_boundary.md",
    repro=["python3 tools/module_boundary_report.py --decompose paper/cases/physics_cost/data/*/decompose_*.csv --boundary paper/cases/module_boundary/data/*/*_boundary.csv"],
    claim="폐쇄의 실제 발자국(276 값/열)을 호스트로 왕복시키는 비용은 fp32·400² 이상에서 두 카드 모두 폐쇄의 2.1~2.7배(하한). pinned 필수(pageable 은 5090 2.1×, H100 4.9×)."),
 "kernel_matrix": dict(
    claims="C7, C2(대역폭 지배)",
    sources=[("E08_kernel_matrix", "*_r7*.csv", E08_OLD), ("E03_gpu_microbench", "*.json", set())],
    docs="docs/25_kernel_matrix.md, docs/12_gpu_microbench.md",
    repro=["bash tools/helm_matrix.sh", "bash tools/eos_matrix.sh", "python3 tools/gpu_microbench.py"],
    claim="변계수 Helmholtz 해법 × 정밀도 × 하드웨어(반복 9회 안정화 재측정, MAD 병기), EOS 커널 roofline, 5090 roofline 상수(ridge fp64 1.25 / fp32 59.8 FLOP/byte)."),
 "multi_gpu": dict(
    claims="E19 (strong scaling: fb, DC and DC+TKE; 2 and 4 RTX 5090 in one node by peer copy, 2 H100 across two nodes by MPI; bit-identical gate vs 1 device)",
    sources=[("E19_multinode_scaling", "gpgpu_20260924/*.csv", set()), ("E19_multinode_scaling", "gpgpu_20260924/*.log", set()), ("E19_multinode_scaling", "gpgpu_20260924/*.txt", set()),
             ("E19_multinode_scaling", "ktcloud/*.csv", set()), ("E19_multinode_scaling", "ktcloud/*.log", set()), ("E19_multinode_scaling", "ktcloud/*.txt", set())],
    docs="docs/42_multigpu_scaling.md, docs/41_multinode_design.md, docs/03 S12",
    repro=["make -C libs/cuda multi", "bash tools/e19_run.sh"],
    claim="Single-process multi-GPU (y-slab, deep halo H=16) strong scaling of the fb scheme on 1, 2 and 4 RTX 5090 (idle node, 2026-09-24) and MPI across two H100 nodes; each run's state is bit-identical to the 1-device state (max_abs_diff_vs_1dev = 0)."),
 "comm_bound": dict(
    claims="Appendix C (communication constants for the single-node-vs-single-card limitation)",
    sources=[("E11_multigpu_comm", "*.txt", set())],
    docs="docs/28_multigpu_communication.md",
    repro=["make -C libs/cuda comm", "mpirun -np 2 libs/cuda/build/comm_bench"],
    claim="2-rank halo/Allreduce/PCG-iteration/barotropic-substep costs on RTX 5090 (PCIe, one node) — the measured constants behind the statement that multi-device scaling is out of scope but bounded."),
 "verification": dict(
    claims="M",
    sources=[("E05_3d_reference", "*.json", set()), ("E05_3d_reference", "gates_20260915/*", set()),
             ("E07_v05_physics", "*.json", set()), ("E14_v06_representative", "v06_verification*.json", set()),
             ("E16_mixing_length", "v06_verification_recursive*.json", set())],
    docs="docs/40_verification_summary.md",
    repro=["python3 main.py verify", "bash tools/gate3d5.sh", "python3 tools/verify_v05.py --nx 48 --nz 20 --steps 100",
           "python3 tools/verify_v06.py"],
    claim="다섯 구현 × 세 노드가 1e-12(1스텝)/1e-9(전체) 안에서 같은 답을 낸다는 게이트 기록, 수렴차수 V3D-1…5 · V5-1…6 · V6, 반복수 자리까지 일치."),
}

def quarantined(p: Path) -> bool:
    return any(tag in part for part in p.parts for tag in QUARANTINE)

def gather(case: str) -> list[tuple[Path, Path]]:
    out = []
    for exp, pattern, exclude in PLAN[case]["sources"]:
        base = EXPR / exp / "data"
        found = [p for p in sorted(base.glob(pattern)) if p.is_file()]
        if not found:
            raise SystemExit(f"{case}: pattern {exp}/data/{pattern} matched 0 files")
        for p in found:
            if quarantined(p.relative_to(base)) or p.name in exclude:
                continue
            out.append((p, Path(exp) / p.relative_to(base)))
    return out

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]

def write_case(case: str) -> int:
    spec = PLAN[case]; dst = CASES / case; data = dst / "data"
    if dst.exists():
        shutil.rmtree(dst)
    data.mkdir(parents=True)
    files = gather(case)
    rows = []
    for src, rel in files:
        tgt = data / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
        rows.append(f"| `{rel}` | `expr/{src.relative_to(EXPR)}` | {sha(src)} |")
    lines = [f"# {case}", "",
             f"**뒷받침하는 주장:** paper/TARGET.md {spec['claims']}",
             f"**결과 문서:** {spec['docs']}",
             f"**출처 실험:** " + ", ".join(sorted({s[0] for s in spec['sources']})),
             f"**승격:** 2026-09-16, `python3 tools/promote_cases.py` — 격리 태그 {', '.join(QUARANTINE)} 파일 제외",
             "", "## 주장", "", spec["claim"], "",
             "## 재생산", "", *[f"- `{c}`" for c in spec["repro"]], "",
             f"## 파일 ({len(rows)})", "", "| 여기 | 출처 | sha256[:12] |", "|---|---|---|", *rows, ""]
    (dst / "README.md").write_text("\n".join(lines))
    return len(rows)

def check() -> int:
    bad = 0
    for p in CASES.rglob("*"):
        if p.is_file() and quarantined(p.relative_to(CASES)):
            logging.error(f"quarantined file inside paper/cases: {p}"); bad += 1
    for case in PLAN:
        readme = CASES / case / "README.md"
        if not readme.exists():
            logging.error(f"missing {readme}"); bad += 1; continue
        for src, rel in gather(case):
            tgt = CASES / case / "data" / rel
            if not tgt.exists() or sha(tgt) != sha(src):
                logging.error(f"{case}: {rel} missing or differs from expr"); bad += 1
    print(f"paper/cases check: {bad} problems")
    return bad

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.check:
        return 1 if check() else 0
    for case in PLAN:
        n = write_case(case); print(f"{case:20s} {n:3d} files")
    return 1 if check() else 0

if __name__ == "__main__":
    sys.exit(main())
