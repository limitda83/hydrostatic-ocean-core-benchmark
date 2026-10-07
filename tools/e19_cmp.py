#########################################################################
#  Module: e19_cmp                                                       #
#  Description: Compare gathered multi-device/MPI state files with the   #
#               single-device state (E19 gate: max |diff| must be 0).    #
#  Pipeline: tools/e19_gate.sh / e19_run_mpi.sh -> stdout               #
#########################################################################
import glob, sys
import numpy as np
gate_dir = sys.argv[1]
for tok in ("DC", "TKE"):
    ref = np.fromfile(f"{gate_dir}/{tok}_400_single.bin")
    for f in sorted(glob.glob(f"{gate_dir}/{tok}_400_*.bin")):
        if f.endswith("_single.bin") or "_domain" in f or "_init" in f:
            continue
        a = np.fromfile(f)
        print(f"  {f.split('/')[-1]:36s} n={a.size} max|diff|={np.abs(a - ref).max():.3e} n_diff={(a != ref).sum()} finite={np.isfinite(a).all()}")
