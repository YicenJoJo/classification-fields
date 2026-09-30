"""Select an experiment and keep its outputs in a separate directory."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = {
    "affine": ("synthetic_benchmarks.py", "EXP_RUN_FRACTAL"),
    "nonlinear": ("synthetic_benchmarks_nonlinear_ifs.py", "EXP_RUN_FRACTAL"),
    "cifar": ("cifar_clip.py", None),
    "depth": ("synthetic_benchmarks.py", "EXP_RUN_LTRAIN_SWEEP"),
    "scale": ("synthetic_benchmarks.py", "EXP_RUN_S_ABLATION"),
    "branching": ("synthetic_benchmarks.py", "EXP_RUN_K_ABLATION"),
    "ordering": ("synthetic_benchmarks.py", "EXP_RUN_PERM_CHECK"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, help="Output directory; default: results/<experiment>")
    args = parser.parse_args()

    script, flag = EXPERIMENTS[args.experiment]
    output = (args.output or Path("results") / args.experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Select exactly one suite, even if a previous shell run left flags set.
    for _, selection_flag in EXPERIMENTS.values():
        if selection_flag:
            env.pop(selection_flag, None)
    env.pop("EXP_SYNTHETIC_RUN_ORIGINAL", None)
    if flag:
        env[flag] = "1"
    env["EXP_RESULTS_DIR"] = str(output)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    source_dir = Path(__file__).resolve().parent
    print(f"Experiment: {args.experiment}\nOutputs: {output}", flush=True)
    # A separate process preserves the standalone scripts' initialization order.
    completed = subprocess.run([sys.executable, str(source_dir / script)], env=env, cwd=source_dir)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
