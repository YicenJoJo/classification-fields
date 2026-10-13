"""Select an experiment and keep its outputs in a separate directory."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = {
    "affine": ("synthetic_benchmarks.py", "EXP_RUN_FRACTAL"),
    "nonlinear": ("synthetic_benchmarks_nonlinear_ifs.py", "EXP_RUN_FRACTAL"),
    "cfg": ("matched_cfg.py", None),
    "depth": ("synthetic_benchmarks.py", "EXP_RUN_LTRAIN_SWEEP"),
    "scale": ("synthetic_benchmarks.py", "EXP_RUN_S_ABLATION"),
    "branching": ("synthetic_benchmarks.py", "EXP_RUN_K_ABLATION"),
    "ordering": ("synthetic_benchmarks.py", "EXP_RUN_PERM_CHECK"),
}

IMAGE_EXPERIMENTS = {
    "cifar": ("cifar100", "cifar10"),
    "natural": ("all",),
    **{name: (name,) for name in ("cifar10", "cifar100", "food101", "aircraft", "inaturalist")},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=[*EXPERIMENTS, *IMAGE_EXPERIMENTS])
    parser.add_argument("--output", type=Path, help="Output directory; default: results/<experiment>")
    args, extra = parser.parse_known_args()
    if extra and args.experiment not in IMAGE_EXPERIMENTS:
        parser.error("unrecognized arguments: " + " ".join(extra))

    output = (args.output or Path("results") / args.experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    if args.experiment in IMAGE_EXPERIMENTS:
        seed_args = [] if any(arg.split("=")[0] in ("--seed", "--trial") for arg in extra) else ["--seed", "all"]
        for dataset in IMAGE_EXPERIMENTS[args.experiment]:
            command = [sys.executable, str(source_dir / "run_natural_experiment.py"),
                       "run", "--dataset", dataset, "--output-root", str(output), *seed_args, *extra]
            completed = subprocess.run(command)
            if completed.returncode:
                return completed.returncode
        return 0

    script, flag = EXPERIMENTS[args.experiment]
    env = os.environ.copy()
    # Select one synthetic suite.
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
    print(f"Experiment: {args.experiment}\nOutputs: {output}", flush=True)
    completed = subprocess.run([sys.executable, str(source_dir / script)], env=env, cwd=source_dir)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
