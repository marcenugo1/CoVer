"""
Run all 5 benchmark evaluations for the 7B model (Table 1).
Results are written to evaluation/results/ by eval.py.

Container: 2× NVIDIA B200 (179 GB each)
Default:   2 engines, one per GPU — [[0],[1]]

Usage:
    python run_table1_7B.py
    python run_table1_7B.py --model /path/to/ckpt
    python run_table1_7B.py --gpu_groups "[[0,1]]"
    python run_table1_7B.py --datasets CodeContests
    python run_table1_7B.py --dry_run
"""

import os
import sys
import ast
import subprocess
import argparse
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Backbone baseline by default; pass --model /path/to/ckpt to evaluate a trained checkpoint.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Datasets for Table 1 (MBPP-ReasonFlux is the MBPP subset used in the paper)
DATASETS = [
    "LiveBench",
    "MBPP-ReasonFlux",
    "LiveCodeBench",
    "CodeContests",
    "CodeForces",
]

K_CODE = 16

# 2× B200 in this container — one engine per GPU maximises throughput for a 7B model
DEFAULT_GPU_GROUPS = "[[0,1]]"

MAX_MODEL_LEN = 20000
MAX_GENERATION_TOKEN = 10000

# ── Helpers ───────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent          # .../CoVer/evaluation/
EVAL_SCRIPT = EVAL_DIR / "eval.py"


def build_cmd(dataset: str, gpu_groups: str, model: str) -> list[str]:
    return [
        sys.executable, str(EVAL_SCRIPT),
        "--pretrained_model",     model,
        "--dataset",              dataset,
        "--use_api",              "False",
        "--k_code",               str(K_CODE),
        "--gpu_groups",           gpu_groups,
        "--max_model_len",        str(MAX_MODEL_LEN),
        "--max_generation_token", str(MAX_GENERATION_TOKEN),
    ]


def run_dataset(dataset: str, gpu_groups: str, model: str, dry_run: bool) -> bool:
    cmd = build_cmd(dataset, gpu_groups, model)
    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset}")
    print(f"  Command : {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [dry run — skipping execution]")
        return True

    result = subprocess.run(cmd, cwd=str(EVAL_DIR))
    if result.returncode != 0:
        print(f"\n[ERROR] eval.py exited with code {result.returncode} for {dataset}")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Table 1 evaluations for 7B model")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to the model checkpoint to evaluate (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--gpu_groups", type=str, default=DEFAULT_GPU_GROUPS,
        help='vLLM GPU groups, e.g. "[[0,1]]" for one 2-GPU engine (default: [[0],[1]] — 2 engines × 1 B200)'
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS,
        help="Subset of datasets to run (default: all 5)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()

    # Validate gpu_groups parses correctly
    try:
        ast.literal_eval(args.gpu_groups)
    except Exception as e:
        print(f"[ERROR] --gpu_groups is not valid Python: {e}")
        sys.exit(1)

    print(f"\nModel   : {args.model}")
    print(f"GPUs    : {args.gpu_groups}")
    print(f"k_code  : {K_CODE}")
    print(f"BoN     : {SCALE_TUPLE_LIST}")
    print(f"Datasets: {args.datasets}")

    failed = []
    for dataset in args.datasets:
        ok = run_dataset(dataset, args.gpu_groups, args.model, args.dry_run)
        if not ok:
            failed.append(dataset)

    print(f"\n{'='*60}")
    print(f"Done. Results in: {EVAL_DIR / 'results'}/")
    if failed:
        print(f"[FAILED] {failed}")
        sys.exit(1)
    else:
        print("All benchmarks completed successfully.")


if __name__ == "__main__":
    main()
