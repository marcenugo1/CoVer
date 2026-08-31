#!/usr/bin/env python3
"""
Sequential overnight comparison:
  1. Baseline run  (enable_oracle_correction=False, 20 steps)
  2. Oracle run    (enable_oracle_correction=True,  20 steps)

Patches optimization_config.py between runs so no manual editing is needed.
Logs from each run are tee'd to run_baseline.log / run_oracle.log.
"""

import re
import subprocess
import sys
import os

COVER_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(COVER_DIR, "optimization", "optimization_config.py")
RUN_PY_PATH = os.path.join(COVER_DIR, "run.py")
PYTHON      = sys.executable          # whichever env is active


def patch_file(path: str, patches: dict):
    with open(path, "r") as f:
        content = f.read()
    for pattern, replacement in patches.items():
        new_content = re.sub(pattern, replacement, content)
        if new_content == content:
            print(f"  WARNING: pattern did not match in {os.path.basename(path)}: {pattern!r}")
        content = new_content
    with open(path, "w") as f:
        f.write(content)


def patch_config(patches: dict):
    patch_file(CONFIG_PATH, patches)


def run_training(label: str, logfile: str):
    print(f"\n{'='*60}")
    print(f"  Starting {label} training run")
    print(f"  Log: {logfile}")
    print(f"{'='*60}\n", flush=True)

    log_path = os.path.join(COVER_DIR, logfile)
    # Use the env's Python binary directly — avoids conda run activation scripts
    # that can crash on missing cross-compiler packages.
    # set -o pipefail ensures tee cannot mask a non-zero exit from python.
    cmd = f'bash -c "set -o pipefail; {PYTHON} run.py 2>&1 | tee {log_path}"'
    result = subprocess.run(cmd, shell=True, cwd=COVER_DIR)

    if result.returncode != 0:
        print(f"\nERROR: {label} run failed with exit code {result.returncode}")
        print(f"Check {logfile} for details.")
        sys.exit(result.returncode)

    print(f"\n  {label} run completed successfully.")


# ── Baseline run (resume from step 13) ───────────────────────────────────────
# Steps 0-13 completed before the freeze. rl_data.json still has step-13 data
# and GRPO_7B_baseline/ has the step-13 weights, so we resume at step 13
# (re-runs one pipeline step, then continues 14→19 fresh).
print("Patching config + run.py for baseline resume...")
patch_config({
    r'total_steps = \d+':
        'total_steps = 20',
    r'optimized_model_name = ".*?"':
        'optimized_model_name = "GRPO_7B_baseline"',
    r'wandb_project = ".*?"':
        'wandb_project = "CoVer_async_oracle_comparison"',
    r'wandb_run_name = .*':
        'wandb_run_name = pretrained_model.replace("/", "-") + "_baseline"',
    r'enable_oracle_correction = .*':
        'enable_oracle_correction = False',
})
patch_file(RUN_PY_PATH, {
    r'start_from_scratch = (?:True|False)': 'start_from_scratch = False',
    r'resume_step = \d+':                   'resume_step = 13',
})
print("  Done. Running baseline (resume)...")
run_training("baseline", "run_baseline.log")


# ── Oracle correction run (fresh start) ──────────────────────────────────────
print("Patching config + run.py for oracle fresh start...")
patch_config({
    r'optimized_model_name = ".*?"':
        'optimized_model_name = "GRPO_7B_oracle"',
    r'wandb_run_name = .*':
        'wandb_run_name = pretrained_model.replace("/", "-") + "_oracle"',
    r'enable_oracle_correction = .*':
        'enable_oracle_correction = True',
})
patch_file(RUN_PY_PATH, {
    r'start_from_scratch = (?:True|False)': 'start_from_scratch = True',
})
print("  Done. Running oracle correction...")
run_training("oracle", "run_oracle.log")


print("\n" + "="*60)
print("  Both runs completed. Compare results:")
print("  - WandB project: CoVer_async_oracle_comparison")
print("    Runs: *_baseline  vs  *_oracle")
print("  - Checkpoints:")
print("    optimization/ckpt/GRPO_7B_baseline/")
print("    optimization/ckpt/GRPO_7B_oracle/")
print("  - Logs:")
print("    run_baseline.log  /  run_oracle.log")
print("="*60)
