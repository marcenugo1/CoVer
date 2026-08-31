#!/usr/bin/env bash
# Run run_table1_7B.py sequentially across multiple checkpoints.
#
# Usage (detached tmux session — survives logout):
#   tmux new -d -s table1 'bash /workspace/storage/codeRL/CoVer/evaluation/run_table1_7B_all.sh; echo ALL DONE; sleep infinity'
#   tmux attach -t table1     # watch live; Ctrl-b d to detach
#   tmux ls                   # check the session is alive
#
# Per-model logs go to evaluation/logs/table1_<tag>_<timestamp>.log
# A failure in one model does NOT abort the rest.

set -u
cd "$(dirname "$0")"
mkdir -p logs

# Use the python of the currently active env. Override if needed, e.g.
#   PYTHON=/path/to/env/bin/python bash run_table1_7B_all.sh
# Calling the interpreter directly (rather than `conda run`) also sidesteps envs
# whose activate.d hooks fail before python ever starts.
PYTHON="${PYTHON:-$(command -v python)}"
if [ -z "$PYTHON" ]; then
  echo "No python found on PATH; activate your env or set PYTHON=..." >&2
  exit 1
fi

# ── EDIT THESE 4 PATHS ────────────────────────────────────────────────────────
MODELS=(
  "/workspace/storage/codeRL/CoVer/optimization/ckpt/Qwen_7B_CoVer_reward_diversity_with_punish"
  "/workspace/storage/codeRL/CoVer/optimization/ckpt/Qwen_7B_CoVer_reward_diversity_with_punish_checkpoints/iter300"
  "/workspace/storage/codeRL/CoVer/optimization/ckpt/Qwen_7B_CoVer_reward_diversity_with_punish_checkpoints/iter250"
  "/workspace/storage/codeRL/CoVer/optimization/ckpt/Qwen_7B_CoVer_reward_diversity_with_punish_checkpoints/iter200"
)
# ──────────────────────────────────────────────────────────────────────────────

# Optional: forward extra args through to run_table1_7B.py
#   e.g. bash run_table1_7B_all.sh --datasets CodeContests
EXTRA_ARGS=("$@")

START_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="logs/table1_summary_${START_TS}.log"
echo "Run started: $(date)"            | tee -a "$SUMMARY"
echo "Models: ${#MODELS[@]}"            | tee -a "$SUMMARY"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}" | tee -a "$SUMMARY"
echo                                    | tee -a "$SUMMARY"

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  IDX=$((i + 1))
  TAG="$(basename "$(dirname "$M")")_$(basename "$M")"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="logs/table1_${TAG}_${TS}.log"

  echo "=== [$IDX/${#MODELS[@]}] $(date) :: START $M ===" | tee -a "$SUMMARY" "$LOG"

  if [[ ! -d "$M" ]]; then
    echo "[WARN] $M does not exist — skipping" | tee -a "$SUMMARY" "$LOG"
    continue
  fi

  "$PYTHON" run_table1_7B.py \
      --model "$M" "${EXTRA_ARGS[@]}" \
      >>"$LOG" 2>&1
  RC=$?

  echo "=== [$IDX/${#MODELS[@]}] $(date) :: END   $M (rc=$RC) ===" | tee -a "$SUMMARY" "$LOG"
  echo                                                              | tee -a "$SUMMARY"
done

echo "Run finished: $(date)" | tee -a "$SUMMARY"
