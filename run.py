import os
import sys
import signal
import subprocess
import time
import json
from contextlib import nullcontext
from termcolor import cprint

from optimization import optimization_config
from profiler import StepProfiler

_COVER_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if optimization_config.use_wandb:
    import wandb
    wandb.init(
        project=optimization_config.wandb_project,
        name=optimization_config.wandb_run_name,
        id=optimization_config.wandb_run_id,
        resume="allow" if optimization_config.wandb_run_id else None,
    )

def _log_wandb(step):
    if not optimization_config.use_wandb:
        return
    import json, wandb
    combined = {}
    for fname in ["optimization/wandb_metrics.json", "optimization/wandb_reward_metrics.json", "optimization/wandb_train_metrics.json"]:
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            data.pop("_step", None)
            combined.update(data)
    print(f"[wandb] logging {len(combined)} metrics at step {step}: {list(combined.keys())[:5]}...")
    if combined:
        wandb.log(combined, step=step)
        print(f"[wandb] logged successfully")


# if you are the first time to train the model, set this to be True.
# if you have stopped the process and want to keep training, set this to be False and set resume_step to the last completed checkpoint step.
start_from_scratch = True
resume_step = 0  # set to the last completed training step when resuming
debug_logs = False  # set to True to write sample_step*.log and train_step*.log



eval_interval = optimization_config.eval_interval
save_interval = optimization_config.save_interval
total_steps = optimization_config.total_steps
pretrain_model = optimization_config.pretrained_model
model = os.path.abspath("") + "/optimization/ckpt/" +  optimization_config.optimized_model_name
if start_from_scratch == False:
    pretrain_model = model
eval_dataset = optimization_config.eval_dataset
train_dataset = optimization_config.train_dataset
gpu_groups = optimization_config.gpu_groups
eval_k_code = optimization_config.eval_k_code
eval_num_chunks = optimization_config.eval_num_chunks
eval_no_example = optimization_config.eval_no_example
eval_max_test = optimization_config.eval_max_test


def begin_with(file_name):
    with open(file_name, "w") as f:
        f.write("")

if start_from_scratch:
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("optimization/results", exist_ok=True)
    #begin_with("evaluation/results/results-eval-" + pretrain_model.replace("/", ".") + "-" + eval_dataset + ".txt")
    #begin_with("optimization/results/results-rl-" + pretrain_model.replace("/", ".") + "-" + train_dataset + ".txt")
    begin_with("evaluation/results/results-eval-" + model.replace("/", ".") + "-" + eval_dataset + ".txt")
    begin_with("optimization/results/results-rl-" + model.replace("/", ".") + "-" + train_dataset + ".txt")

# evaluation
def evaluation(model, eval_dataset, gpu_groups):
    cprint(f"This is the {i}-th step for evaluation.", color = "green")
    eval_env = os.environ.copy()
    eval_env.pop("VLLM_PORT", None)
    eval_env["VLLM_BASE_PORT"] = "29510"
    subprocess.run(
        f'python eval.py '
        f'--pretrained_model {model} '
        f'--dataset {eval_dataset} '
        '--use_api False '
        '--exe_verbose False '
        '--is_final_eval False '
        f'--k_code {eval_k_code} '
        f'--num_chunks {eval_num_chunks} '
        f'--no_example {eval_no_example} '
        f'--max_test {eval_max_test} '
        f'--gpu_groups "{repr(gpu_groups)}" ',
        shell=True,
        cwd='evaluation',
        check=True,
        env=eval_env,
    )


# sample — launches sample.py as a Popen (non-blocking) so execute-GT can overlap
def sample_popen(model):
    """Start sample.py in a background Popen and return the process handle."""
    cprint(f"This is the {i}-th step for sampling (async).", color="green")
    proc = subprocess.Popen(
        f'python sample.py --pretrained_model {model}',
        shell=True,
        cwd='optimization',
    )
    return proc


def _wait_for_sentinel(sentinel_path, sample_proc, poll_interval=0.5, timeout=7200):
    """Block until sentinel_path appears or sample_proc dies (error)."""
    import time as _time
    deadline = time.time() + timeout
    while not os.path.exists(sentinel_path):
        ret = sample_proc.poll()
        if ret is not None and ret != 0:
            raise RuntimeError(f"sample.py exited with code {ret} before sentinel appeared")
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for sentinel {sentinel_path}")
        _time.sleep(poll_interval)


def _run_execute(model, mode, label="execution"):
    """Run execute.py with the given --mode flag; raises on non-zero exit."""
    cprint(f"This is the {i}-th step for {label} (mode={mode}).", color="green")
    result = subprocess.run(
        f'python execute.py --pretrained_model {model} --mode {mode}',
        shell=True,
        cwd='optimization',
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            cprint("stdout:", "yellow")
            print(result.stdout)
        if result.stderr:
            cprint("stderr:", "yellow")
            print(result.stderr)
        result.check_returncode()


def execute_and_sample_async(model):
    """Phase 1 async: run execute-GT concurrently with case generation.

    Timeline:
      t=0      sample.py starts (code gen + case gen)
      t≈40s    code gen done → part1 written → sentinel appears
      t=40s    execute-GT starts (GT tests ~33s) ──┐
      t=40s    case gen continues (~35s)            │ overlap!
      t=75s    sample.py exits (case gen done)      │
      t=73s    execute-GT finishes ─────────────────┘
      t=75s    execute-case starts (generated cases ~65s)
      t=140s   execute-case finishes → stats written

    Falls back to sequential sample → execute when COVER_DEBUG_IN_PROCESS=1
    (in-process mode doesn't support Popen-based async).
    """
    if os.environ.get("COVER_DEBUG_IN_PROCESS", "").strip() == "1":
        # In-process debug mode: fall back to sequential for simplicity
        import runpy
        _cwd = os.getcwd()
        _opt_dir = os.path.join(_COVER_ROOT, "optimization")
        try:
            os.chdir(_opt_dir)
            if _opt_dir not in sys.path:
                sys.path.insert(0, _opt_dir)
            sys.argv = ["sample.py", "--pretrained_model", model]
            runpy.run_path("sample.py", run_name="__main__")
        finally:
            os.chdir(_cwd)
        execute(model)
        return

    sentinel_path = os.path.join("optimization", "temp_data", ".code_gen_done")

    # clean up any stale sentinel from a previous crashed run
    if os.path.exists(sentinel_path):
        os.unlink(sentinel_path)

    # launch sample as background process
    sample_proc = sample_popen(model)

    # wait for code-gen sentinel (appears after Phase 1 of sample.py)
    cprint("[async] waiting for code-gen sentinel...", "cyan")
    _wait_for_sentinel(sentinel_path, sample_proc)
    cprint("[async] sentinel detected — launching execute-GT concurrently", "cyan")

    # launch execute-GT as a background thread so case gen can overlap
    import threading
    gt_error = []
    def _run_gt():
        try:
            _run_execute(model, "gt", label="execute-GT")
        except Exception as e:
            gt_error.append(e)

    gt_thread = threading.Thread(target=_run_gt, daemon=True)
    gt_thread.start()

    # wait for sample.py to finish (case gen completes)
    sample_proc.wait()
    if sample_proc.returncode != 0:
        raise RuntimeError(f"sample.py exited with code {sample_proc.returncode}")
    cprint("[async] sample.py done — waiting for execute-GT to finish", "cyan")

    # wait for execute-GT to finish
    gt_thread.join()
    if gt_error:
        raise gt_error[0]
    cprint("[async] execute-GT done — launching execute-case", "cyan")

    # oracle correction: replace hallucinated test oracles with execution consensus
    # (runs only when enable_oracle_correction=True in optimization_config)
    if optimization_config.enable_oracle_correction:
        cprint("[async] running oracle correction...", "cyan")
        _run_execute(model, "oracle_correction", label="oracle-correction")

    # run execute-case (merges GT results, executes generated cases, writes stats)
    _run_execute(model, "case", label="execute-case")


# execute (legacy sequential mode — kept for COVER_DEBUG_IN_PROCESS and backward compat)
def execute(model):
    cprint(f"This is the {i}-th step for execution.", color = "green")
    result = subprocess.run(
        f'python execute.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            cprint("stdout:", "yellow")
            print(result.stdout)
        if result.stderr:
            cprint("stderr:", "yellow")
            print(result.stderr)
        result.check_returncode()

# collect_data: sample (vLLM) → execute (CPU) → reward (CPU)
def collect_data(model, step_label, profiler=None):
    cprint(f"[collect] step {step_label}: sample → execute → reward", color="cyan")

    def _phase(name):
        return profiler.phase(f"collect/{name}") if profiler is not None else nullcontext()

    def _run_exe(mode, label=None):
        """Run execute.py --mode <mode>; raise on non-zero exit."""
        tag = label or mode
        cprint(f"[collect] step {step_label}: execute ({tag})", color="cyan")
        r = subprocess.run(
            f'python execute.py --pretrained_model {model} --mode {mode}',
            shell=True, cwd='optimization', capture_output=True, text=True,
        )
        if r.returncode != 0:
            if r.stdout: cprint("stdout:", "yellow"); print(r.stdout)
            if r.stderr: cprint("stderr:", "yellow"); print(r.stderr)
            r.check_returncode()

    # Pin vLLM's port allocation below the OS ephemeral range (32768+) so it
    # never collides with Ray actor ports from the concurrent training step.
    # DeepSpeed uses MASTER_PORT=29500; VLLM_BASE_PORT=29510 gives each worker
    # a private port range (29510, 29520, ...) while still supporting a single
    # TP=2 worker on [[0, 1]] for larger models.
    with _phase("sample"):
        sample_env = os.environ.copy()
        sample_env.pop("VLLM_PORT", None)
        sample_env["VLLM_BASE_PORT"] = "29510"
        # Kill any stale vLLM processes before launching to prevent port-collision hangs
        subprocess.run("pkill -9 -f 'sample.py'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'EngineCore'", shell=True, stderr=subprocess.DEVNULL)
        sample_log = os.path.join(_COVER_ROOT, f"optimization/results/sample_step{step_label}.log")
        _sample_cmd = f'python sample.py --pretrained_model {model}'
        if debug_logs:
            _sample_cmd += f' 2>&1 | tee -a {sample_log}'
        for attempt in range(2):
            try:
                result = subprocess.run(
                    _sample_cmd,
                    shell=True,
                    cwd='optimization',
                    env=sample_env,
                    timeout=450,  # 5 min hard cap — sampling normally takes ~141s; deadlock hang takes ~300s
                )
                break
            except subprocess.TimeoutExpired:
                cprint(f"[collect] step {step_label}: sample.py FROZEN after 450s (attempt {attempt+1}/2) — killing", color="red")
                subprocess.run("pkill -9 -f 'sample.py'", shell=True)
                subprocess.run("pkill -9 -f 'EngineCore'", shell=True)
                subprocess.run("ray stop --force", shell=True)  # clear stale Ray shared-memory before retry
                if attempt == 1:
                    raise
                cprint(f"[collect] step {step_label}: retrying sample.py (Ray/vLLM deadlock recovery)...", color="yellow")
        if result.returncode != 0:
            result.check_returncode()

    if optimization_config.enable_oracle_correction:
        # Split pipeline required for oracle correction:
        #   prepare_part1 → gt → oracle_correction → case
        # prepare_part1 seeds part1.json from the full outputs file so that
        # execute-GT (which reads part1) can proceed without the async sample
        # two-phase write that the async pipeline relies on.
        with _phase("prepare_part1"):
            _run_exe("prepare_part1", "prepare-part1")
        with _phase("execute_gt"):
            _run_exe("gt", "execute-GT")
        with _phase("oracle_correction"):
            _run_exe("oracle_correction", "oracle-correction")
        with _phase("execute_case"):
            _run_exe("case", "execute-case")
    else:
        # Original all-mode execution — identical to baseline behaviour.
        with _phase("execute_all"):
            result = subprocess.run(
                f'python execute.py --pretrained_model {model}',
                shell=True, cwd='optimization', capture_output=True, text=True,
            )
            if result.returncode != 0:
                if result.stdout: cprint("stdout:", "yellow"); print(result.stdout)
                if result.stderr: cprint("stderr:", "yellow"); print(result.stderr)
                result.check_returncode()

    # reward (CPU-only)
    with _phase("reward"):
        subprocess.run(
            f'python reward.py --pretrained_model {model}',
            shell=True,
            cwd='optimization',
            check=True,
        )

# train: ZeRO-3 data parallel across both GPUs (2 nodes × 1 GPU each)
def train(model, step_label):
    cprint(f"[train] step {step_label}: training on GPU 1", color="cyan")
    # Sequential mode: collect always finishes before train starts, so Ray is
    # never active concurrently with vLLM. Stop any stale Ray state from the
    # previous training step before launching the new one.
    subprocess.run("ray stop --force", shell=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["COVER_STEP"] = str(step_label)
    train_timing_file = os.path.join(_COVER_ROOT, f"optimization/results/train_timing_step{step_label}.jsonl")
    env["COVER_TRAIN_TIMING_FILE"] = train_timing_file
    try:
        os.remove(train_timing_file)
    except FileNotFoundError:
        pass
    # Pin DeepSpeed's torch.distributed rendezvous to a fixed port so it never
    # collides with vLLM's EngineCore which also calls init_process_group on a
    # randomly selected port.  Port 29500 is below the OS ephemeral range
    # (32768+), so get_open_port() inside vLLM will never return it once
    # DeepSpeed has it bound.
    env.setdefault("MASTER_PORT", "29500")
    train_log = os.path.join(_COVER_ROOT, f"optimization/results/train_step{step_label}.log")
    _train_cmd = f'python -m train --pretrain {model}'
    if debug_logs:
        _train_cmd += f' 2>&1 | tee {train_log}'
    try:
        result = subprocess.run(
            _train_cmd,
            shell=True,
            cwd='optimization',
            env=env,
            timeout=900,  # 15 min hard cap — training normally takes ~5 min
        )
    except subprocess.TimeoutExpired:
        cprint(f"[train] step {step_label}: TIMEOUT after 900s — killing and continuing", "red")
        subprocess.run("pkill -9 -f 'python -m train'", shell=True)
        subprocess.run("ray stop --force", shell=True)
        return  # skip this step rather than block forever
    subprocess.run("rm -f optimization/ckpt/event*", shell=True, check=True)
    if result.returncode != 0:
        result.check_returncode()
    return train_timing_file


def summarize_train_timing(step_label, timing_file):
    """Persist a compact per-step summary of train.py's internal timers."""
    if not timing_file or not os.path.exists(timing_file):
        return None

    phase_stats = {}
    records = []
    with open(timing_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            phase = rec.get("phase", "unknown")
            stats = phase_stats.setdefault(
                phase,
                {"count": 0, "duration_s": 0.0, "max_s": 0.0, "errors": 0},
            )
            dur = float(rec.get("duration_s", 0.0) or 0.0)
            stats["count"] += 1
            stats["duration_s"] += dur
            stats["max_s"] = max(stats["max_s"], dur)
            if rec.get("status") != "ok":
                stats["errors"] += 1

    summary = {
        "step": step_label,
        "timing_file": timing_file,
        "records": len(records),
        "phases": {
            phase: {
                "count": stats["count"],
                "duration_s": round(stats["duration_s"], 3),
                "max_s": round(stats["max_s"], 3),
                "errors": stats["errors"],
            }
            for phase, stats in sorted(phase_stats.items())
        },
    }
    summary_path = os.path.join(_COVER_ROOT, "optimization/results/train_timing_summary.jsonl")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary

# save checkpoint
def save(model_from, model_to):
    os.makedirs(model_to, exist_ok=True)
    subprocess.run(f"rm -rf {model_to}/*", shell=True, check=True)
    subprocess.run(f"cp -r {model_from}/* {model_to}/", shell=True, check=True)


# ── Phase 2 pipeline helper ──────────────────────────────────────────────────

def run_step_pipelined(model, train_step, collect_step):
    """Train on data_train_step (GPU 1) concurrently with collecting data_collect_step (GPU 0).

    At entry:
      - rl_data.json contains data for train_step (written by the previous collect)
      - model checkpoint has weights from training train_step-1

    At exit:
      - rl_data.json contains data for collect_step (overwritten at end of collect)
      - model checkpoint has new weights from training train_step
      - GPU 0 used data that existed at vLLM startup (~30s in) so no disk conflict
        with training which writes the checkpoint only at the very end (~200s+)

    Off-policy lag: collect uses the checkpoint that existed at launch time
    (before training completes), so samples are 1 step behind. Acceptable for GRPO.
    """
    import threading

    train_errors = []
    train_dur_box = [0.0]  # mutable box so _train() can write its duration

    # Stop Ray BEFORE launching either thread.  train() previously called
    # ray stop --force at its own start, but that raced with vLLM's EngineCore
    # IPC initialisation running concurrently in collect_data — ray stop destroys
    # POSIX shared-memory segments and zmq handles that vLLM was setting up,
    # leaving llm.generate() deadlocked.  Stopping Ray here, while both GPUs are
    # idle, is safe and removes the race entirely.
    cprint("[pipeline] stopping Ray before launching train+collect threads", "cyan")
    subprocess.run("ray stop --force", shell=True)

    def _train():
        t_train = time.time()
        try:
            train(model, step_label=train_step)
        except Exception as e:
            train_errors.append(e)
        finally:
            train_dur_box[0] = time.time() - t_train

    t0 = time.time()
    train_thread = threading.Thread(target=_train, daemon=True)
    train_thread.start()

    # GPU 0 samples with whatever weights vLLM loads at startup.
    # Training only overwrites the checkpoint at the very end (~200s+),
    # well after vLLM has finished loading (~30s). No disk conflict.
    collect_error = None
    try:
        collect_data(model, step_label=collect_step)
    except Exception as e:
        collect_error = e
    collect_dur = time.time() - t0

    train_thread.join()
    train_dur = train_dur_box[0]
    wall_dur = time.time() - t0

    if train_errors:
        raise train_errors[0]
    if collect_error:
        raise collect_error

    bottleneck = "train" if train_dur >= collect_dur else "collect"
    cprint(
        f"[pipeline] train_step={train_step} collect_step={collect_step}: "
        f"collect={collect_dur:.1f}s  train={train_dur:.1f}s  "
        f"wall={wall_dur:.1f}s  ({bottleneck} was bottleneck)",
        color="magenta",
    )
    return collect_dur, train_dur, wall_dur


# ── Main loop ────────────────────────────────────────────────────────────────
#
# Sequential structure (no pipelining, no warm-up, no n+1 offset):
#
#   Step i:   collect(model_i) using both GPUs via TP=2           ~265s
#             train(model_i)   on GPU 1 only                      ~294s
#             wall = collect + train ≈ 559s
#
# Both collect and train use the same model checkpoint (step_model).
# rl_data.json is written by collect and read by train in the same iteration.

i = 0
skip_to_train = False  # set True to skip collect and jump straight to train (for debugging)

if not start_from_scratch:
    i = resume_step
    cprint(f"Resuming from step {i} using checkpoint optimization/ckpt/{optimization_config.optimized_model_name}", color="cyan")
    # On resume, collect(i) will regenerate fresh data for step i before training.

# ── Sequential training loop ─────────────────────────────────────────────────
# Each iteration: collect(model_i) → train(model_i) → model_{i+1}
# Both phases use the same model checkpoint.  No warm-up needed, no n+1 offset.
while i <= total_steps:

    eval_save_model = pretrain_model if i == 0 else model
    if i % eval_interval == 0 and i > 0:
        evaluation(eval_save_model, eval_dataset, gpu_groups)
    if i % save_interval == 0 and i > 0:
        save(eval_save_model, f"optimization/ckpt/{optimization_config.optimized_model_name}_checkpoints/iter{i}")

    if i == total_steps:
        break

    os.environ["COVER_STEP"] = str(i)
    prof = StepProfiler(step=i)

    # Step 0: checkpoint doesn't exist yet → use pretrain_model.
    # Steps 1+: use local checkpoint updated by the previous training step.
    step_model = pretrain_model if i == 0 else model

    t0 = time.time()
    with prof.phase("collect"):
        collect_data(step_model, step_label=i, profiler=prof)
    collect_dur = time.time() - t0
    t1 = time.time()
    with prof.phase("train"):
        train_timing_file = train(step_model, step_label=i)
    train_dur = time.time() - t1
    wall_dur = time.time() - t0
    train_timing_summary = summarize_train_timing(i, train_timing_file)
    if train_timing_summary:
        prof.data["train_timing"] = train_timing_summary
        top_phases = sorted(
            train_timing_summary["phases"].items(),
            key=lambda item: item[1]["duration_s"],
            reverse=True,
        )[:10]
        prof._log_event(
            "[train/timing] "
            + ", ".join(
                f"{name}={stats['duration_s']:.1f}s"
                + (f"x{stats['count']}" if stats["count"] > 1 else "")
                for name, stats in top_phases
            )
        )

    prof.save()

    step_duration = wall_dur
    step_msg = (
        f"step: {i} | wall: {step_duration:.1f}s | "
        f"collect: {collect_dur:.1f}s | train: {train_dur:.1f}s"
    )
    print(step_msg)
    with open("optimization/results/time_log.txt", "a") as f:
        f.write(step_msg + "\n")
    _log_wandb(i)
    i += 1
