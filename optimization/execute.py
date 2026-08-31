import io
import os
import sys
import ast
import json
import time
import argparse
import collections
import queue
import signal
import warnings
import resource
import numpy as np
import multiprocessing
from termcolor import cprint

import optimization_config


def _norm_case_text(x):
    return " ".join(str(x).replace("\r", "").split())


def _compute_dup_metrics(per_problem_inputs, per_problem_outputs):
    """Aggregate (input, expected_output) duplication metrics across problems.

    For each problem, considers only generated tests with non-empty (input, output)
    after canonicalization. Reports two rates:
      - dup_io_rate              : fraction of tests whose (input, output) tuple is
                                   shared with ≥1 sibling test in the same problem.
                                   High = redundant repetition of identical tests.
      - conflict_input_test_rate : fraction of tests whose input is shared with a
                                   sibling test that has a *different* expected output.
                                   High = "diversity-by-fabrication" pathology where
                                   the test model emits multiple inconsistent answers
                                   for the same input.

    `per_problem_inputs` / `per_problem_outputs` are lists of lists of generated-test
    fields (one outer list per problem); the inner lists must be aligned position-wise.
    """
    from collections import defaultdict

    n_total = 0
    n_dup_io = 0
    n_conflict = 0

    for inputs, outputs in zip(per_problem_inputs, per_problem_outputs):
        if not inputs or not outputs:
            continue
        io_counts = defaultdict(int)
        input_outputs = defaultdict(set)
        valid = []
        for inp, out in zip(inputs, outputs):
            if inp is None or out is None:
                continue
            ni = _norm_case_text(inp)
            no = _norm_case_text(out)
            if not ni or not no:
                continue
            io_counts[(ni, no)] += 1
            input_outputs[ni].add(no)
            valid.append((ni, no))
        for ni, no in valid:
            n_total += 1
            if io_counts[(ni, no)] > 1:
                n_dup_io += 1
            if len(input_outputs[ni]) > 1:
                n_conflict += 1

    safe = lambda d, n: (d / n) if n > 0 else 0.0
    return {
        "n_total_tests": n_total,
        "n_dup_io_tests": n_dup_io,
        "n_conflict_input_tests": n_conflict,
        "dup_io_rate": safe(n_dup_io, n_total),
        "conflict_input_test_rate": safe(n_conflict, n_total),
    }


def _slice_generated_case_fields(entry, keep_generated):
    """Keep all GT columns plus selected generated-case columns."""
    n_gt = entry["num_ground_truth_test"]
    keep_full = list(range(n_gt)) + [n_gt + j for j in keep_generated]

    full_fields = [
        "all_case_input",
        "all_case_output",
        "case_parse_ok",
        "oracle_status",
        "oracle_corrected",
        "model_oracle_match",
        "oracle_confidence",
    ]
    generated_fields = [
        "case_input",
        "case_output",
        "case_text",
        "full_case_generation",
        "case_response_length",
        "oracle_logprob",
    ]

    for field in full_fields:
        if field in entry and isinstance(entry[field], list) and len(entry[field]) >= max(keep_full, default=-1) + 1:
            entry[field] = [entry[field][idx] for idx in keep_full]

    for field in generated_fields:
        if field in entry and isinstance(entry[field], list):
            entry[field] = [entry[field][idx] for idx in keep_generated if idx < len(entry[field])]

    if entry.get("all_case_bool_table") is not None:
        entry["all_case_bool_table"] = entry["all_case_bool_table"][:, keep_full]
    if entry.get("all_case_exe_results") is not None:
        entry["all_case_exe_results"] = [
            [row[idx] for idx in keep_full]
            for row in entry["all_case_exe_results"]
        ]


def prune_generated_cases_by_duplication(data):
    """Prune over-sampled generated tests to k_case_keep using input/pass-signature duplication."""
    keep_n = globals().get("k_case_keep", getattr(optimization_config, "k_case_keep", None))
    if keep_n is None:
        return data

    pruned_problems = 0
    pruned_cases = 0
    for entry in data:
        table = entry.get("all_case_bool_table")
        if table is None:
            continue
        if not isinstance(table, np.ndarray):
            table = np.array(table, dtype=bool)
            entry["all_case_bool_table"] = table

        n_gt = entry["num_ground_truth_test"]
        n_all = table.shape[1]
        n_generated = n_all - n_gt
        if n_generated <= keep_n:
            continue

        input_keys = [
            _norm_case_text(entry["all_case_input"][n_gt + j])
            if n_gt + j < len(entry.get("all_case_input", [])) else ""
            for j in range(n_generated)
        ]
        pass_sigs = [
            tuple(bool(table[row_idx, n_gt + j]) for row_idx in range(table.shape[0]))
            for j in range(n_generated)
        ]
        input_counts = collections.Counter(input_keys)
        pass_counts = collections.Counter(pass_sigs)

        scored = []
        for j in range(n_generated):
            output = entry["all_case_output"][n_gt + j] if n_gt + j < len(entry.get("all_case_output", [])) else ""
            parse_ok = (
                bool(entry["case_parse_ok"][n_gt + j])
                if n_gt + j < len(entry.get("case_parse_ok", [])) else True
            )
            invalid = (not parse_ok) or (not input_keys[j]) or (not _norm_case_text(output))
            badness = (
                1000 * int(invalid)
                + 100 * (input_counts[input_keys[j]] - 1)
                + 10 * (pass_counts[pass_sigs[j]] - 1)
            )
            scored.append((badness, j))

        keep_generated = sorted(j for _, j in sorted(scored)[:keep_n])
        _slice_generated_case_fields(entry, keep_generated)
        pruned_problems += 1
        pruned_cases += n_generated - len(keep_generated)

    if pruned_problems:
        cprint(
            f"[case-prune] pruned {pruned_cases} generated tests across "
            f"{pruned_problems} problems (keep={keep_n})",
            "cyan",
        )
    return data




####### execute the scripts with unit tests #########

class _ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ExecutionTimeout("Timeout Error")


def _execute_script_once(script, input_val, time_limit, compile_cache=None):
    """Execute one generated script with a per-task timeout.

    This runs inside a worker process.  Generated programs are untrusted and
    often malformed, so all ordinary exceptions are converted into result
    strings matching the legacy executor.
    """
    input_lines = iter(input_val.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)

    context = {
        "__name__": "__main__",
        "input": fake_input,
    }

    old_handler = None
    timer_was_set = False
    try:
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        if hasattr(signal, "SIGALRM") and time_limit is not None and time_limit > 0:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            # Give a small grace window so legacy 1s limits do not become
            # stricter due to signal/queue overhead.
            signal.setitimer(signal.ITIMER_REAL, float(time_limit) + 0.05)
            timer_was_set = True

        if compile_cache is not None:
            code = compile_cache.get(script)
            if code is None:
                code = compile(script, "<generated>", "exec")
                if len(compile_cache) > 4096:
                    compile_cache.clear()
                compile_cache[script] = code
            exec(code, context)
        else:
            exec(script, context)
        return stdout_capture.getvalue()

    except _ExecutionTimeout:
        return "Timeout Error"
    except SystemExit:
        return stdout_capture.getvalue()
    except Exception as e:
        return f"errorType: {type(e).__name__} error: {e}"
    finally:
        if timer_was_set:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        sys.stdout = original_stdout
        sys.stdin = original_stdin

def worker(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"errorType: {type(e).__name__} error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def _pool_worker(worker_id, task_queue, result_queue):
    compile_cache = {}
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, script, input_val, time_limit = task
        try:
            result_queue.put(("START", worker_id, idx, time.time()))
            code_obj = compile_cache.get(script)
            if code_obj is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    code_obj = compile(script, "<generated>", "exec")
                if len(compile_cache) > 4096:
                    compile_cache.clear()
                compile_cache[script] = code_obj

            parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
            proc = multiprocessing.Process(
                target=_child_exec,
                args=(child_conn, code_obj, input_val, time_limit),
            )
            proc.daemon = False
            proc.start()
            child_conn.close()

            wall_limit = float(time_limit or 1.0) + 1.0
            proc.join(timeout=wall_limit)

            if proc.is_alive():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.join(timeout=0.3)
                result = "Timeout Error"
            elif parent_conn.poll(0):
                try:
                    result = parent_conn.recv()
                except Exception as e:
                    result = f"Execution Error: {e}"
            else:
                result = f"Execution Error: child exited {proc.exitcode}"
            parent_conn.close()
        except BaseException as e:
            # Last-ditch guard: never let one generated program silently kill
            # the worker for ordinary Python-level failures.
            result = f"Execution Error: worker {worker_id}: {type(e).__name__}: {e}"
        result_queue.put(("DONE", worker_id, idx, result))


def _child_exec(conn, code_obj, input_val, time_limit):
    input_lines = iter(input_val.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)

    context = {
        "__name__": "__main__",
        "input": fake_input,
    }

    result = "Timeout Error"
    old_handler = None
    timer_was_set = False
    try:
        try:
            os.setsid()
        except Exception:
            pass

        try:
            cpu_cap = max(1, int(float(time_limit or 1.0)) + 2)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap + 1))
        except Exception:
            pass

        warnings.filterwarnings("ignore", category=SyntaxWarning)
        if hasattr(signal, "SIGALRM") and time_limit is not None and time_limit > 0:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, float(time_limit) + 0.05)
            timer_was_set = True

        exec(code_obj, context)
        result = stdout_capture.getvalue()
    except _ExecutionTimeout:
        result = "Timeout Error"
    except SystemExit:
        result = stdout_capture.getvalue()
    except Exception as e:
        result = f"errorType: {type(e).__name__} error: {e}"
    finally:
        if timer_was_set:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
        if old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
        sys.stdout = original_stdout
        sys.stdin = original_stdin

    try:
        conn.send(result)
    except Exception:
        pass
    conn.close()


def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x, y):
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    chunk_size = n // num_chunks
    remainder = n % num_chunks
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        indices.append((start, end))
        start = end
    return indices

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks):

    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    i = 0
    for start, end in chunks:
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(sub_code_list, sub_test_input_list, sub_time_limit_list, worker)
        exe_results = exe_results + sub_exe_results
        i += 1
    return exe_results


def run_scripts_with_pool(code_list, test_input_list, time_limit_list, num_workers):
    total = len(code_list)
    if total == 0:
        return []

    num_workers = max(1, min(int(num_workers), total))
    task_queues = [multiprocessing.Queue(maxsize=2) for _ in range(num_workers)]
    result_queue = multiprocessing.Queue()
    processes = []
    in_flight = {}
    results = [None] * total
    next_idx = 0
    completed = 0

    def hard_timeout_s(time_limit):
        limit = float(time_limit or 1.0)
        # Parent-side backstop: looser than the task limit, but no longer as
        # permissive as 9s for 1s tasks.
        return max(limit + 3.5, limit * 4.5)

    def start_worker(worker_id):
        p = multiprocessing.Process(
            target=_pool_worker,
            args=(worker_id, task_queues[worker_id], result_queue),
        )
        p.daemon = False
        p.start()
        return p

    processes = [start_worker(i) for i in range(num_workers)]

    def assign(worker_id):
        nonlocal next_idx
        if next_idx >= total:
            return False
        idx = next_idx
        task = (idx, code_list[idx], test_input_list[idx], time_limit_list[idx])
        task_queues[worker_id].put(task)
        now = time.time()
        in_flight[worker_id] = {
            "idx": idx,
            "queued_at": now,
            "start": None,
            "limit": float(time_limit_list[idx] or 1.0),
            "deadline": None,
        }
        next_idx += 1
        return True

    try:
        for worker_id in range(num_workers):
            assign(worker_id)

        last_progress = time.time()
        while completed < total:
            try:
                message = result_queue.get(timeout=0.2)
                msg_type = message[0]
                if msg_type == "START":
                    _, worker_id, idx, start_ts = message
                    info = in_flight.get(worker_id)
                    if info is not None and info["idx"] == idx:
                        info["start"] = start_ts
                        info["deadline"] = start_ts + hard_timeout_s(info["limit"])
                elif msg_type == "DONE":
                    _, worker_id, idx, result = message
                    info = in_flight.get(worker_id)
                    if info is None or info["idx"] != idx:
                        continue
                    if results[idx] is None:
                        results[idx] = result
                        completed += 1
                    in_flight.pop(worker_id, None)
                    if worker_id < len(processes) and processes[worker_id].is_alive():
                        assign(worker_id)
                    last_progress = time.time()
            except queue.Empty:
                pass

            for worker_id, proc in enumerate(processes):
                if proc.is_alive():
                    continue
                info = in_flight.pop(worker_id, None)
                idx = info["idx"] if info else None
                if idx is not None and results[idx] is None:
                    results[idx] = f"Execution Error: worker exited with code {proc.exitcode}"
                    completed += 1
                if next_idx < total:
                    task_queues[worker_id] = multiprocessing.Queue(maxsize=2)
                    processes[worker_id] = start_worker(worker_id)
                    assign(worker_id)

            now = time.time()
            for worker_id, info in list(in_flight.items()):
                if info["deadline"] is None or now < info["deadline"]:
                    continue
                proc = processes[worker_id]
                idx = info["idx"]
                elapsed = now - info["start"]
                cprint(
                    f"[execute-pool] hard-timeout worker={worker_id} idx={idx} "
                    f"elapsed={elapsed:.2f}s limit={info['limit']:.2f}s",
                    "red",
                )
                if proc.is_alive():
                    if hasattr(proc, "kill"):
                        proc.kill()
                    else:
                        proc.terminate()
                    proc.join(timeout=1)
                if results[idx] is None:
                    results[idx] = "Timeout Error"
                    completed += 1
                in_flight.pop(worker_id, None)
                if next_idx < total:
                    task_queues[worker_id] = multiprocessing.Queue(maxsize=2)
                    processes[worker_id] = start_worker(worker_id)
                    assign(worker_id)

            # If no output has arrived for a long time, surface progress rather
            # than silently looking hung.  Per-task SIGALRM should prevent this.
            if time.time() - last_progress > 120:
                pending = total - completed
                cprint(f"[execute-pool] waiting: completed={completed}/{total}, pending={pending}", "yellow")
                last_progress = time.time()

        return [r if r is not None else "Execution Error: missing result" for r in results]

    finally:
        for q in task_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in processes:
            if p.is_alive():
                p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=1)


def run_scripts(code_list, test_input_list, time_limit_list, worker, num_chunks):
    if globals().get("use_executor_pool", True):
        return run_scripts_with_pool(
            code_list,
            test_input_list,
            time_limit_list,
            globals().get("num_executors", multiprocessing.cpu_count()),
        )
    return run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks)

def _compute_and_save_stats(data, outputs_name):
    """Compute stats from data (which must have all_case_bool_table as np.ndarray) and write results."""

    stats_single = {
        "BoN_score": 0,
        "BoN_num": 0,
        "BoN_acc_score": 0,
        "BoN_acc_num": 0
    }
    stats = []
    for i in range(len(scale_tuple_list)):
        stats_i = stats_single.copy()
        stats_i["tuple"] = scale_tuple_list[i]
        stats.append(stats_i)
    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0
    case_score = 0
    case_num = 0
    case_acc_score = 0
    case_acc_num = 0
    p_01_score = 0
    p_01_num = 0
    p_00_score = 0
    p_00_num = 0
    for i in range(len(data)):
        if data[i]["all_case_exe_results"] is None:
            continue
        t = data[i]["num_ground_truth_test"]
        all_test_table_i = data[i]["all_case_bool_table"][:, :t].copy()
        all_case_table_i = data[i]["all_case_bool_table"][:, t:].copy()
        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]
        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]
        sub_case_table_i = all_case_table_i[correct_code_list, :].copy()
        correct_case_list = np.where(sub_case_table_i.all(axis=0))[0].tolist()
        if len(correct_code_list) > 0:
            case_score += len(correct_case_list)
            case_num += sub_case_table_i.shape[1]
            case_acc_score += np.sum(sub_case_table_i).item()
            case_acc_num += sub_case_table_i.shape[0] * sub_case_table_i.shape[1]
            # get ps
            wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
            wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
            if len(wrong_code_list) > 0:
                if len(correct_case_list) > 0:
                    wrong_code_correct_case_table_i = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                    p_01_score += np.sum(~wrong_code_correct_case_table_i).item()
                    p_01_num += wrong_code_correct_case_table_i.shape[0] * wrong_code_correct_case_table_i.shape[1]
                if len(wrong_case_list) > 0:
                    wrong_code_wrong_case_table_i = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                    p_00_score += np.sum(wrong_code_wrong_case_table_i).item()
                    p_00_num += wrong_code_wrong_case_table_i.shape[0] * wrong_code_wrong_case_table_i.shape[1]

        index_id = 0
        for scale_num_code, scale_num_case in scale_tuple_list:
            case_table_i = all_case_table_i[:scale_num_code, :scale_num_case].copy()
            test_table_i = all_test_table_i[:scale_num_code, :].copy()
            best_code_index = np.sum(case_table_i, 1).argmax()
            sub_test_table_i = test_table_i[best_code_index, :].copy()
            stats[index_id]["BoN_score"] = stats[index_id]["BoN_score"] + int(all(sub_test_table_i))
            stats[index_id]["BoN_num"] = stats[index_id]["BoN_num"] + 1
            stats[index_id]["BoN_acc_score"] = stats[index_id]["BoN_acc_score"] + np.sum(sub_test_table_i).item()
            stats[index_id]["BoN_acc_num"] = stats[index_id]["BoN_acc_num"] + len(sub_test_table_i)
            assert int(all(sub_test_table_i)) / 1 <= np.sum(sub_test_table_i).item() / len(sub_test_table_i), "error"
            index_id += 1

    # --- generated-test duplicate / conflict metrics ---
    # Slice off the GT prefix (first num_ground_truth_test columns) and look only at
    # generated-test fields, which is what the test model is trained on.
    per_problem_inputs = []
    per_problem_outputs = []
    for entry in data:
        if entry.get("all_case_exe_results") is None:
            continue
        t = entry["num_ground_truth_test"]
        all_in = entry.get("all_case_input") or []
        all_out = entry.get("all_case_output") or []
        per_problem_inputs.append(all_in[t:])
        per_problem_outputs.append(all_out[t:])
    dup_metrics = _compute_dup_metrics(per_problem_inputs, per_problem_outputs)

    os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
    with open("./results/results-" + outputs_name + ".txt", "a") as f:
        # Save + print
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        # Step header — makes the appended summary file easier to scan as runs grow.
        step = int(os.environ.get("COVER_STEP", 0))
        save_and_print(f"==================== step {step} ====================")

        # Your values
        def safe_divide(d1, d2):
            if d2 == 0:
                return 0
            return d1/d2
        code_acc = safe_divide(code_score, code_num)
        code_acc_acc = safe_divide(code_acc_score, code_acc_num)
        case_acc = safe_divide(case_score, case_num)
        case_acc_acc = safe_divide(case_acc_score, case_acc_num)
        p_01 = safe_divide(p_01_score, p_01_num)
        p_00 = safe_divide(p_00_score, p_00_num)

        save_and_print(f"code acc: {code_acc}, code accumulate acc: {code_acc_acc}")
        save_and_print(f"case acc: {case_acc}, case accumulate acc: {case_acc_acc}")
        save_and_print(f"p_01: {1 - p_01}")
        save_and_print(f"p_00: {p_00}")
        save_and_print(
            f"generated-test duplication: dup_io_rate={dup_metrics['dup_io_rate']:.4f} "
            f"({dup_metrics['n_dup_io_tests']}/{dup_metrics['n_total_tests']}), "
            f"conflict_input_test_rate={dup_metrics['conflict_input_test_rate']:.4f} "
            f"({dup_metrics['n_conflict_input_tests']}/{dup_metrics['n_total_tests']})"
        )

        # --- unit test error analysis ---
        err_counts = collections.Counter()
        total_cells = 0
        for entry in data:
            if entry["all_case_exe_results"] is None:
                continue
            n_gt = entry["num_ground_truth_test"]
            oracle_discarded = entry.get("oracle_discarded", [])
            for row_exe, row_bool in zip(entry["all_case_exe_results"], entry["all_case_bool_table"]):
                for k, (cell, passed) in enumerate(zip(row_exe, row_bool)):
                    if k < n_gt:
                        continue  # skip ground-truth test slots, only analyse generated tests
                    total_cells += 1
                    if passed:
                        err_counts["correct"] += 1
                    elif cell == "":
                        if k < len(oracle_discarded) and oracle_discarded[k]:
                            err_counts["oracle_discarded"] += 1
                        else:
                            err_counts["parse_fail"] += 1
                    elif "Timeout" in cell or "timeout" in cell:
                        err_counts["timeout"] += 1
                    elif cell.startswith("errorType:"):
                        # format: "errorType: {ExcType} error: {message}"
                        rest = cell[len("errorType:"):].strip()
                        exc_type = rest.split(" error: ", 1)[0].strip()
                        err_counts[exc_type] += 1
                    else:
                        err_counts["wrong_output"] += 1
        save_and_print(f"--- unit test error analysis (generated tests only, total={total_cells}) ---")
        for err_type, count in err_counts.most_common():
            save_and_print(f"  {err_type}: {count} ({100*count/total_cells:.1f}%)")
        save_and_print(f"---")

        if optimization_config.use_wandb:
            bon_metrics = {}
            for s in stats:
                label = f"BoN_{s['tuple']}"
                bon_metrics[f"{label}/acc"] = s["BoN_score"] / s["BoN_num"]
                bon_metrics[f"{label}/acc_acc"] = s["BoN_acc_score"] / s["BoN_acc_num"]
            err_metrics = {f"ut_errors/{k}": v / total_cells for k, v in err_counts.items()} if total_cells > 0 else {}
            wandb_metrics = {
                "train/code_acc": code_acc,
                "train/code_acc_acc": code_acc_acc,
                "train/case_acc": case_acc,
                "train/case_acc_acc": case_acc_acc,
                "train/p_01": 1 - p_01,
                "train/p_00": p_00,
                "train/dup_io_rate": dup_metrics["dup_io_rate"],
                "train/conflict_input_test_rate": dup_metrics["conflict_input_test_rate"],
                **bon_metrics,
                **err_metrics,
                "_step": step,
            }
            import json as _json
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_metrics.json"), "w") as _f:
                _json.dump(wandb_metrics, _f)

        for i in range(len(stats)):
            tuple_name = stats[i]["tuple"]
            save_and_print(f"BoN setting {tuple_name}:")
            acc = stats[i]["BoN_score"] / stats[i]["BoN_num"]
            acc_acc = stats[i]["BoN_acc_score"] / stats[i]["BoN_acc_num"]
            save_and_print(f"acc: {acc}, accumulate acc: {acc_acc}")


    # convert np to list
    for i in range(len(data)):
        if data[i]["all_case_exe_results"] is None:
            continue
        data[i]["all_case_bool_table"] = data[i]["all_case_bool_table"].tolist()

    # output the data
    os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
    with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # save a checkpoint copy every 10 steps for analysis
    step = int(os.environ.get("COVER_STEP", 0))
    if step % 10 == 0:
        ckpt_dir = "./results/output_checkpoints/" + outputs_name
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"output_step_{step}.json")
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def execute_scripts(outputs_name, num_chunks):

    os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + '.json'), exist_ok=True)
    with open("./temp_data/outputs-" + outputs_name + '.json', 'r') as f:
        data = json.load(f)

    # get input lists
    index_list = []
    position_list = []
    code_list = []
    case_input_list = []
    case_output_list = []
    time_limit_list = []
    for i in range(len(data)):
        if len(data[i]["all_case_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["all_case_exe_results"] = None
            data[i]["all_case_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["all_case_input"])
            data[i]["all_case_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["all_case_bool_table"] = np.full((n_row, n_col), False, dtype=bool)

        data_i = data[i]
        for j in range(len(data_i["generated_code"])):
            for k in range(len(data_i["all_case_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["all_case_input"][k]
                case_output = data_i["all_case_output"][k]
                code_list.append(code)
                case_input_list.append(case_input)
                case_output_list.append(case_output)
                time_limit_list.append(1)
                index_list.append(i)
                position_list.append((j, k))
    
    # execute
    exe_results = run_scripts(code_list, case_input_list, time_limit_list, worker, num_chunks)

    for i in range(len(index_list)):
        index_i = index_list[i]
        j, k = position_list[i]
        data[index_i]["all_case_exe_results"][j][k] = exe_results[i]
        data[index_i]["all_case_bool_table"][j][k] = test_if_eq(exe_results[i], case_output_list[i])

    data = prune_generated_cases_by_duplication(data)
    _compute_and_save_stats(data, outputs_name)


def execute_scripts_gt(outputs_name, num_chunks):
    """Execute only the ground-truth test columns from the part1 partial file.

    Results are stored back in the part1 file as ``gt_exe_results`` and
    ``gt_bool_table`` so that ``execute_scripts_case`` can merge them later.
    """
    partial_path = "./temp_data/outputs-" + outputs_name + "-part1.json"
    with open(partial_path, "r") as f:
        data = json.load(f)

    index_list = []
    position_list = []
    code_list = []
    input_list = []
    output_list = []
    time_limit_list = []

    for i in range(len(data)):
        n_gt = data[i]["num_ground_truth_test"]
        n_code = len(data[i]["generated_code"])
        if n_gt == 0 or n_code == 0:
            data[i]["gt_exe_results"] = None
            data[i]["gt_bool_table"] = None
            continue
        data[i]["gt_exe_results"] = [["" for _ in range(n_gt)] for _ in range(n_code)]
        data[i]["gt_bool_table"] = np.full((n_code, n_gt), False, dtype=bool)
        for j in range(n_code):
            for k in range(n_gt):
                code_list.append(data[i]["generated_code"][j])
                input_list.append(data[i]["all_case_input"][k])
                output_list.append(data[i]["all_case_output"][k])
                time_limit_list.append(1)
                index_list.append(i)
                position_list.append((j, k))

    exe_results = run_scripts(code_list, input_list, time_limit_list, worker, num_chunks)

    for idx in range(len(index_list)):
        i = index_list[idx]
        j, k = position_list[idx]
        data[i]["gt_exe_results"][j][k] = exe_results[idx]
        data[i]["gt_bool_table"][j][k] = test_if_eq(exe_results[idx], output_list[idx])

    # convert np arrays to list for JSON serialization
    for i in range(len(data)):
        if data[i]["gt_bool_table"] is not None:
            data[i]["gt_bool_table"] = data[i]["gt_bool_table"].tolist()

    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    cprint(f"[execute-GT] done — GT results saved to {partial_path}", "cyan")


def execute_scripts_prepare_part1(outputs_name):
    """Create part1.json from the full sample outputs file.

    In the async pipeline, sample.py writes part1.json mid-run (after code gen,
    before case gen). In the sequential pipeline used by collect_data(), sample.py
    finishes completely before any execution starts, so part1.json does not exist.
    This function creates it by copying the full outputs file.

    execute_scripts_gt reads part1.json and only accesses all_case_input/output at
    indices k < num_ground_truth_test, which are always the GT entries placed at the
    start of the list by sample.py. So the full file is a valid part1 source.
    """
    full_path = "./temp_data/outputs-" + outputs_name + ".json"
    partial_path = "./temp_data/outputs-" + outputs_name + "-part1.json"

    with open(full_path, "r") as f:
        data = json.load(f)
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    cprint(f"[prepare-part1] wrote {partial_path} from {full_path}", "cyan")


def execute_scripts_oracle_correction(outputs_name, num_chunks,
                                      topk, confidence_threshold,
                                      min_quality, min_total_weight):
    """Correct hallucinated test-model oracles using weighted execution consensus.

    Reads:
      - part1.json  (has gt_bool_table from execute_scripts_gt)
      - full outputs .json (written by sample.py; has all_case_input/output)

    For each generated test (index k >= num_ground_truth_test):
      1. Select the top-k generated codes by GT pass rate.
      2. If the best code's GT pass rate < min_quality, skip problem.
      3. Execute each selected code on the test input; collect raw string outputs.
      4. Compute a weighted majority vote (w_j = gt_pass_rate_j^2).
      5. If consensus_ratio >= confidence_threshold, replace all_case_output[k]
         with the execution oracle and record whether the original model oracle
         already matched (for the reasoning bonus in reward.py).
      6. If confidence is too low, mark the test as discarded so that
         execute_scripts_case can skip it.

    Writes updated full outputs .json (in-place).
    """
    full_path = "./temp_data/outputs-" + outputs_name + ".json"
    partial_path = "./temp_data/outputs-" + outputs_name + "-part1.json"

    with open(full_path, "r") as f:
        data = json.load(f)
    with open(partial_path, "r") as f:
        part1 = json.load(f)

    # ---- collect execution tasks ------------------------------------------------
    index_list = []      # problem index
    position_list = []   # (code_rank, test_k) within its problem
    code_list = []
    input_list = []
    time_limit_list = []

    # per-problem bookkeeping: top-k code indices and their weights
    problem_topk_info = {}  # i -> {"code_indices": [...], "weights": [...]}

    for i in range(len(data)):
        n_gt = data[i]["num_ground_truth_test"]
        n_all = len(data[i]["all_case_input"])
        n_generated = n_all - n_gt
        n_code = len(data[i]["generated_code"])

        # Initialise oracle correction fields
        data[i]["oracle_corrected"] = [False] * n_all
        data[i]["model_oracle_match"] = [False] * n_all
        data[i]["oracle_confidence"] = [0.0] * n_all
        if "oracle_status" not in data[i] or len(data[i]["oracle_status"]) != n_all:
            n_gt = data[i]["num_ground_truth_test"]
            data[i]["oracle_status"] = ["ground_truth"] * n_gt + ["pending"] * max(n_all - n_gt, 0)

        if n_code == 0 or n_generated == 0:
            for k_idx in range(n_gt, n_all):
                if data[i]["oracle_status"][k_idx] == "pending":
                    data[i]["oracle_status"][k_idx] = "no_code_or_case"
            continue
        if part1[i].get("gt_bool_table") is None:
            for k_idx in range(n_gt, n_all):
                if data[i]["oracle_status"][k_idx] == "pending":
                    data[i]["oracle_status"][k_idx] = "missing_gt_execution"
            continue

        gt_bool = np.array(part1[i]["gt_bool_table"], dtype=bool)  # (n_code, n_gt)
        gt_pass_rates = gt_bool.mean(axis=1)  # (n_code,)

        # Quality gate: best code must clear the floor
        if gt_pass_rates.max() < min_quality:
            for k_idx in range(n_gt, n_all):
                if data[i]["oracle_status"][k_idx] == "pending":
                    data[i]["oracle_status"][k_idx] = "quality_gate_failed"
            continue

        # Select top-k codes (may be fewer than topk if not enough codes)
        k = min(topk, n_code)
        topk_indices = np.argsort(gt_pass_rates)[::-1][:k].tolist()
        topk_weights = [float(gt_pass_rates[j] ** 2) for j in topk_indices]

        problem_topk_info[i] = {"code_indices": topk_indices, "weights": topk_weights}

        # Queue execution of each (top-k code, generated test input) pair
        for rank, j in enumerate(topk_indices):
            for k_idx in range(n_gt, n_all):
                if data[i]["oracle_status"][k_idx] == "parse_fail":
                    continue
                code_list.append(data[i]["generated_code"][j])
                input_list.append(data[i]["all_case_input"][k_idx])
                time_limit_list.append(1)
                index_list.append(i)
                position_list.append((rank, k_idx))

    if not code_list:
        cprint("[oracle-correction] nothing to correct — writing data unchanged", "yellow")
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    cprint(f"[oracle-correction] running {len(code_list)} executions "
           f"({len(problem_topk_info)} problems eligible)", "cyan")

    exe_results = run_scripts(code_list, input_list, time_limit_list, worker, num_chunks)

    # ---- collect raw outputs per (problem, test) --------------------------------
    # raw_outputs[i][k_idx] = list of (weight, output_str) for each top-k code
    raw_outputs = {}
    for idx in range(len(index_list)):
        i = index_list[idx]
        rank, k_idx = position_list[idx]
        if i not in raw_outputs:
            raw_outputs[i] = {}
        if k_idx not in raw_outputs[i]:
            raw_outputs[i][k_idx] = []
        raw_outputs[i][k_idx].append(
            (problem_topk_info[i]["weights"][rank], exe_results[idx])
        )

    # ---- apply weighted majority vote ------------------------------------------
    corrected_count = 0
    discarded_count = 0
    bonus_count = 0

    for i, test_outputs in raw_outputs.items():
        n_gt = data[i]["num_ground_truth_test"]
        for k_idx, weighted_outputs in test_outputs.items():
            # Filter out error / timeout outputs
            valid = [(w, out) for w, out in weighted_outputs
                     if not out.startswith("errorType:") and "Timeout" not in out]
            if not valid:
                # All top-k codes failed on this input — discard the test
                data[i]["oracle_status"][k_idx] = "all_topk_error"
                data[i]["all_case_output"][k_idx] = ""   # sentinel: skip in case exec
                discarded_count += 1
                continue

            total_valid_weight = sum(w for w, _ in valid)

            if total_valid_weight < min_total_weight:
                # Single weak voter or all codes have near-zero quality — unreliable
                data[i]["oracle_status"][k_idx] = "low_valid_weight"
                data[i]["all_case_output"][k_idx] = ""
                discarded_count += 1
                continue

            # Aggregate vote weight per normalised output string
            vote_weights: dict[str, float] = {}
            for w, out in valid:
                key = " ".join(out.split())   # normalise whitespace (same as test_if_eq)
                vote_weights[key] = vote_weights.get(key, 0.0) + w

            best_key = max(vote_weights, key=lambda k: vote_weights[k])
            consensus_ratio = vote_weights[best_key] / total_valid_weight
            data[i]["oracle_confidence"][k_idx] = consensus_ratio

            if consensus_ratio < confidence_threshold:
                # Low consensus — not confident enough; discard
                data[i]["oracle_status"][k_idx] = "low_consensus"
                data[i]["all_case_output"][k_idx] = ""
                discarded_count += 1
                continue

            # Accepted oracle: use the best-vote raw output (preserve original formatting)
            # Pick the first raw output whose normalised form equals best_key
            exec_oracle = next(out for _, out in valid
                               if " ".join(out.split()) == best_key)

            original_oracle = data[i]["all_case_output"][k_idx]
            model_matched = test_if_eq(original_oracle, exec_oracle)

            # Do NOT overwrite all_case_output — record model_oracle_match /
            # oracle_status instead so downstream consumers can use the
            # execution-derived oracle without losing the original expected output.
            data[i]["oracle_corrected"][k_idx] = True
            data[i]["model_oracle_match"][k_idx] = model_matched
            data[i]["oracle_status"][k_idx] = "accepted"
            corrected_count += 1
            if model_matched:
                bonus_count += 1

    cprint(f"[oracle-correction] corrected={corrected_count}, "
           f"model_matched={bonus_count}, discarded={discarded_count}", "cyan")

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"[oracle-correction] done — updated {full_path}", "cyan")


def execute_scripts_case(outputs_name, num_chunks):
    """Execute generated-case columns; merge GT results from part1, then compute stats.

    Reads the full sample output (written by sample.py at end of Phase 2) and
    the part1 file (which has GT execution results from execute_scripts_gt).
    """
    full_path = "./temp_data/outputs-" + outputs_name + ".json"
    partial_path = "./temp_data/outputs-" + outputs_name + "-part1.json"

    with open(full_path, "r") as f:
        data = json.load(f)
    with open(partial_path, "r") as f:
        part1 = json.load(f)

    index_list = []
    position_list = []
    code_list = []
    input_list = []
    output_list = []
    time_limit_list = []

    for i in range(len(data)):
        n_gt = data[i]["num_ground_truth_test"]
        n_all = len(data[i]["all_case_input"])
        n_code = len(data[i]["generated_code"])

        if n_code == 0 or n_all == 0:
            data[i]["all_case_exe_results"] = None
            data[i]["all_case_bool_table"] = None
            continue

        data[i]["all_case_exe_results"] = [["" for _ in range(n_all)] for _ in range(n_code)]
        data[i]["all_case_bool_table"] = np.full((n_code, n_all), False, dtype=bool)

        # Merge GT columns from part1 execution results
        if part1[i].get("gt_bool_table") is not None:
            gt_bool = np.array(part1[i]["gt_bool_table"], dtype=bool)
            gt_exe = part1[i]["gt_exe_results"]
            for j in range(n_code):
                for k in range(n_gt):
                    data[i]["all_case_bool_table"][j, k] = gt_bool[j, k]
                    data[i]["all_case_exe_results"][j][k] = gt_exe[j][k]

        # Queue generated-case columns for execution.
        # Skip tests with empty expected output — these are either parse failures
        # from sample.py or tests discarded by oracle correction (low-confidence
        # oracle).  They remain False in the bool table, which is the conservative
        # correct behaviour.
        for j in range(n_code):
            for k in range(n_gt, n_all):
                if data[i]["all_case_output"][k] == "":
                    continue
                code_list.append(data[i]["generated_code"][j])
                input_list.append(data[i]["all_case_input"][k])
                output_list.append(data[i]["all_case_output"][k])
                time_limit_list.append(1)
                index_list.append(i)
                position_list.append((j, k))

    exe_results = run_scripts(code_list, input_list, time_limit_list, worker, num_chunks)

    for idx in range(len(index_list)):
        i = index_list[idx]
        j, k = position_list[idx]
        data[i]["all_case_exe_results"][j][k] = exe_results[idx]
        data[i]["all_case_bool_table"][j][k] = test_if_eq(exe_results[idx], output_list[idx])

    data = prune_generated_cases_by_duplication(data)
    _compute_and_save_stats(data, outputs_name)


# read the configurations and convert them to global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--num_chunks", type=int, default=optimization_config.num_chunks)
    parser.add_argument("--num_executors", type=int,
                        default=getattr(optimization_config, "num_executors", multiprocessing.cpu_count()))
    parser.add_argument("--use_executor_pool", type=str2bool,
                        default=getattr(optimization_config, "use_executor_pool", True))
    parser.add_argument("--scale_tuple_list", type=ast.literal_eval, default=optimization_config.scale_tuple_list)
    parser.add_argument("--k_case_keep", type=int,
                        default=getattr(optimization_config, "k_case_keep", None))
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "gt", "case", "oracle_correction", "prepare_part1"],
        help=(
            "Execution mode: "
            "'all' = execute all columns sequentially (default, backward-compat); "
            "'gt'  = execute only ground-truth columns from the part1 partial file; "
            "'case'= execute only generated-case columns, merging GT results from part1; "
            "'oracle_correction' = correct hallucinated test oracles using top-k code consensus; "
            "'prepare_part1' = seed part1.json from the full outputs file (sequential pipeline helper)."
        ),
    )
    parser.add_argument("--oracle_correction_topk", type=int,
                        default=optimization_config.oracle_correction_topk)
    parser.add_argument("--oracle_correction_confidence", type=float,
                        default=optimization_config.oracle_correction_confidence)
    parser.add_argument("--oracle_correction_min_quality", type=float,
                        default=optimization_config.oracle_correction_min_quality)
    parser.add_argument("--oracle_correction_min_total_weight", type=float,
                        default=optimization_config.oracle_correction_min_total_weight)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))


# read processed data
outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset

# dispatch
if mode == "prepare_part1":
    execute_scripts_prepare_part1(outputs_name)
elif mode == "gt":
    execute_scripts_gt(outputs_name, num_chunks)
elif mode == "oracle_correction":
    execute_scripts_oracle_correction(
        outputs_name, num_chunks,
        topk=oracle_correction_topk,
        confidence_threshold=oracle_correction_confidence,
        min_quality=oracle_correction_min_quality,
        min_total_weight=oracle_correction_min_total_weight,
    )
elif mode == "case":
    execute_scripts_case(outputs_name, num_chunks)
else:
    execute_scripts(outputs_name, num_chunks)
