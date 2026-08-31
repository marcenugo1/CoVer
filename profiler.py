"""
CoVer training pipeline profiler.
Captures wall-clock time, GPU utilization/memory, CPU/RAM usage for each phase
and writes structured JSON lines + a human-readable summary to disk.

Usage in run.py:
    from profiler import StepProfiler
    prof = StepProfiler(step=i)
    with prof.phase("sample"):
        sample(model)
    prof.save()
    print(prof.summary())
"""

import os
import time
import json
import threading
import subprocess
from datetime import datetime


# ---------------------------------------------------------------------------
# Low-level stat collectors
# ---------------------------------------------------------------------------

def get_gpu_stats():
    """Snapshot GPU stats via nvidia-smi. Returns a list of dicts, one per GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                gpus.append(
                    {
                        "gpu": int(parts[0]),
                        "util_pct": float(parts[1]),
                        "mem_used_mb": float(parts[2]),
                        "mem_total_mb": float(parts[3]),
                        "temp_c": float(parts[4]),
                        "power_w": float(parts[5]) if parts[5] not in ("[N/A]", "N/A") else None,
                    }
                )
            except (ValueError, IndexError):
                pass
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_cpu_stats():
    """Snapshot CPU/RAM stats via psutil (if available)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "cpu_pct": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_pct": vm.percent,
        }
    except ImportError:
        # psutil not installed — fall back to /proc
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem = {}
            for l in lines:
                k, v = l.split(":", 1)
                mem[k.strip()] = int(v.split()[0])  # kB
            total_gb = mem.get("MemTotal", 0) / 1e6
            avail_gb = mem.get("MemAvailable", 0) / 1e6
            used_gb = total_gb - avail_gb
            return {
                "ram_used_gb": round(used_gb, 2),
                "ram_total_gb": round(total_gb, 2),
                "ram_pct": round(100 * used_gb / total_gb, 1) if total_gb else 0,
            }
        except Exception:
            return {}


def _fmt_gpu(gpus):
    """One-line GPU summary string."""
    parts = []
    for g in gpus:
        if "error" in g:
            parts.append(f"GPU?:error")
        else:
            parts.append(
                f"GPU{g['gpu']}: {g['util_pct']:.0f}% util "
                f"{g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MB"
            )
    return " | ".join(parts) if parts else "no GPUs"


def _fmt_cpu(cpu):
    if not cpu:
        return "no data"
    parts = []
    if "cpu_pct" in cpu:
        parts.append(f"CPU {cpu['cpu_pct']:.0f}%")
    if "ram_used_gb" in cpu:
        parts.append(f"RAM {cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.1f}GB ({cpu['ram_pct']:.0f}%)")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Background monitor thread
# ---------------------------------------------------------------------------

class PhaseMonitor:
    """Samples GPU + CPU stats in a background thread while a phase runs."""

    def __init__(self, interval_s: float = 15.0):
        self.interval_s = interval_s
        self.samples: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="PhaseMonitor")
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=20)
        return self.samples

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.samples.append(
                {
                    "t": time.time(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "gpu": get_gpu_stats(),
                    "cpu": get_cpu_stats(),
                }
            )


# ---------------------------------------------------------------------------
# Phase context manager
# ---------------------------------------------------------------------------

class PhaseContext:
    def __init__(self, profiler: "StepProfiler", name: str, monitor_interval_s: float):
        self.profiler = profiler
        self.name = name
        self._monitor = PhaseMonitor(interval_s=monitor_interval_s)
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.time()
        self.profiler._log_event(f"[{self.name}] START")
        self._monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        samples = self._monitor.stop()
        duration = time.time() - self._t0
        ok = exc_type is None

        gpu_after = get_gpu_stats()
        cpu_after = get_cpu_stats()

        phase_data = {
            "phase": self.name,
            "duration_s": round(duration, 2),
            "ok": ok,
            "gpu_after": gpu_after,
            "cpu_after": cpu_after,
            "monitor_samples": samples,
        }
        self.profiler.data["phases"][self.name] = phase_data

        status = "OK" if ok else f"ERROR({exc_type.__name__ if exc_type else '?'})"
        msg = (
            f"[{self.name}] END  {status}  duration={duration:.1f}s"
            f"  gpu={_fmt_gpu(gpu_after)}"
            f"  {_fmt_cpu(cpu_after)}"
        )
        if samples:
            all_utils = [
                g["util_pct"]
                for s in samples
                for g in s["gpu"]
                if isinstance(g.get("util_pct"), (int, float))
            ]
            if all_utils:
                avg_u = sum(all_utils) / len(all_utils)
                max_u = max(all_utils)
                msg += f"  bg_gpu_util avg={avg_u:.0f}% max={max_u:.0f}%"

        self.profiler._log_event(msg)
        return False  # don't suppress exceptions


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class StepProfiler:
    """
    One instance per training step. Collects timing and resource data across
    all phases (sample, execute, reward, train) and writes results to disk.

    Log files (relative to the directory where run.py is launched from):
      optimization/results/profiling_log.jsonl   — one JSON object per step
      optimization/results/profiling_events.log  — human-readable event stream
    """

    _JSONL_PATH = "optimization/results/profiling_log.jsonl"
    _EVENT_PATH = "optimization/results/profiling_events.log"

    def __init__(self, step: int, monitor_interval_s: float = 15.0):
        self.step = step
        self.monitor_interval_s = monitor_interval_s
        self.data: dict = {
            "step": step,
            "start_wall": datetime.now().isoformat(timespec="seconds"),
            "start_ts": time.time(),
            "phases": {},
        }
        os.makedirs("optimization/results", exist_ok=True)
        self._log_event(
            f"\n{'='*72}\n"
            f"STEP {step} START  {self.data['start_wall']}\n"
            f"  gpu_init: {_fmt_gpu(get_gpu_stats())}\n"
            f"  cpu_init: {_fmt_cpu(get_cpu_stats())}\n"
            f"{'='*72}"
        )

    def phase(self, name: str) -> PhaseContext:
        return PhaseContext(self, name, self.monitor_interval_s)

    def save(self):
        """Append this step's data to the JSONL log file."""
        self.data["end_wall"] = datetime.now().isoformat(timespec="seconds")
        self.data["total_duration_s"] = round(time.time() - self.data["start_ts"], 2)

        # Summarise background samples to keep the JSON compact
        for name, phase in self.data["phases"].items():
            samples = phase.get("monitor_samples", [])
            if samples:
                phase["monitor_summary"] = _summarise_samples(samples)
            # Drop raw samples to keep file size reasonable
            phase.pop("monitor_samples", None)

        with open(self._JSONL_PATH, "a") as f:
            f.write(json.dumps(self.data) + "\n")

        summary = self.summary()
        self._log_event(summary)
        return summary

    def summary(self) -> str:
        lines = [
            f"\n{'='*72}",
            f"STEP {self.step} PHASE BREAKDOWN",
            f"{'='*72}",
        ]
        total = 0.0
        phase_names = set(self.data["phases"].keys())
        ordered_names = []
        for name in self.data["phases"].keys():
            if "/" not in name:
                ordered_names.append(name)
                prefix = f"{name}/"
                ordered_names.extend(
                    child for child in self.data["phases"].keys() if child.startswith(prefix)
                )
        ordered_names.extend(
            name for name in self.data["phases"].keys() if name not in set(ordered_names)
        )

        for name in ordered_names:
            phase = self.data["phases"][name]
            d = phase.get("duration_s", 0.0)
            # Nested phases use slash-separated names, e.g. collect/sample.
            # Keep them visible, but don't double-count them when the parent
            # phase is also present.
            is_nested = "/" in name and name.split("/", 1)[0] in phase_names
            if not is_nested:
                total += d
            ok_str = "OK" if phase.get("ok") else "FAILED"
            lines.append(f"  {name:<24s} {d:>8.1f}s  [{ok_str}]")
            # GPU after
            for g in phase.get("gpu_after", []):
                if "error" not in g:
                    lines.append(
                        f"    GPU{g['gpu']:d}: util={g['util_pct']:.0f}%  "
                        f"mem={g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MB  "
                        f"temp={g['temp_c']:.0f}°C"
                        + (f"  power={g['power_w']:.0f}W" if g.get("power_w") else "")
                    )
            # CPU after
            cpu = phase.get("cpu_after", {})
            if cpu and "ram_used_gb" in cpu:
                lines.append(f"    CPU: {_fmt_cpu(cpu)}")
            # Background monitoring summary
            ms = phase.get("monitor_summary")
            if ms:
                for gi, gs in enumerate(ms.get("gpu_summaries", [])):
                    lines.append(
                        f"    GPU{gi} during: util avg={gs['util_avg']:.0f}% max={gs['util_max']:.0f}%  "
                        f"mem avg={gs['mem_avg']:.0f}MB max={gs['mem_max']:.0f}MB"
                    )

        lines.append(f"  {'TOTAL':<24s} {total:>8.1f}s")
        lines.append(f"{'='*72}\n")
        return "\n".join(lines)

    def _log_event(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stamped = f"[{ts}] {msg}"
        print(stamped, flush=True)
        with open(self._EVENT_PATH, "a") as f:
            f.write(stamped + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_samples(samples: list) -> dict:
    """Summarise a list of background monitor samples into per-GPU stats."""
    n_gpus = max((len(s.get("gpu", [])) for s in samples), default=0)
    gpu_summaries = []
    for gi in range(n_gpus):
        utils = [
            s["gpu"][gi]["util_pct"]
            for s in samples
            if gi < len(s.get("gpu", [])) and "util_pct" in s["gpu"][gi]
        ]
        mems = [
            s["gpu"][gi]["mem_used_mb"]
            for s in samples
            if gi < len(s.get("gpu", [])) and "mem_used_mb" in s["gpu"][gi]
        ]
        gpu_summaries.append(
            {
                "util_avg": round(sum(utils) / len(utils), 1) if utils else 0,
                "util_max": round(max(utils), 1) if utils else 0,
                "mem_avg": round(sum(mems) / len(mems), 1) if mems else 0,
                "mem_max": round(max(mems), 1) if mems else 0,
                "n_samples": len(utils),
            }
        )

    # CPU summary
    cpus = [s.get("cpu", {}) for s in samples]
    cpu_pcts = [c["cpu_pct"] for c in cpus if "cpu_pct" in c]
    ram_pcts = [c["ram_pct"] for c in cpus if "ram_pct" in c]

    return {
        "n_samples": len(samples),
        "gpu_summaries": gpu_summaries,
        "cpu_avg_pct": round(sum(cpu_pcts) / len(cpu_pcts), 1) if cpu_pcts else None,
        "ram_avg_pct": round(sum(ram_pcts) / len(ram_pcts), 1) if ram_pcts else None,
    }
