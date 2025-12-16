from __future__ import annotations
import json
import math
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------- helpers ----------

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return None


def _extract_time_ms(run: Dict[str, Any]) -> Optional[float]:
    # common candidates in your repo(s)
    v = _get_first(run, ["timeMs", "time_ms", "time", "run_time_ms", "runTimeMs"])
    t = _to_float(v)
    if t is None:
        return None
    # Heuristic: if someone stored seconds, it will be small (< 1e6) for minutes-scale runs.
    # Your wv.ts stats used ms, so assume ms unless clearly seconds.
    if t < 60_000 and t > 0:  # < 1 min in ms OR seconds-scale
        # ambiguous; keep as-is (most repos store ms, even short tasks)
        return t
    return t


def _extract_action_count(run: Dict[str, Any]) -> Optional[float]:
    # direct
    v = _get_first(run, ["actionCount", "action_count", "actions", "numActions"])
    a = _to_float(v)
    if a is not None:
        return a

    # sometimes stored in signals
    sig = run.get("signals") or run.get("sig") or run.get("runSignals")
    if isinstance(sig, dict):
        v = _get_first(sig, ["actionCount", "action_count", "actions"])
        a = _to_float(v)
        if a is not None:
            return a

    # last resort: derive from observations if present (very rough)
    obs = run.get("observations")
    if isinstance(obs, list):
        # count entries that look like actions
        cnt = 0
        for o in obs:
            if isinstance(o, dict) and ("action" in o or "tool" in o or "type" in o):
                cnt += 1
        if cnt > 0:
            return float(cnt)

    return None


def _extract_tokens(run: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    in_v = _get_first(run, ["totalInputTokens", "total_input_tokens", "inputTokens", "inTokens"])
    out_v = _get_first(run, ["totalOutputTokens", "total_output_tokens", "outputTokens", "outTokens"])
    inp = _to_float(in_v)
    out = _to_float(out_v)

    # sometimes nested
    if inp is None or out is None:
        usage = run.get("usage") or run.get("tokenUsage") or run.get("tokens")
        if isinstance(usage, dict):
            if inp is None:
                inp = _to_float(_get_first(usage, ["input", "in", "prompt_tokens", "input_tokens"]))
            if out is None:
                out = _to_float(_get_first(usage, ["output", "out", "completion_tokens", "output_tokens"]))

    return inp, out


def _extract_success(evalj: Dict[str, Any]) -> Optional[bool]:
    """Match wv.ts getTaskStatus(): isSuccess = evalData.result === "SUCCESS".
    - If evalj has a string field "result", return True iff it's exactly SUCCESS (case-insensitive).
    - Otherwise return None (caller decides how to count missing/invalid evals).
    """
    r = evalj.get("result")
    if isinstance(r, str):
        return r.strip().upper() == "SUCCESS"
    return None


@dataclass
class RunSummary:
    run_dir: str
    n_tasks: int
    success_rate: float
    time_min_stats: Dict[str, float]
    action_stats: Dict[str, float]
    in_tok_stats: Dict[str, float]
    out_tok_stats: Dict[str, float]


def _stats(arr: List[float]) -> Dict[str, float]:
    a = np.array(arr, dtype=float)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if len(a) >= 2 else 0.0,
    }


def summarize_one_run(results_dir: str) -> RunSummary:
    """
    results_dir: a single run folder, e.g. "results/online_mind2web_mini_30"
    Expected:
      - per-task run json: <task_id>.json
      - per-task eval json: <task_id>.eval.json   (if missing, success_rate will be NaN)
    """
    # run jsons: exclude *.eval.json
    run_json_paths = sorted([
        p for p in glob.glob(os.path.join(results_dir, "*.json"))
        if not p.endswith(".eval.json")
    ])

    # map task_id -> run json
    task_ids: List[str] = []
    time_min_list: List[float] = []
    action_list: List[float] = []
    in_tok_list: List[float] = []
    out_tok_list: List[float] = []

    for p in run_json_paths:
        task_id = os.path.splitext(os.path.basename(p))[0]
        runj = _read_json(p)
        if not isinstance(runj, dict):
            continue

        t_ms = _extract_time_ms(runj)
        a = _extract_action_count(runj)
        inp, out = _extract_tokens(runj)

        # keep task if at least has time+actions (adjust if you want)
        if t_ms is None and a is None and inp is None and out is None:
            continue

        task_ids.append(task_id)

        if t_ms is not None:
            time_min_list.append(float(t_ms) / 1000.0 / 60.0)
        if a is not None:
            action_list.append(float(a))
        if inp is not None:
            in_tok_list.append(float(inp))
        if out is not None:
            out_tok_list.append(float(out))

    # success rate from eval jsons (match wv.ts: evalData.result === "SUCCESS")
    eval_paths = sorted(glob.glob(os.path.join(results_dir, "*.eval.json")))
    total_for_sr = len(eval_paths) if len(eval_paths) > 0 else len(task_ids)

    success = 0
    # Treat any unreadable/invalid eval as failure (still counted in denominator).
    for ep in eval_paths:
        ej = _read_json(ep)
        if isinstance(ej, dict) and _extract_success(ej) is True:
            success += 1

    success_rate = (float(success) / float(total_for_sr)) if total_for_sr > 0 else float("nan")
    
    # stats (if lists are empty, set NaN)
    def safe_stats(xs: List[float]) -> Dict[str, float]:
        if len(xs) == 0:
            return {"min": math.nan, "max": math.nan, "mean": math.nan, "median": math.nan, "std": math.nan}
        return _stats(xs)

    return RunSummary(
        run_dir=results_dir,
        n_tasks=len(task_ids),
        success_rate=success_rate,
        time_min_stats=safe_stats(time_min_list),
        action_stats=safe_stats(action_list),
        in_tok_stats=safe_stats(in_tok_list),
        out_tok_stats=safe_stats(out_tok_list),
    )


def summarize_runs(run_dirs: List[str], base_results_dir: Optional[str] = None) -> List[RunSummary]:
    """
    run_dirs: list of folder names OR absolute paths.
      - If base_results_dir is provided, each entry in run_dirs will be joined with it.
        e.g. base_results_dir="results", run_dirs=["runA","runB"] -> results/runA, results/runB
    Prints per-run tables and an across-runs mean±std summary.
    """
    summaries: List[RunSummary] = []
    for rd in run_dirs:
        path = os.path.join(base_results_dir, rd) if base_results_dir else rd
        summaries.append(summarize_one_run(path))

    # print per-run quick summary (mean/median + success)
    for s in summaries:
        print(f"\n=== Run: {s.run_dir} ===")
        print(f"Tasks: {s.n_tasks}")
        sr = s.success_rate
        print(f"Overall success rate: {sr:.4f}" if not math.isnan(sr) else "Overall success rate: NaN (no eval json parsed)")
        print("Metric                |   Min   |   Max   |   Mean  |  Median")
        print("----------------------|---------|---------|---------|---------")
        print(f"Time (min)            | {s.time_min_stats['min']:.1f} | {s.time_min_stats['max']:.1f} | {s.time_min_stats['mean']:.1f} | {s.time_min_stats['median']:.1f}")
        print(f"Action count          | {s.action_stats['min']:.1f} | {s.action_stats['max']:.1f} | {s.action_stats['mean']:.1f} | {s.action_stats['median']:.1f}")
        print(f"Total input tokens    | {s.in_tok_stats['min']:.1f} | {s.in_tok_stats['max']:.1f} | {s.in_tok_stats['mean']:.1f} | {s.in_tok_stats['median']:.1f}")
        print(f"Total output tokens   | {s.out_tok_stats['min']:.1f} | {s.out_tok_stats['max']:.1f} | {s.out_tok_stats['mean']:.1f} | {s.out_tok_stats['median']:.1f}")

    # across-runs mean ± std (using per-run MEANs, and success_rate)
    def mean_std(values: List[float]) -> Tuple[float, float]:
        vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
        if len(vals) == 0:
            return math.nan, math.nan
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    sr_m, sr_s = mean_std([s.success_rate for s in summaries])
    t_m, t_s = mean_std([s.time_min_stats["mean"] for s in summaries])
    a_m, a_s = mean_std([s.action_stats["mean"] for s in summaries])
    in_m, in_s = mean_std([s.in_tok_stats["mean"] for s in summaries])
    out_m, out_s = mean_std([s.out_tok_stats["mean"] for s in summaries])

    # NEW: across-runs mean ± std of per-run MEDIANs
    t_med_m, t_med_s = mean_std([s.time_min_stats["median"] for s in summaries])
    a_med_m, a_med_s = mean_std([s.action_stats["median"] for s in summaries])
    in_med_m, in_med_s = mean_std([s.in_tok_stats["median"] for s in summaries])
    out_med_m, out_med_s = mean_std([s.out_tok_stats["median"] for s in summaries])

    print("\n===== Across-runs (mean ± std over runs) =====")
    print("----------------------------------------------")
    print(f"Success rate        : {sr_m:.4f} ± {sr_s:.4f}")
    print("----------------------------------------------")

    print(f"Time mean (min)     : {t_m:.3f} ± {t_s:.3f}")
    print(f"Action mean         : {a_m:.3f} ± {a_s:.3f}")
    print(f"Input tokens mean   : {in_m:.3f} ± {in_s:.3f}")
    print(f"Output tokens mean  : {out_m:.3f} ± {out_s:.3f}")

    print("----------------------------------------------")

    print(f"Time median (min)   : {t_med_m:.3f} ± {t_med_s:.3f}")
    print(f"Action median       : {a_med_m:.3f} ± {a_med_s:.3f}")
    print(f"Input tokens median : {in_med_m:.3f} ± {in_med_s:.3f}")
    print(f"Output tokens median: {out_med_m:.3f} ± {out_med_s:.3f}")

    return summaries


# -------- example usage --------
# summarize_runs(["run1", "run2", "run3"], base_results_dir="results")
# or:
# summarize_runs(["/abs/path/to/results/run1", "/abs/path/to/results/run2"])
