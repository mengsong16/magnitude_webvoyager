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


def _extract_total_from_reflect(reflectj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (time_ms, action_count, input_tokens, output_tokens) summed across attempts.
    Expected reflect format (new):
      { "task_id": "...", "attempts": [ { "run": { "timeMs":..., "actionCount":..., "inputTokens":..., "outputTokens":... } }, ... ] }
    Backward compatible:
      - If no "attempts" list, treat reflectj itself as one attempt record.
    If no usable per-attempt run stats exist, returns (None, None, None, None).
    """
    attempts: List[Any] = []
    if isinstance(reflectj.get("attempts"), list):
        attempts = reflectj.get("attempts")  # type: ignore
    else:
        # old single-record format
        attempts = [reflectj]

    sum_time = 0.0
    sum_actions = 0.0
    sum_inp = 0.0
    sum_out = 0.0
    has_any = False

    for a in attempts:
        if not isinstance(a, dict):
            continue
        run = a.get("run")
        if not isinstance(run, dict):
            continue

        t = _to_float(_get_first(run, ["timeMs", "time_ms", "time"]))
        ac = _to_float(_get_first(run, ["actionCount", "action_count", "actions"]))
        inp = _to_float(_get_first(run, ["inputTokens", "totalInputTokens", "total_input_tokens"]))
        out = _to_float(_get_first(run, ["outputTokens", "totalOutputTokens", "total_output_tokens"]))

        if t is not None:
            sum_time += t
            has_any = True
        if ac is not None:
            sum_actions += ac
            has_any = True
        if inp is not None:
            sum_inp += inp
            has_any = True
        if out is not None:
            sum_out += out
            has_any = True

    if not has_any:
        return None, None, None, None

    return sum_time, sum_actions, sum_inp, sum_out


def _extract_attempts_count_from_reflect(reflectj: Dict[str, Any]) -> Optional[float]:
    """Return number of attempts recorded in <task_id>.reflect.json.
    Expected new format: {"attempts": [...] } -> len(attempts)
    Backward compatible: old single-record format -> 1
    Returns None if reflectj is not usable.
    """
    if not isinstance(reflectj, dict):
        return None
    atts = reflectj.get("attempts")
    if isinstance(atts, list):
        return float(len(atts))
    # old single-record format (has attempt/eval/suggestion at top-level)
    if any(k in reflectj for k in ["attempt", "eval", "suggestion", "run", "ts"]):
        return 1.0
    return None

@dataclass
class RunSummary:
    run_dir: str
    n_tasks: int
    success_rate: float
    first_try_successes: int
    retry_successes: int
    total_attempts: int
    total_retries: int
    retries_stats: Dict[str, float]  # retries per task (attempts-1)
    # Attempts per task (from <task_id>.reflect.json if present; otherwise assume 1)
    attempts_stats: Dict[str, float]
    # Final attempt metrics (from <task_id>.json)
    time_min_stats: Dict[str, float]
    action_stats: Dict[str, float]
    in_tok_stats: Dict[str, float]
    out_tok_stats: Dict[str, float]
    # Total metrics with retries (from <task_id>.reflect.json attempts[].run), fallback to final if missing
    time_min_total_stats: Dict[str, float]
    action_total_stats: Dict[str, float]
    in_tok_total_stats: Dict[str, float]
    out_tok_total_stats: Dict[str, float]


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
        if (not p.endswith(".eval.json")) and (not p.endswith(".reflect.json"))
    ])

    # map task_id -> run json
    task_ids: List[str] = []
    time_min_list: List[float] = []
    action_list: List[float] = []
    in_tok_list: List[float] = []
    out_tok_list: List[float] = []

    # attempts per task (from .reflect.json if present; otherwise assume 1)
    attempts_list: List[float] = []
    retries_list: List[float] = []  # retries per task = max(0, attempts-1)
    attempts_by_task: Dict[str, int] = {}

    # total metrics with retries (sum attempts from .reflect.json if available)
    time_min_total_list: List[float] = []
    action_total_list: List[float] = []
    in_tok_total_list: List[float] = []
    out_tok_total_list: List[float] = []

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


        # total metrics with retries: prefer .reflect.json attempts[].run sums; fallback to final attempt values
        t_total_ms: Optional[float] = None
        a_total: Optional[float] = None
        inp_total: Optional[float] = None
        out_total: Optional[float] = None

        attempts_n: Optional[float] = 1.0
        reflect_p = os.path.join(results_dir, f"{task_id}.reflect.json")
        if os.path.exists(reflect_p):
            rfj = _read_json(reflect_p)
            if isinstance(rfj, dict):
                t_total_ms, a_total, inp_total, out_total = _extract_total_from_reflect(rfj)
                attempts_n = _extract_attempts_count_from_reflect(rfj) or attempts_n


        if t_total_ms is None:
            t_total_ms = t_ms
        if a_total is None:
            a_total = a
        if inp_total is None:
            inp_total = inp
        if out_total is None:
            out_total = out

        if t_total_ms is not None:
            time_min_total_list.append(float(t_total_ms) / 1000.0 / 60.0)
        if a_total is not None:
            action_total_list.append(float(a_total))
        if inp_total is not None:
            in_tok_total_list.append(float(inp_total))
        if out_total is not None:
            out_tok_total_list.append(float(out_total))

        # record attempts + retries per task
        att_int = int(attempts_n) if attempts_n is not None else 1
        if att_int < 1:
            att_int = 1
        attempts_by_task[task_id] = att_int
        attempts_list.append(float(att_int))
        retries_list.append(float(max(0, att_int - 1)))
    # success rate from eval jsons (match wv.ts: evalData.result === "SUCCESS")
    eval_paths = sorted(glob.glob(os.path.join(results_dir, "*.eval.json")))
    if len(eval_paths) == 0:
        success_rate = float("nan")
        first_try_successes = 0
        retry_successes = 0
    else:
        success = 0
        first_try_successes = 0
        retry_successes = 0
        # Treat any unreadable/invalid eval as failure (still counted in denominator).
        for task_id in task_ids:
            ep = os.path.join(results_dir, f"{task_id}.eval.json")
            ej = _read_json(ep)
            is_succ = isinstance(ej, dict) and (_extract_success(ej) is True)
            if is_succ:
                success += 1
                att_int = attempts_by_task.get(task_id, 1)
                if att_int == 1:
                    first_try_successes += 1
                else:
                    retry_successes += 1
        total_for_sr = len(task_ids)
        success_rate = (float(success) / float(total_for_sr)) if total_for_sr > 0 else float("nan")
    
    # stats (if lists are empty, set NaN)
    def safe_stats(xs: List[float]) -> Dict[str, float]:
        if len(xs) == 0:
            return {"min": math.nan, "max": math.nan, "mean": math.nan, "median": math.nan, "std": math.nan}
        return _stats(xs)

    total_attempts = int(sum(attempts_by_task.values()))
    total_retries = max(0, total_attempts - len(task_ids))

    return RunSummary(
        run_dir=results_dir,
        n_tasks=len(task_ids),
        success_rate=success_rate,
        first_try_successes=int(first_try_successes) if 'first_try_successes' in locals() else 0,
        retry_successes=int(retry_successes) if 'retry_successes' in locals() else 0,
        total_attempts=int(total_attempts),
        total_retries=int(total_retries),
        retries_stats=safe_stats(retries_list),
        attempts_stats=safe_stats(attempts_list),
        time_min_stats=safe_stats(time_min_list),
        action_stats=safe_stats(action_list),
        in_tok_stats=safe_stats(in_tok_list),
        out_tok_stats=safe_stats(out_tok_list),
        time_min_total_stats=safe_stats(time_min_total_list),
        action_total_stats=safe_stats(action_total_list),
        in_tok_total_stats=safe_stats(in_tok_total_list),
        out_tok_total_stats=safe_stats(out_tok_total_list),
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
        print(f"First-try successes: {s.first_try_successes}/{s.n_tasks}")
        denom_retry = max(0, s.n_tasks - s.first_try_successes)
        print(f"Retry successes: {s.retry_successes}/{denom_retry}")
        print("----------------------- Final attempt ------------------------")
        print("Metric                |   Min   |   Max   |   Mean  |  Median")
        print("----------------------|---------|---------|---------|---------")
        print(f"Time (min)            | {s.time_min_stats['min']:.1f} | {s.time_min_stats['max']:.1f} | {s.time_min_stats['mean']:.1f} | {s.time_min_stats['median']:.1f}")
        print(f"Action count          | {s.action_stats['min']:.1f} | {s.action_stats['max']:.1f} | {s.action_stats['mean']:.1f} | {s.action_stats['median']:.1f}")
        print(f"Total input tokens    | {s.in_tok_stats['min']:.1f} | {s.in_tok_stats['max']:.1f} | {s.in_tok_stats['mean']:.1f} | {s.in_tok_stats['median']:.1f}")
        print(f"Total output tokens   | {s.out_tok_stats['min']:.1f} | {s.out_tok_stats['max']:.1f} | {s.out_tok_stats['mean']:.1f} | {s.out_tok_stats['median']:.1f}")
        print(f"Attempts             | {s.attempts_stats['min']:.1f} | {s.attempts_stats['max']:.1f} | {s.attempts_stats['mean']:.2f} | {s.attempts_stats['median']:.1f}")
        print("---------------------- Total w/ retries ----------------------")
        print("Metric                |   Min   |   Max   |   Mean  |  Median")
        print("----------------------|---------|---------|---------|---------")
        print(f"Time (min)            | {s.time_min_total_stats['min']:.1f} | {s.time_min_total_stats['max']:.1f} | {s.time_min_total_stats['mean']:.1f} | {s.time_min_total_stats['median']:.1f}")
        print(f"Action count          | {s.action_total_stats['min']:.1f} | {s.action_total_stats['max']:.1f} | {s.action_total_stats['mean']:.1f} | {s.action_total_stats['median']:.1f}")
        print(f"Total input tokens    | {s.in_tok_total_stats['min']:.1f} | {s.in_tok_total_stats['max']:.1f} | {s.in_tok_total_stats['mean']:.1f} | {s.in_tok_total_stats['median']:.1f}")
        print(f"Total output tokens   | {s.out_tok_total_stats['min']:.1f} | {s.out_tok_total_stats['max']:.1f} | {s.out_tok_total_stats['mean']:.1f} | {s.out_tok_total_stats['median']:.1f}")

    # across-runs mean ± std (using per-run MEANs, and success_rate)
    def mean_std(values: List[float]) -> Tuple[float, float]:
        vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
        if len(vals) == 0:
            return math.nan, math.nan
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    sr_m, sr_s = mean_std([s.success_rate for s in summaries])
    ft_m, ft_s = mean_std([float(s.first_try_successes) for s in summaries])
    rs_m, rs_s = mean_std([float(s.retry_successes) for s in summaries])
    retries_m, retries_s = mean_std([float(s.total_retries) for s in summaries])
    t_m, t_s = mean_std([s.time_min_stats["mean"] for s in summaries])
    a_m, a_s = mean_std([s.action_stats["mean"] for s in summaries])
    in_m, in_s = mean_std([s.in_tok_stats["mean"] for s in summaries])
    out_m, out_s = mean_std([s.out_tok_stats["mean"] for s in summaries])

    # Total w/ retries: across-runs mean ± std of per-run MEANs
    t_tot_m, t_tot_s = mean_std([s.time_min_total_stats["mean"] for s in summaries])
    a_tot_m, a_tot_s = mean_std([s.action_total_stats["mean"] for s in summaries])
    in_tot_m, in_tot_s = mean_std([s.in_tok_total_stats["mean"] for s in summaries])
    out_tot_m, out_tot_s = mean_std([s.out_tok_total_stats["mean"] for s in summaries])

    # NEW: across-runs mean ± std of per-run MEDIANs
    t_med_m, t_med_s = mean_std([s.time_min_stats["median"] for s in summaries])
    a_med_m, a_med_s = mean_std([s.action_stats["median"] for s in summaries])
    in_med_m, in_med_s = mean_std([s.in_tok_stats["median"] for s in summaries])
    out_med_m, out_med_s = mean_std([s.out_tok_stats["median"] for s in summaries])

    # Total w/ retries: across-runs mean ± std of per-run MEDIANs
    t_tot_med_m, t_tot_med_s = mean_std([s.time_min_total_stats["median"] for s in summaries])
    a_tot_med_m, a_tot_med_s = mean_std([s.action_total_stats["median"] for s in summaries])
    in_tot_med_m, in_tot_med_s = mean_std([s.in_tok_total_stats["median"] for s in summaries])
    out_tot_med_m, out_tot_med_s = mean_std([s.out_tok_total_stats["median"] for s in summaries])

    print("\n===== Across-runs (mean ± std over runs) =====")
    print("----------------------------------------------")
    print(f"Success rate        : {sr_m:.4f} ± {sr_s:.4f}")
    print(f"Retry successes     : {rs_m:.3f} ± {rs_s:.3f}")
    print(f"First-try successes : {ft_m:.3f} ± {ft_s:.3f}")
    print("----------------------------------------------")
    print(f'Retries            : {retries_m:.3f} ± {retries_s:.3f}\n')
    
    print("============== Final attempt =================")
    print(f"Time mean (min)     : {t_m:.3f} ± {t_s:.3f}")
    print(f"Action mean         : {a_m:.3f} ± {a_s:.3f}")
    print(f"Input tokens mean   : {in_m:.3f} ± {in_s:.3f}")
    print(f"Output tokens mean  : {out_m:.3f} ± {out_s:.3f}")
    print("----------------------------------------------")
    print(f"Time median (min)   : {t_med_m:.3f} ± {t_med_s:.3f}")
    print(f"Action median       : {a_med_m:.3f} ± {a_med_s:.3f}")
    print(f"Input tokens median : {in_med_m:.3f} ± {in_med_s:.3f}")
    print(f"Output tokens median: {out_med_m:.3f} ± {out_med_s:.3f}\n")

    print("============= Total w/ retries ===============")
    print(f"Time mean (min)     : {t_tot_m:.3f} ± {t_tot_s:.3f}")
    print(f"Action mean         : {a_tot_m:.3f} ± {a_tot_s:.3f}")
    print(f"Input tokens mean   : {in_tot_m:.3f} ± {in_tot_s:.3f}")
    print(f"Output tokens mean  : {out_tot_m:.3f} ± {out_tot_s:.3f}")
    print("----------------------------------------------")
    print(f"Time median (min)   : {t_tot_med_m:.3f} ± {t_tot_med_s:.3f}")
    print(f"Action median       : {a_tot_med_m:.3f} ± {a_tot_med_s:.3f}")
    print(f"Input tokens median : {in_tot_med_m:.3f} ± {in_tot_med_s:.3f}")
    print(f"Output tokens median: {out_tot_med_m:.3f} ± {out_tot_med_s:.3f}")

    return summaries


# -------- example usage --------
# summarize_runs(["run1", "run2", "run3"], base_results_dir="results")
# or:
# summarize_runs(["/abs/path/to/results/run1", "/abs/path/to/results/run2"])