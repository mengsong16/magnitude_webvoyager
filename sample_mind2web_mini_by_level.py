#!/usr/bin/env python3
"""
Stratified sampling by level for Online-Mind2Web.

Default:
  - tasks input (WV-style): ./data/online_mind2web_as_wv.jsonl
  - raw input with level:  ./data/online_mind2web.json
  - output:               ./data/online_mind2web_mini_{n}.jsonl

We assume:
  - raw file items contain fields: task_id, level
  - WV-style items contain fields: id, web_name, ques, web

This script builds a mapping task_id -> level from the raw file,
then attaches level to WV tasks and samples by level.

Strategies:
  - proportional (default): keep original level proportions
  - equal: try to sample equal counts per level

Example:
  python3 sample_mind2web_mini_by_level.py --n 30 --seed 42
  python3 sample_mind2web_mini_by_level.py --n 30 --strategy equal
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_level_map(raw_path: str) -> Dict[str, str]:
    data = read_json(raw_path)

    # raw export is expected to be a list
    if not isinstance(data, list):
        raise ValueError(f"Raw file must be a JSON list: {raw_path}")

    level_map: Dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        tid = obj.get("task_id")
        lvl = obj.get("level")
        if isinstance(tid, str) and isinstance(lvl, str):
            level_map[tid] = lvl
    return level_map


def attach_levels(
    wv_tasks: List[Dict[str, Any]],
    level_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with_level = []
    missing = []
    for t in wv_tasks:
        tid = t.get("id")
        lvl = level_map.get(tid) if isinstance(tid, str) else None
        if lvl:
            t2 = dict(t)
            t2["level"] = lvl
            with_level.append(t2)
        else:
            missing.append(t)
    return with_level, missing


def proportional_counts(level_counts: Dict[str, int], n: int) -> Dict[str, int]:
    total = sum(level_counts.values())
    if total == 0:
        return {}

    # initial floor allocation
    alloc = {lvl: int(n * c / total) for lvl, c in level_counts.items()}
    # distribute remainder by largest fractional parts
    remainder = n - sum(alloc.values())

    # compute fractional parts
    fracs = []
    for lvl, c in level_counts.items():
        exact = n * c / total
        fracs.append((exact - int(exact), lvl))
    fracs.sort(reverse=True)

    i = 0
    while remainder > 0 and fracs:
        _, lvl = fracs[i % len(fracs)]
        alloc[lvl] += 1
        remainder -= 1
        i += 1

    return alloc


def equal_counts(levels: List[str], n: int) -> Dict[str, int]:
    if not levels:
        return {}
    k = len(levels)
    base = n // k
    rem = n % k
    alloc = {lvl: base for lvl in levels}
    # give extra 1 to first rem levels (stable order)
    for lvl in levels[:rem]:
        alloc[lvl] += 1
    return alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./data/online_mind2web_as_wv.jsonl",
        help="WV-style tasks JSONL (default: ./data/online_mind2web_as_wv.jsonl)",
    )
    parser.add_argument(
        "--raw_with_level",
        default="./data/online_mind2web.json",
        help="Raw Online-Mind2Web JSON with level (default: ./data/online_mind2web.json)",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of tasks to sample")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: ./data/online_mind2web_mini_{n}.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--strategy",
        choices=["proportional", "equal"],
        default="proportional",
        help="Stratified allocation strategy",
    )
    args = parser.parse_args()

    if args.n <= 0:
        print("Error: --n must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    # Load data
    wv_tasks = read_jsonl(args.input)
    if not wv_tasks:
        print(f"Error: empty input JSONL: {args.input}", file=sys.stderr)
        sys.exit(1)

    level_map = build_level_map(args.raw_with_level)

    with_level, missing = attach_levels(wv_tasks, level_map)

    if not with_level:
        print(
            "Error: could not attach level to any task. "
            "Check that raw_with_level has matching task_id values.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build buckets
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in with_level:
        buckets[t["level"]].append(t)

    levels = sorted(buckets.keys())
    level_sizes = {lvl: len(buckets[lvl]) for lvl in levels}

    if args.n > sum(level_sizes.values()):
        print(
            f"Error: requested n={args.n} but only {sum(level_sizes.values())} tasks with level available.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Decide allocation
    if args.strategy == "proportional":
        alloc = proportional_counts(level_sizes, args.n)
    else:
        alloc = equal_counts(levels, args.n)

    # If any allocation exceeds bucket size, cap and redistribute
    # Simple greedy redistribution to remaining capacity.
    remaining_n = args.n
    final_alloc = {lvl: 0 for lvl in levels}

    # First pass: cap by availability
    for lvl in levels:
        want = alloc.get(lvl, 0)
        cap = level_sizes[lvl]
        take = min(want, cap)
        final_alloc[lvl] = take
        remaining_n -= take

    if remaining_n > 0:
        # Build list of levels with remaining capacity
        capacities = {lvl: level_sizes[lvl] - final_alloc[lvl] for lvl in levels}
        # Distribute one by one
        lvl_cycle = [lvl for lvl in levels if capacities[lvl] > 0]
        i = 0
        while remaining_n > 0 and lvl_cycle:
            lvl = lvl_cycle[i % len(lvl_cycle)]
            if capacities[lvl] > 0:
                final_alloc[lvl] += 1
                capacities[lvl] -= 1
                remaining_n -= 1
            i += 1
            lvl_cycle = [l for l in levels if capacities[l] > 0]

    # Sample per bucket
    sampled: List[Dict[str, Any]] = []
    for lvl in levels:
        k = final_alloc.get(lvl, 0)
        if k <= 0:
            continue
        sampled.extend(random.sample(buckets[lvl], k))

    # Optional: shuffle final output for mixing levels
    random.shuffle(sampled)

    if args.output is None:
        args.output = f"./data/online_mind2web_mini_{args.n}.jsonl"

    write_jsonl(args.output, sampled)

    print(f"Input WV tasks: {len(wv_tasks)}")
    print(f"Tasks with level attached: {len(with_level)}")
    print(f"Tasks missing level mapping: {len(missing)}")
    print("Level bucket sizes:", level_sizes)
    print("Final allocation:", final_alloc)
    print(f"Sampled {len(sampled)} tasks -> {args.output}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
