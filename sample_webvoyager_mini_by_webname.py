#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sample_webvoyager_mini_by_webname.py

通用工具：
- 从 WebVoyager 的全量 JSONL（如 data/patchedTasks.jsonl）中，
  按 web_name 的比例抽样指定数量的任务
- 输出为一个新的 JSONL 文件（如 data/webvoyager_mini_30.jsonl）

可以：
1）当作脚本直接用命令行调用
2）在其他 Python 文件中 import 后调用
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Any, Sequence


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件中读取所有任务，每行一个 JSON。"""
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def compute_quota_by_category(counts: Dict[str, int], total_samples: int) -> Dict[str, int]:
    """
    按照 counts 中每个类别的数量比例，分配 total_samples 个名额。

    使用“比例 + 最大余数法”：
      ideal[c] = counts[c] / sum(counts) * total_samples
      先取 floor，再按小数部分从大到小补齐剩余名额。
    """
    total = sum(counts.values())
    if total_samples > total:
        raise ValueError(
            f"total_samples={total_samples} is larger than total number of tasks={total}"
        )

    quota: Dict[str, int] = {}
    remainders = []

    for cat, cnt in counts.items():
        ideal = cnt / total * total_samples
        base = math.floor(ideal)
        quota[cat] = base
        remainders.append((ideal - base, cat))

    # 剩余名额按小数部分从大到小分配（保证可复现，按类别名二次排序）
    used = sum(quota.values())
    remaining = total_samples - used
    remainders.sort(key=lambda x: (-x[0], x[1]))

    for i in range(remaining):
        _, cat = remainders[i]
        quota[cat] += 1

    # 保险：每个类别分配数不能超过自身数量
    for cat, q in list(quota.items()):
        if q > counts[cat]:
            quota[cat] = counts[cat]

    return quota


def sample_by_webname(
    tasks: Sequence[Dict[str, Any]],
    n_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    根据 web_name 的比例，从 tasks 中抽样 n_samples 个任务。
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    random.seed(seed)

    # 按 web_name 分组
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        cat = t.get("web_name")
        if cat is None:
            raise ValueError(f"Task missing 'web_name': {t}")
        by_cat[cat].append(t)

    counts = Counter({cat: len(v) for cat, v in by_cat.items()})
    quota = compute_quota_by_category(counts, n_samples)

    sampled: List[Dict[str, Any]] = []
    for cat, q in quota.items():
        if q <= 0:
            continue
        cat_tasks = by_cat[cat]
        if q > len(cat_tasks):
            raise ValueError(
                f"Quota for category {cat} is {q}, "
                f"but only {len(cat_tasks)} tasks available."
            )
        sampled.extend(random.sample(cat_tasks, q))

    if len(sampled) != n_samples:
        raise RuntimeError(
            f"Sampled {len(sampled)} tasks but expected {n_samples}. "
            f"Please check the quota logic."
        )

    return sampled


def write_jsonl(tasks: Sequence[Dict[str, Any]], path: str) -> None:
    """把任务列表写成 JSONL 文件。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def sample_webvoyager_mini_by_webname(
    input_path: str,
    output_path: str,
    n_samples: int,
    seed: int = 42,
) -> None:
    """
    通用函数：从 input_path 抽样 n_samples 个任务并写到 output_path。

    参数：
      input_path  : 全量 JSONL 路径（如 data/patchedTasks.jsonl）
      output_path : 输出 JSONL 路径（如 data/webvoyager_mini_30.jsonl）
      n_samples   : 需要抽样的任务个数
      seed        : 随机种子（保证复现）
    """
    tasks = load_tasks(input_path)
    sampled = sample_by_webname(tasks, n_samples, seed=seed)
    write_jsonl(sampled, output_path)

    # 打印一个简单的分布对比，方便肉眼确认
    counts_all = Counter(t["web_name"] for t in tasks)
    counts_sampled = Counter(t["web_name"] for t in sampled)
    print(f"Total tasks: {len(tasks)}")
    print(f"Sampled tasks: {len(sampled)}")
    print("Original distribution (web_name -> count):")
    for cat, cnt in sorted(counts_all.items()):
        print(f"  {cat:20s}: {cnt}")
    print("Sampled distribution (web_name -> count):")
    for cat, cnt in sorted(counts_sampled.items()):
        print(f"  {cat:20s}: {cnt}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample WebVoyager mini JSONL by web_name proportion."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/patchedTasks.jsonl",
        help="Path to full WebVoyager JSONL (e.g., data/patchedTasks.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/webvoyager_mini_30.jsonl",
        help="Output JSONL path (e.g., data/webvoyager_mini_30.jsonl)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of tasks to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    sample_webvoyager_mini_by_webname(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
