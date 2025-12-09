#!/usr/bin/env python3
"""Convert Online-Mind2Web tasks to WebVoyager-compatible JSONL.

This script maps each Online-Mind2Web task:
  - task_id -> id
  - website -> web
  - task_description -> ques
  - derived hostname -> web_name

The output JSONL can be used as a drop-in replacement for
data/patchedTasks.jsonl in the WebVoyager-style evaluator.

Usage:
    python convert_online_mind2web.py --input online.json --output online_as_wv.jsonl

The input file can be:
  - a JSON array of task objects, or
  - a JSONL file (one task per line).

The script will auto-detect format.
"""

import argparse
import json
import sys
from urllib.parse import urlparse


def iter_tasks(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON input must be a list of tasks")
            for obj in data:
                yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def hostname_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or "Online-Mind2Web"
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return "Online-Mind2Web"


def convert_task(raw: dict) -> dict:
    task_id = raw.get("task_id")
    website = raw.get("website")

    desc = raw.get("confirmed_task") or raw.get("task_description")

    if not (isinstance(task_id, str) and isinstance(website, str) and isinstance(desc, str)):
        raise ValueError("Missing required fields: task_id, website, confirmed_task")

    return {
        "web_name": hostname_from_url(website),
        "id": task_id,
        "ques": desc,
        "web": website,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Online-Mind2Web tasks to WebVoyager JSONL")
    parser.add_argument("--input", required=True, help="Path to Online-Mind2Web tasks (json or jsonl)")
    parser.add_argument("--output", required=True, help="Output path for WebVoyager-style jsonl")
    args = parser.parse_args()

    total = 0
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for raw in iter_tasks(args.input):
            total += 1
            try:
                task = convert_task(raw)
            except Exception as e:
                print(f"[WARN] Skip task #{total}: {e}", file=sys.stderr)
                continue
            out.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1

    print(f"Converted {written}/{total} tasks -> {args.output}")


if __name__ == "__main__":
    main()
