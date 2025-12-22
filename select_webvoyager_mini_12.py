#!/usr/bin/env python3
"""
Select a 12-task subset from ./data/webvoyager_mini_30.jsonl and write to
./data/webvoyager_mini_12.jsonl by default.

Selection policy (deterministic):
- Pick exactly 1 task from each chosen web_name.
- Within a web_name, choose the task with the lexicographically smallest `id`.

Chosen web_name set (12 total):
  Sensitive/anti-bot: GitHub, Cambridge Dictionary, Google Search
  Heavy pages: Google Map, Huggingface, ArXiv, Booking
  Light pages: Wolfram Alpha, BBC News, Allrecipes
  Extra (medium): Amazon, Coursera
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

CHOSEN_WEBNAMES: List[str] = [
    # Sensitive / anti-bot
    "GitHub",
    "Cambridge Dictionary",
    "Google Search",
    # Heavy pages
    "Google Map",
    "Huggingface",
    "ArXiv",
    "Booking",
    # Light pages
    "Wolfram Alpha",
    "BBC News",
    "Allrecipes",
    # Extra (medium)
    "Amazon",
    "Coursera",
]

DEFAULT_INPUT = Path("data/webvoyager_mini_30.jsonl")
DEFAULT_OUTPUT = Path("data/webvoyager_mini_12.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON decode error in {path} line {i}: {e}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_subset(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_web: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        wn = t.get("web_name")
        if isinstance(wn, str):
            by_web.setdefault(wn, []).append(t)

    missing = [wn for wn in CHOSEN_WEBNAMES if wn not in by_web]
    if missing:
        avail = sorted(by_web.keys())
        raise SystemExit(
            "Missing required web_name(s): "
            + ", ".join(missing)
            + "\nAvailable web_name(s): "
            + ", ".join(avail)
        )

    chosen: List[Dict[str, Any]] = []
    for wn in CHOSEN_WEBNAMES:
        candidates = [c for c in by_web[wn] if isinstance(c.get("id"), str)]
        candidates.sort(key=lambda x: x["id"])
        if not candidates:
            raise SystemExit(f"No candidates with string `id` for web_name={wn}")
        chosen.append(candidates[0])

    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL (default: data/webvoyager_mini_30.jsonl)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL (default: data/webvoyager_mini_12.jsonl)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    tasks = load_jsonl(inp)
    chosen = select_subset(tasks)

    print("Selected 12-task subset:")
    for t in chosen:
        print(f"  {t.get('web_name'):<22}  {t.get('id')}")
    write_jsonl(out, chosen)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
