"""
Generation history dashboard.

Reads generation_history.jsonl and prints summary statistics.

Usage:
  python gen_dashboard.py
  python gen_dashboard.py --top 20
  python gen_dashboard.py --since 2026-03-01
"""
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "generation_history.jsonl")


def load_history(path: str, since: str = None) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if since and entry.get("timestamp", "") < since:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def print_dashboard(entries: list, top_n: int = 15):
    if not entries:
        print("No generation history found.")
        return

    total = len(entries)
    total_images = sum(len(e.get("files", [])) for e in entries)

    # date range
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    first = timestamps[0][:10] if timestamps else "?"
    last = timestamps[-1][:10] if timestamps else "?"

    # lora usage
    lora_counter = Counter()
    for e in entries:
        for lora in e.get("loras", []):
            lora_counter[lora] += 1

    # daily volume
    daily = Counter()
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            daily[ts[:10]] += 1

    # per-day timing (avg generations per active day)
    active_days = len(daily)

    # generation speed (time between consecutive entries)
    durations = []
    for i in range(1, len(entries)):
        try:
            t1 = datetime.fromisoformat(entries[i-1]["timestamp"])
            t2 = datetime.fromisoformat(entries[i]["timestamp"])
            diff = (t2 - t1).total_seconds()
            if 5 < diff < 600:  # filter out pauses
                durations.append(diff)
        except (KeyError, ValueError):
            continue

    avg_duration = sum(durations) / len(durations) if durations else 0

    # batch sizes
    batch_sizes = Counter()
    for e in entries:
        n_files = len(e.get("files", []))
        batch_sizes[n_files] += 1

    # --- print ---
    print("=" * 60)
    print(f"  GENERATION HISTORY DASHBOARD")
    print("=" * 60)
    print()
    print(f"  Period:          {first} to {last}")
    print(f"  Total requests:  {total}")
    print(f"  Total images:    {total_images}")
    print(f"  Active days:     {active_days}")
    print(f"  Avg per day:     {total / active_days:.1f} gens/day" if active_days else "")
    print(f"  Avg gen time:    {avg_duration:.1f}s" if avg_duration else "  Avg gen time:    N/A")
    print(f"  Unique LoRAs:    {len(lora_counter)}")
    print()

    # top loras
    print(f"  TOP {top_n} LORAS")
    print(f"  {'LoRA':<45s} {'Uses':>6s}")
    print(f"  {'-'*45} {'-'*6}")
    for lora, count in lora_counter.most_common(top_n):
        bar = "#" * min(count, 30)
        print(f"  {lora:<45s} {count:>5d}  {bar}")
    print()

    # daily activity
    print(f"  DAILY ACTIVITY (last 14 days)")
    print(f"  {'Date':<12s} {'Gens':>6s}")
    print(f"  {'-'*12} {'-'*6}")
    sorted_days = sorted(daily.keys(), reverse=True)[:14]
    for day in sorted_days:
        count = daily[day]
        bar = "#" * min(count, 40)
        print(f"  {day:<12s} {count:>5d}  {bar}")
    print()

    # batch size distribution
    print(f"  BATCH SIZE DISTRIBUTION")
    print(f"  {'Images/gen':<12s} {'Count':>6s}")
    print(f"  {'-'*12} {'-'*6}")
    for size, count in sorted(batch_sizes.items()):
        print(f"  {size:<12d} {count:>5d}")
    print()

    # busiest hour
    hourly = Counter()
    for e in entries:
        ts = e.get("timestamp", "")
        if len(ts) >= 13:
            hourly[ts[11:13]] += 1
    if hourly:
        print(f"  HOURLY DISTRIBUTION")
        for hour in sorted(hourly.keys()):
            count = hourly[hour]
            bar = "#" * min(count // 2, 30)
            print(f"  {hour}:00  {count:>4d}  {bar}")
        print()

    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Generation history dashboard")
    p.add_argument("--top", type=int, default=15, help="Top N loras to show")
    p.add_argument("--since", default=None, help="Only show entries since date (YYYY-MM-DD)")
    p.add_argument("--file", default=HISTORY_FILE, help="Path to history file")
    args = p.parse_args()

    if not os.path.isfile(args.file):
        print(f"History file not found: {args.file}")
        sys.exit(1)

    entries = load_history(args.file, since=args.since)
    print_dashboard(entries, top_n=args.top)


if __name__ == "__main__":
    main()
