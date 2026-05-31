#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import defaultdict


def result_files(result_dir):
    log_dir = os.path.join(result_dir, "log")
    if os.path.exists(log_dir):
        result_dir = log_dir
    return sorted([os.path.join(result_dir, x) for x in os.listdir(result_dir) if x.endswith(".json")])


def score(x):
    try:
        return int(float(x) > 0)
    except:
        return 0


def num(x):
    try:
        x = float(x)
    except:
        return 0.0
    return 0.0 if math.isnan(x) or math.isinf(x) else x


def get_sr(data):
    return score(data.get("success", data.get("sr", 0)))


def print_sr_spl(name, results):
    if len(results) == 0:
        return
    sr = sum(get_sr(x) for x in results) / len(results) * 100
    spl = sum(num(x.get("spl", 0)) for x in results) / len(results) * 100
    print(f"{name:<12} N={len(results):<6} SR={sr:6.2f}% SPL={spl:6.2f}%")


def analyze_single(result_dir):
    by_type = defaultdict(list)
    all_results = []

    for path in result_files(result_dir):
        data = json.load(open(path, "r"))
        task_id = data.get("id", os.path.basename(path).replace("stats_", "").replace(".json", ""))
        task_type = data.get("task_type")
        by_type[task_type].append(data)
        all_results.append(data)

    print("\n=== Single-goal ===")
    print_sr_spl("overall", all_results)
    for task_type in ["object", "room", "region", "instance", "unknown"]:
        print_sr_spl(task_type, by_type[task_type])


def load_multi(result_dir):
    seq_dict = defaultdict(dict)
    old_pattern = re.compile(r"(.+)_sequence_(\d+)_(\d+)$")

    for path in result_files(result_dir):
        data = json.load(open(path, "r"))

        # New format: one JSON file per episode, with data["sequence"].
        if isinstance(data.get("sequence"), list):
            scene_name = data.get("scene_name", os.path.basename(path).split("_sequence_")[0])
            episode_id = data.get("episode_id", os.path.basename(path).split("_sequence_")[-1].replace(".json", ""))
            seq_key = f"{scene_name}_sequence_{episode_id}"
            for item in data["sequence"]:
                seq_dict[seq_key][int(item["task_id"])] = item
            continue

        # Old format: one JSON file per subgoal, stats_{scene}_sequence_{episode_id}_{task_id}.json.
        result_id = data.get("id", os.path.basename(path).replace("stats_", "").replace(".json", ""))
        match = old_pattern.match(result_id)
        if match:
            seq_key = f"{match.group(1)}_sequence_{match.group(2)}"
            seq_dict[seq_key][int(match.group(3))] = data

    return seq_dict


def analyze_multi(result_dir, seq_len=5):
    seq_dict = load_multi(result_dir)
    all_results = [x for seq in seq_dict.values() for x in seq.values()]

    print("\n=== Multi-goal ===")
    sr = sum(get_sr(x) for x in all_results) / len(all_results) * 100 if len(all_results) else 0
    spl = sum(num(x.get("spl", 0)) for x in all_results) / len(all_results) * 100 if len(all_results) else 0
    print(f"SR:      {sr:.2f}%")
    print(f"SPL:     {spl:.2f}%")

    for k in range(1, seq_len + 1):
        valid = [seq for seq in seq_dict.values() if all(i in seq for i in range(k))]
        passed = sum(all(get_sr(seq[i]) for i in range(k)) for seq in valid)
        seqsr = passed / len(valid) * 100 if len(valid) else 0
        print(f"SeqSR@{k}: {seqsr:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/bo/Documents/VLN/AAA_LangMap_Models/LangMap/langmap_results")
    parser.add_argument("--eval-type", choices=["single", "multi"], default="multi")
    parser.add_argument("--seq-len", type=int, default=5)
    args = parser.parse_args()

    # This script analyzes all existing result files. Check result completeness before running it.
    if args.eval_type == "single":
        analyze_single(args.results_dir)
    else:
        analyze_multi(args.results_dir, args.seq_len)


if __name__ == "__main__":
    main()
