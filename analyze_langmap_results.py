#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from pathlib import Path
import math

# Single-goal tasks with annotation issues: excluded (same list as black_task_ids_ in VLN_CE/habitat_extensions/task.py).
BLACK_TASK_IDS = {
    '00800-TEEsavR23oF_room_74', '00810-CrMo8WxCyVb_object_30', '00810-CrMo8WxCyVb_object_31',
    '00810-CrMo8WxCyVb_region_75', '00810-CrMo8WxCyVb_region_76', '00810-CrMo8WxCyVb_room_53',
    '00810-CrMo8WxCyVb_room_54', '00820-mL8ThkuaVTM_object_29', '00820-mL8ThkuaVTM_region_47',
    '00820-mL8ThkuaVTM_room_39', '00823-7MXmsvcQjpJ_room_100', '00823-7MXmsvcQjpJ_room_101',
    '00823-7MXmsvcQjpJ_room_74', '00823-7MXmsvcQjpJ_room_98', '00823-7MXmsvcQjpJ_room_99', '00829-QaLdnwvtxbs_room_14',
    '00829-QaLdnwvtxbs_room_16', '00829-QaLdnwvtxbs_room_21', '00829-QaLdnwvtxbs_room_22', '00829-QaLdnwvtxbs_room_23',
    '00829-QaLdnwvtxbs_room_24', '00829-QaLdnwvtxbs_room_25', '00829-QaLdnwvtxbs_room_5', '00829-QaLdnwvtxbs_room_7',
    '00829-QaLdnwvtxbs_room_9', '00832-qyAac8rV8Zk_region_18', '00832-qyAac8rV8Zk_room_18',
    '00839-zt1RVoi7PcG_object_25', '00839-zt1RVoi7PcG_region_70', '00839-zt1RVoi7PcG_room_54',
    '00862-LT9Jq6dN3Ea_object_46', '00862-LT9Jq6dN3Ea_region_135', '00862-LT9Jq6dN3Ea_region_37',
    '00862-LT9Jq6dN3Ea_room_21', '00862-LT9Jq6dN3Ea_room_93', '00871-VBzV5z6i1WS_object_21',
    '00871-VBzV5z6i1WS_object_28', '00871-VBzV5z6i1WS_object_29', '00871-VBzV5z6i1WS_object_38',
    '00871-VBzV5z6i1WS_object_8', '00871-VBzV5z6i1WS_region_32', '00871-VBzV5z6i1WS_region_58',
    '00871-VBzV5z6i1WS_region_67', '00871-VBzV5z6i1WS_region_68', '00871-VBzV5z6i1WS_region_77',
    '00871-VBzV5z6i1WS_room_21', '00871-VBzV5z6i1WS_room_39', '00871-VBzV5z6i1WS_room_46', '00871-VBzV5z6i1WS_room_47',
    '00871-VBzV5z6i1WS_room_56', '00873-bxsVRursffK_object_29', '00873-bxsVRursffK_region_52',
    '00873-bxsVRursffK_room_39', '00876-mv2HUxq3B53_object_16', '00876-mv2HUxq3B53_object_31',
    '00876-mv2HUxq3B53_object_40', '00876-mv2HUxq3B53_region_33', '00876-mv2HUxq3B53_region_51',
    '00876-mv2HUxq3B53_region_69', '00876-mv2HUxq3B53_region_81', '00876-mv2HUxq3B53_region_92',
    '00876-mv2HUxq3B53_region_93', '00876-mv2HUxq3B53_room_19', '00876-mv2HUxq3B53_room_31',
    '00876-mv2HUxq3B53_room_45', '00876-mv2HUxq3B53_room_50', '00876-mv2HUxq3B53_room_59',
    '00877-4ok3usBNeis_object_44', '00877-4ok3usBNeis_region_58', '00877-4ok3usBNeis_region_8',
    '00877-4ok3usBNeis_room_52', '00877-4ok3usBNeis_room_9', '00878-XB4GS9ShBRE_object_18',
    '00878-XB4GS9ShBRE_region_33', '00878-XB4GS9ShBRE_room_32', '00880-Nfvxx8J5NCo_object_13',
    '00880-Nfvxx8J5NCo_object_30', '00880-Nfvxx8J5NCo_object_31', '00880-Nfvxx8J5NCo_region_22',
    '00880-Nfvxx8J5NCo_region_45', '00880-Nfvxx8J5NCo_region_46', '00880-Nfvxx8J5NCo_room_23',
    '00880-Nfvxx8J5NCo_room_44', '00880-Nfvxx8J5NCo_room_45', '00890-6s7QHgap2fW_region_19',
    '00890-6s7QHgap2fW_room_16', '00891-cvZr5TUy5C5_object_12', '00891-cvZr5TUy5C5_object_43',
    '00891-cvZr5TUy5C5_object_44', '00891-cvZr5TUy5C5_region_109', '00891-cvZr5TUy5C5_region_110',
    '00891-cvZr5TUy5C5_region_111', '00891-cvZr5TUy5C5_region_112', '00891-cvZr5TUy5C5_region_39',
    '00891-cvZr5TUy5C5_region_40', '00891-cvZr5TUy5C5_region_44', '00891-cvZr5TUy5C5_room_34',
    '00891-cvZr5TUy5C5_room_35', '00891-cvZr5TUy5C5_room_36', '00891-cvZr5TUy5C5_room_92', '00891-cvZr5TUy5C5_room_93',
    '00891-cvZr5TUy5C5_room_94', '00891-cvZr5TUy5C5_room_95', '00891-cvZr5TUy5C5_room_96', '00891-cvZr5TUy5C5_room_97',
    '00894-HY1NcmCgn3n_instance_7'
}


def parse_filename(filename):
    """
    Parse filename like: stats_00800-TEEsavR23oF_sequence_0_2.json
    Returns: (episode_id, scene_id, sequence_num, run_num)
    """
    basename = os.path.basename(filename)
    # Remove 'stats_' prefix and '.json' suffix
    name = basename.replace('stats_', '').replace('.json', '')

    # Split by '_sequence_'
    parts = name.split('_sequence_')
    if len(parts) != 2:
        return None

    episode_scene = parts[0]  # e.g., "00800-TEEsavR23oF"
    seq_run = parts[1]  # e.g., "0_2"

    # Split episode_scene by first hyphen
    hyphen_idx = episode_scene.find('-')
    if hyphen_idx == -1:
        return None

    episode_id = episode_scene[:hyphen_idx]
    scene_id = episode_scene[hyphen_idx + 1:]

    # Split sequence and run
    seq_run_parts = seq_run.split('_')
    if len(seq_run_parts) != 2:
        return None

    sequence_num = int(seq_run_parts[0])
    run_num = int(seq_run_parts[1])

    return (episode_id, scene_id, sequence_num, run_num)


def load_episode_results(results_dir):
    """
    Load all result files and organize by sequence.
    Returns: dict mapping sequence_id -> list of (run_num, result_dict)
    """
    sequences = defaultdict(dict)  # sequence_id -> {run_num: result}

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return sequences

    log_dir = results_path / "log"
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return sequences

    json_files = [x for x in log_dir.glob("stats_*.json")]
    for json_file in json_files:
        parsed = parse_filename(str(json_file))
        if parsed is None:
            continue

        episode_id, scene_id, sequence_num, run_num = parsed

        # Create unique sequence identifier
        sequence_id = f"{episode_id}-{scene_id}_sequence_{sequence_num}"
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                sequences[sequence_id][run_num] = result
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return sequences


def analyze_sequences(sequences):
    """
    Analyze sequences and compute metrics.
    """
    total_sequences = len(sequences)
    total_tasks = 0
    successful_tasks = 0
    total_spl = 0.0
    valid_spl_count = 0
    total_seqsr_5, total_seqsr_4, total_seqsr_3, total_seqsr_2, total_seqsr_1 = 0, 0, 0, 0, 0
    for sequence_id, runs in sequences.items():
        runs = dict(sorted(runs.items()))
        # Get all runs (0-4)
        run_nums = sorted(runs.keys())
        for run_num in run_nums:
            result = runs[run_num]
            success = result.get('success', 0.0)
            spl = result.get('spl', 0.0)
            total_tasks += 1
            if success == 1.0:
                successful_tasks += 1
            total_spl += check_inf_nan(float(spl)) if spl is not None else 0.0
            valid_spl_count += 1
        # Check sequence-level success
        if len(run_nums) == 5:
            total_seqsr_5 += 1 if all(runs[i]['success'] for i in range(5)) else 0
            total_seqsr_4 += 1 if all(runs[i]['success'] for i in range(4)) else 0
            total_seqsr_3 += 1 if all(runs[i]['success'] for i in range(3)) else 0
            total_seqsr_2 += 1 if all(runs[i]['success'] for i in range(2)) else 0
            total_seqsr_1 += 1 if all(runs[i]['success'] for i in range(1)) else 0
        else:
            print(f"Warning: incomplete sequence {sequence_id}: expected 5 runs, got {len(run_nums)}.")

    def pct(v, n):
        return round(v / n * 100, 2)

    """Print formatted results."""
    sr = (successful_tasks / total_tasks * 100.0) if total_tasks > 0 else 0.0
    spl = (total_spl / valid_spl_count * 100.0) if valid_spl_count > 0 else 0.0
    print(f"Total Sequences: {total_sequences}; Total Tasks: {total_tasks}; Successful Tasks: {successful_tasks}")
    print("-" * 60)
    print("Metrics:")
    print("-" * 60)
    print(f"SR:         {sr:.2f}%")
    print(f"SPL:        {spl:.2f}%")
    seqsr = [total_seqsr_1, total_seqsr_2, total_seqsr_3, total_seqsr_4, total_seqsr_5]
    print("SeqSR@1-5:  " + " | ".join(f"@{k}: {pct(v, total_sequences):.1f}%" for k, v in enumerate(seqsr, 1)))
    print("-" * 60)


def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value

def analyze_single_goal_results(results_path):
    results_path = os.path.join(results_path, "log")
    result_files = sorted([
        x for x in os.listdir(results_path)
        if x.endswith(".json")
    ])
    if len(result_files) == 0:
        print(f"Warning: no valid result files found in {results_path}.")
        return

    total_tasks = 0
    successful_tasks = 0
    total_spl = 0.0

    sr_by_level = dict()
    spl_by_level = dict()

    for result_file in result_files:
        result_path = os.path.join(results_path, result_file)
        with open(result_path, "r") as f:
            data = json.load(f)

        if data["id"] in BLACK_TASK_IDS:
            continue
        goal_level = data["id"].split("_")[-2]
        success = check_inf_nan(int(data["success"]))
        spl = check_inf_nan(float(data["spl"]))

        total_tasks += 1
        successful_tasks += success
        total_spl += spl

        sr_by_level.setdefault(goal_level, []).append(success)
        spl_by_level.setdefault(goal_level, []).append(spl)

    sr = successful_tasks / total_tasks * 100.0
    spl = total_spl / total_tasks * 100.0

    print("-" * 60)
    print("Overall Metrics (concise instructions, all four goal levels):")
    print(f"SR:      {sr:.1f}%")
    print(f"SPL:     {spl:.1f}%")
    print("-" * 60)

    print("Per-level Metrics:")
    print("-" * 60)
    for goal_level in ["object", "room", "region", "instance"]:
        if goal_level not in sr_by_level:
            continue
        level_sr = sum(sr_by_level[goal_level]) / len(sr_by_level[goal_level]) * 100.0
        level_spl = sum(spl_by_level[goal_level]) / len(spl_by_level[goal_level]) * 100.0
        label = f"{goal_level.capitalize()}-level"
        print(f"{label:<15} | SR: {level_sr:6.1f}% | SPL: {level_spl:6.1f}%")
    print("-" * 60)



def main():
    """
    Analyze sequence results from LangMap / Uni-NaVid evaluation.

    Metrics:
    - SR: Success Rate
    - SPL: Success weighted by Path Length
    - SeqSR@K: Sequence Success Rate, defined as the percentage of complete
      sequences where the first K subtasks all succeed.
    """
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to results directory'
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Evaluation type, single-goal or multi-goal tasks.",
    )
    args = parser.parse_args()

    print(f"=================== Loading {args.results_dir} ============================")
    if args.eval_type == "multi":
        sequences = load_episode_results(args.results_dir)
        analyze_sequences(sequences)
    elif args.eval_type == "single":
        analyze_single_goal_results(args.results_dir)


if __name__ == '__main__':
    main()

