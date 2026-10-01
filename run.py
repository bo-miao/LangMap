#!/usr/bin/env python3

import os
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['GLOG_minloglevel'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat import Env
from tqdm import trange
import json
import math
import numpy as np
from planner import call_planner, check_planner, convert_numpy_types
from navmesh_utils import install_habitat_lab_navmesh, nearest_goal_distance

SUCCESS_DISTANCE = 0.25  # success: final geodesic distance to the nearest goal viewpoint (m)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--exp-save",
        type=str,
        required=True,
        help="results types requried to be saved",
    )

    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    parser.add_argument(
        "--vlm",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--check", type=int, default=1,
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, exp_save: str, vlm: str, check: int, opts=None) -> None:
    config = get_config(exp_config, opts)
    config.defrost()
    config.TASK_CONFIG.DATASET.CHECK = 1 if check else 0
    config.freeze()
    if vlm:
        check_planner(vlm)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num, allow_uneven_splits=True)[split_id]

    evaluate_agent(config, split_id, dataset_split, model_path, result_path, exp_save, vlm=vlm)


def evaluate_agent(config, split_id, dataset, model_path, result_path, exp_save, vlm=None) -> None:
    print(f"Creating environment with config: {config.TASK_CONFIG.DATASET.TYPE}")
    print(f"Scenes directory: {config.TASK_CONFIG.DATASET.SCENES_DIR}")
    print(f"Data path: {config.TASK_CONFIG.DATASET.DATA_PATH}")
    print(f"Navmesh preset: {config.TASK_CONFIG.SIMULATOR.NAVMESH.PRESET}")
    install_habitat_lab_navmesh()
    env = Env(config.TASK_CONFIG, dataset)
    print("Environment created successfully!")

    from agent_model import UniNaVid_Agent
    agent = UniNaVid_Agent(model_path, result_path, exp_save)

    num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS # 500

    # Check for already completed episodes (resume functionality)
    completed_episodes = set()
    log_dir = os.path.join(result_path, "log")
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("stats_") and f.endswith(".json"):
                episode_id = f.replace("stats_", "").replace(".json", "")
                completed_episodes.add(episode_id)

    mllm_fail_count = 0
    for episode_idx in trange(num_episodes, desc=config.EVAL.IDENTIFICATION + "-{}".format(split_id)):
        obs = env.reset()
        # Skip if already completed
        if env.current_episode.episode_id in completed_episodes:
            print(f"[Episode {episode_idx}] Task {env.current_episode.episode_id}: skipped because result already exists.")
            continue

        print(f"[Episode {episode_idx}] Task ID: {env.current_episode.episode_id}  Scene: {env.current_episode.scene_id}")
        print(f"[Episode {episode_idx}] Instruction: {env.current_episode.instruction.instruction_text}")
        agent.reset()
        continuse_rotation_count = 0
        last_dtg = 999
        iter_step, init_steps = 0, 0
        goals = env.current_episode.goals
        gt_goal_dist = nearest_goal_distance(env.sim, env.sim.get_agent_state().position, goals)  # shortest path for SPL

        if vlm != "" and vlm is not None:
            panorama, target_steps = [], []
            for step in range(12):
                if step % 4 == 0:
                    img = obs["rgb"]
                    H, W, C = img.shape
                    mid = W // 2
                    panorama.append(img[:, :mid, :])
                    panorama.append(img[:, mid:, :])
                    target_steps.append((step + 1) % 12)
                    target_steps.append((step - 1) % 12)

                obs = env.step({"action": 2})
                init_steps += 1

            target_index, final_reason = call_planner(instruction=obs["instruction"]["text"], images=panorama, model=vlm, temporal=False)
            if final_reason == "Call planner failed.":
                mllm_fail_count += 1
                if mllm_fail_count >= 2:
                    raise RuntimeError(f"[{env.current_episode.episode_id}] The VLM planner failed {mllm_fail_count} times "
                                       "in a row; stopping the evaluation.")
            else:
                mllm_fail_count = 0
            left_steps, right_steps = target_steps[target_index], (12 - target_steps[target_index]) % 12
            if right_steps <= left_steps:
                for _ in range(right_steps):
                    obs = env.step({"action": 3})
                    init_steps += 1
            else:
                for _ in range(left_steps):
                    obs = env.step({"action": 2})
                    init_steps += 1

        while not env.episode_over:
            info = env.get_metrics()
            dtg = info["distance_to_goal"]
            if dtg != last_dtg and not (np.isnan(dtg) and np.isnan(last_dtg)):  # NaN != NaN
                last_dtg = dtg
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id, iter_step)
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > (EARLY_STOP_STEPS - init_steps):
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)

        info = env.get_metrics()
        final_pos = env.sim.get_agent_state().position
        # Success: the final position is within SUCCESS_DISTANCE (geodesic) of a goal viewpoint, however the
        # episode ended (STOP, early stop, or step limit). Tasks whose goals are unreachable count as failures.
        final_goal_dist = nearest_goal_distance(env.sim, final_pos, goals)
        success = float(final_goal_dist < SUCCESS_DISTANCE)
        path_length = float(info["path_length"])
        spl = gt_goal_dist / max(path_length, gt_goal_dist) if success and math.isfinite(gt_goal_dist) else 0.0

        print(f"\n[Episode {episode_idx}] ======= FINAL STATE =======")
        print(f"[Episode {episode_idx}] Task completed after {iter_step} steps")
        print(f"[Episode {episode_idx}] Final Position: {final_pos}")
        print(f"[Episode {episode_idx}] SR:  {success} (1=success, 0=failure)")
        print(f"[Episode {episode_idx}] SPL: {spl:.4f}")
        print(f"[Episode {episode_idx}] ==========================\n")

        result_dict = {
            "success": success,
            "spl": spl,
            "path_length": path_length,
            "id": env.current_episode.episode_id,
            "instruction": env.current_episode.instruction.instruction_text,
            "final_goal_dist": final_goal_dist if math.isfinite(final_goal_dist) else None,
        }
        result_dict = convert_numpy_types(result_dict)
        if "data" in exp_save:
            with open(os.path.join(os.path.join(result_path, "log"), "stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()
