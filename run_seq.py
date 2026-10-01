#!/usr/bin/env python3
import os

os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['GLOG_minloglevel'] = '2'
# Prevent multi-threading issues
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import gc
import gzip
import json
from typing import Dict
import copy
import math
import numpy as np
import quaternion
import torch
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from VLN_CE.habitat_extensions.task import ExtendedInstructionData, VLNExtendedEpisode
from habitat import Env
from habitat.tasks.nav.nav import NavigationGoal
from tqdm import trange
from planner import call_planner, check_planner, convert_numpy_types
from navmesh_utils import install_habitat_lab_navmesh, nearest_goal_distance

SUCCESS_DISTANCE = 0.25  # success: final geodesic distance to the nearest goal viewpoint (m)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config", type=str, required=True, help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--exp-save", type=str, required=True, help="results types requried to be saved",
    )

    parser.add_argument(
        "--split-num", type=int, required=True, help="chunks of evluation"
    )

    parser.add_argument(
        "--split-id", type=int, required=True, help="chunks ID of evluation"
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="location of model weights"
    )

    parser.add_argument(
        "--result-path", type=str, required=True, help="location to save results"
    )

    parser.add_argument(
        "--vlm", type=str, default="",
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

    evaluate_agent_seq(config, split_id, dataset_split, model_path, result_path, exp_save, vlm=vlm)


def generate_subtask_instruction(sub_episode, all_goals_dict, region_dict, nav_type):
    """
    Generate instruction text for a subtask based on its navigation type.
    By default, we use concise descriptions for region- and instance-level tasks.
    """
    if nav_type in ["region", "room", "object"]:
        obj_category = sub_episode['object_category']
    else:
        first_target_id = sub_episode["target_object_ids"][0]
        obj_category = all_goals_dict[first_target_id].get("object_category", "object")

    if nav_type == "object":
        instruction_text = f"Find the {obj_category}"
    elif nav_type == "room":
        room_name = sub_episode.get("room_name", "room")
        instruction_text = f"Find the {obj_category} in the {room_name}"
        if 'tie:' in room_name.lower():
            instruction_text += ", 'Tie: A & B' means the area spans both rooms A and B"
    elif nav_type == "region":
        region_id = sub_episode.get("region_id", "")
        if region_id in region_dict:
            region_info = region_dict[region_id]
            category = region_info.get('region_category', 'region').lower()
            desc = region_info.get('concise_description', '')
            instruction_text = f"Find the {obj_category} in the {category} that has {desc}"
            if 'tie:' in category.lower() or 'tie:' in desc.lower():
                instruction_text += ", 'Tie: A & B' means the area spans both rooms A and B"
        else:
            instruction_text = f"Find the {obj_category} in the target region"
    elif nav_type == "instance":
        inst_id = sub_episode.get("instance_id", "")
        if inst_id in all_goals_dict:
            inst_desc = all_goals_dict[inst_id].get('annot_unique_concise_description', 'object')
            instruction_text = f"Find the {inst_desc}"
        else:
            instruction_text = "Find the target instance"
    else:
        instruction_text = f"Find the target {nav_type}"

    return instruction_text


def normalize_observations(obs: Dict) -> Dict:
    if 'rgb' not in obs:
        rgb_keys = ['color_sensor', 'RGB_SENSOR', 'rgb_sensor', 'rgb', 'RGB']
        found_rgb = False

        for key in rgb_keys:
            if key in obs:
                obs['rgb'] = obs[key]
                found_rgb = True
                break

        if not found_rgb:
            for key in obs.keys():
                key_lower = key.lower()
                if 'rgb' in key_lower or ('color' in key_lower and 'sensor' in key_lower):
                    obs['rgb'] = obs[key]
                    found_rgb = True
                    break

        if not found_rgb and len(obs) > 0:
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    if len(value.shape) == 3 and value.shape[2] in [3, 4]:
                        obs['rgb'] = value[:, :, :3] if value.shape[2] == 4 else value
                        found_rgb = True
                        break

        if not found_rgb:
            print(f"WARNING: Could not find RGB observation. Available keys: {list(obs.keys())}")

    return obs

def create_subtask_goals(sub_episode, all_goals_dict):
    """Create goals from sub_episode target_object_ids."""
    goals = []
    for obj_id in sub_episode.get("target_object_ids", []):
        if obj_id in all_goals_dict:
            goal_info = all_goals_dict[obj_id]
            viewpoints = goal_info.get("view_points", [])
            if viewpoints:
                for vp in viewpoints:
                    agent_state = vp.get("agent_state", {})
                    vp_position = agent_state.get("position")
                    if vp_position:
                        goals.append(NavigationGoal(
                            position=vp_position,
                            radius=1.0
                        ))
            else:
                goals.append(NavigationGoal(
                    position=goal_info.get("position", [-100, -100, -100]),
                    radius=1.0
                ))
    return goals


def load_scene_data(scene_name, data_path):
    scene_file = f"{scene_name}.json.gz"
    scene_path = os.path.join(data_path, scene_file)

    with gzip.open(scene_path, 'rt', encoding='utf-8') as f:
        scene_data = json.load(f)

    all_goals_dict = {g['object_id']: g for g in scene_data["goals"]}
    region_dict = scene_data.get("region_annotation", {})
    episode_mapping = {
        "object": scene_data.get('episodes_by_object_level', []),
        "room": scene_data.get('episodes_by_room_level', []),
        "region": scene_data.get('episodes_by_region_level', []),
        "instance": scene_data.get('episodes_by_instance_level', [])
    }

    return all_goals_dict, region_dict, episode_mapping


def get_observations(env, episode):
    """Observations at the current agent pose; needed after the agent is moved outside env.step."""
    sim_obs = env.sim.get_sensor_observations()
    sim_sensor_obs = env.sim.sensor_suite.get_observations(sim_obs)
    task_obs = env.task.sensor_suite.get_observations(observations=sim_obs, episode=episode, task=env.task)
    return normalize_observations({**sim_sensor_obs, **task_obs})


def get_current_pos(env):
    return env.sim.get_agent_state().position.copy()

def get_current_state(env):
    return env.sim.get_agent_state().position.copy(), env.sim.get_agent_state().rotation.copy()

def get_action_list_custom(end_pos, env):
    TURN_ANGLE = np.deg2rad(30)
    FORWARD_STEP = 0.25
    state = env.sim.get_agent_state()
    start_pos, start_rot = state.position, state.rotation

    # ---- current yaw ----
    forward = quaternion.rotate_vectors(start_rot, np.array([0, 0, -1]))
    start_yaw = np.arctan2(forward[0], forward[2])

    # ---- target direction ----
    dx = end_pos[0] - start_pos[0]
    dz = end_pos[2] - start_pos[2]
    target_yaw = np.arctan2(dx, dz)

    # ---- angle difference ----
    diff = target_yaw - start_yaw
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    actions = []
    # ---- rotation actions ----
    if diff > 0:
        n_turn = int(round(diff / TURN_ANGLE))
        actions += ["turn_left"] * n_turn
    else:
        n_turn = int(round(-diff / TURN_ANGLE))
        actions += ["turn_right"] * n_turn
    # ---- forward actions ----
    dist = np.linalg.norm([dx, dz])
    EPS = 1e-2
    n_forward = int(round((dist + EPS) / FORWARD_STEP))
    actions += ["move_forward"] * n_forward
    return actions if n_forward > 0 else []

def perform_actions(action_list, env, init_steps, global_steps):
    for action in action_list:
        if action is not None and not env.episode_over:
            if action == "move_forward":
                run_action = {"action": 1}
            elif action == "turn_left":
                run_action = {"action": 2}
            elif action == "turn_right":
                run_action = {"action": 3}
            env.step(run_action)
            init_steps += 1
            global_steps.append(get_current_state(env))
    return init_steps, global_steps

def ecu_dist(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def move_without_pathfinder(pos2, env, init_steps, global_steps, reach_th=0.25):
    curr_pos = get_current_pos(env)
    init_steps, global_steps = perform_actions(get_action_list_custom(pos2, env), env, init_steps, global_steps)
    if ecu_dist(get_current_pos(env), pos2) < reach_th:
        return True, init_steps, global_steps
    init_steps, global_steps = perform_actions(get_action_list_custom(curr_pos, env), env, init_steps, global_steps)
    return False, init_steps, global_steps

def navigate_using_compressed_trajectory(path_list, env, init_steps, global_steps, dist_th=0.25):
    if len(path_list) <= 3:
        for pos in path_list:
            init_steps, global_steps = perform_actions(get_action_list_custom(pos, env), env, init_steps, global_steps)
        return init_steps, global_steps

    init_steps, global_steps = perform_actions(get_action_list_custom(path_list[0], env), env, init_steps, global_steps)
    i = 0
    while i < len(path_list) - 1 and not env.episode_over:
        merged = False
        for j in range(len(path_list) - 1, i + 1, -1):
            if ecu_dist(path_list[i], path_list[j]) <= dist_th + 0.01:
                tag, init_steps, global_steps = move_without_pathfinder(path_list[j], env, init_steps, global_steps)
                if tag:
                    i = j
                    merged = True
                    break
        if not merged:
            i += 1
            init_steps, global_steps = perform_actions(get_action_list_custom(path_list[i], env), env, init_steps, global_steps)
    return init_steps, global_steps


def evaluate_agent_seq(config, split_id, dataset, model_path, result_path, exp_save, vlm=None) -> None:
    print(f"Creating environment with config: {config.TASK_CONFIG.DATASET.TYPE}")
    print(f"Scenes directory: {config.TASK_CONFIG.DATASET.SCENES_DIR}")
    print(f"Data path: {config.TASK_CONFIG.DATASET.DATA_PATH}")
    install_habitat_lab_navmesh()
    env = Env(config.TASK_CONFIG, dataset)
    print("Environment created successfully!")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    from agent_model import UniNaVid_Agent
    agent = UniNaVid_Agent(model_path, result_path, exp_save)

    num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    completed_tasks = set()
    log_dir = os.path.join(result_path, "log")
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("stats_") and f.endswith(".json"):
                task_id = f.replace("stats_", "").replace(".json", "")
                completed_tasks.add(task_id)
    completed_episodes = set([x.rsplit("_", 1)[0] for x in completed_tasks])

    current_scene_name = None
    current_all_goals_dict = None
    current_region_dict = None
    current_episode_mapping = None
    for episode_idx in trange(num_episodes, desc=config.EVAL.IDENTIFICATION + "-seq-{}".format(split_id)):
        seq_episode = env.episodes[episode_idx]
        if not seq_episode.info.get("is_sequence", False):
            print(f"[Episode {episode_idx}] WARNING: Not a sequence episode, skipping")
            continue
        # skip if already been processed
        episode_unique_id = seq_episode.episode_id
        if episode_unique_id in completed_episodes:
            print(f"[Episode {episode_idx}] Sequence {episode_unique_id}: skipped because result already exists.")
            continue

        scene_name = seq_episode.info.get("scene_name")
        task_sequence = seq_episode.info.get("task_sequence", [])
        print(f"\n\n[Sequence Episode {episode_idx}] Episode ID: {episode_unique_id} **********************************************************")
        print(f"[Sequence Episode {episode_idx}] Number of subtasks: {len(task_sequence)} ---- {task_sequence}")
        # Load scene data in json if needed (for episode_mapping)
        if scene_name != current_scene_name:
            print(f"[Sequence Episode {episode_idx}] Loading scene data for {scene_name}...")
            current_all_goals_dict, current_region_dict, current_episode_mapping = load_scene_data(scene_name, config.TASK_CONFIG.DATASET.DATA_PATH)
            current_scene_name = scene_name

        episode_iter_step, global_steps = 0, []
        agent_current_pos, agent_current_rot = None, None
        final_result_dicts = {}
        mllm_fail_count = 0
        for subtask_idx, task_ref in enumerate(task_sequence):
            first_subtask = subtask_idx == 0
            task_type, task_idx = task_ref
            subtask_unique_id = f"{episode_unique_id}_{subtask_idx}"
            print(f"\n\n[Sequence Episode {episode_idx}] ======== SUBTASK {subtask_idx + 1}/{len(task_sequence)} ========")
            print(f"[Sequence Episode {episode_idx}] Subtask ID: {subtask_unique_id}")
            print(f"[Sequence Episode {episode_idx}] Task type: {task_type}")
            if task_type not in current_episode_mapping or task_idx >= len(current_episode_mapping[task_type]):
                print(f"[Sequence Episode {episode_idx}] ERROR: Invalid task reference [{task_type}, {task_idx}], skipping")
                continue

            # get current single goal task
            sub_episode = current_episode_mapping[task_type][task_idx]
            # Generate instruction and goals for this subtask
            instruction_text = generate_subtask_instruction(sub_episode, current_all_goals_dict, current_region_dict, task_type)
            goals = create_subtask_goals(sub_episode, current_all_goals_dict)
            if not goals:
                continue
            print(f"[Sequence Episode {episode_idx}] Instruction: {instruction_text}")

            assert first_subtask == (subtask_idx == 0), f"Mismatch, {first_subtask} - {subtask_idx}"
            if first_subtask:
                start_pos = list(seq_episode.start_position)
                start_rot = list(seq_episode.start_rotation)
                print(f"[Sequence Episode {episode_idx}] First subtask: starting from the sequence start position {start_pos}")
            else:
                if agent_current_pos is None:
                    print(f"[Sequence Episode {episode_idx}] WARNING: No previous position, using sequence start")  # might be problematic on resume
                    start_pos = list(seq_episode.start_position)
                    start_rot = list(seq_episode.start_rotation)
                else:
                    start_pos = list(agent_current_pos)
                    if hasattr(agent_current_rot, 'components'):
                        comps = list(agent_current_rot.components)
                        start_rot = [comps[1], comps[2], comps[3], comps[0]]  # [x, y, z, w]
                    else:
                        start_rot = [agent_current_rot.imag[0], agent_current_rot.imag[1], agent_current_rot.imag[2], agent_current_rot.real]
                    print(f"[Sequence Episode {episode_idx}] Starting from the previous subtask stop position: {start_pos}")

            subtask_episode = VLNExtendedEpisode(
                episode_id=subtask_unique_id,
                scene_id=seq_episode.scene_id,
                start_position=start_pos,
                start_rotation=start_rot,
                goals=goals,
                instruction=ExtendedInstructionData(
                    instruction_text=instruction_text,
                    instruction_id=subtask_unique_id,
                    language="en-US"
                ),
                trajectory_id=subtask_unique_id,
                info={
                    "navigation_type": task_type,
                    "scene_name": scene_name,
                    "subtask_index": subtask_idx,
                    "sequence_episode_id": episode_unique_id
                }
            )

            # **************************************
            # ***** Reset environment
            # **************************************
            env._current_episode = subtask_episode
            if first_subtask:
                env._reset_stats()
                env.reconfigure(env._config)
                obs = env.task.reset(episode=subtask_episode)
                env._task.measurements.reset_measures(episode=subtask_episode, task=env.task)
                agent.reset(preserve_history=False)
                obs = normalize_observations(obs)
            else:
                env._current_episode = subtask_episode
                env._reset_stats()
                env.task.reset(episode=subtask_episode)
                env.sim.set_agent_state(position=np.array(start_pos), rotation=start_rot)
                env._task.measurements.reset_measures(episode=subtask_episode, task=env.task)
                obs = get_observations(env, subtask_episode)
                agent.reset(preserve_history=True)

            if first_subtask:
                global_steps.append(get_current_state(env))

            iter_step, init_steps, continuse_rotation_count = 0, 0, 0
            last_dtg = 999
            info = env.get_metrics()
            start_position = get_current_pos(env)
            a = np.asarray(start_pos, dtype=np.float64)
            b = np.asarray(start_position, dtype=np.float64)
            assert np.allclose(a, b, atol=1e-4, rtol=0.0), f"Start position does not match agent position {a.tolist()} and {b.tolist()}"
            gt_goal_dist = nearest_goal_distance(env.sim, start_position, goals)  # shortest-path length for SPL

            print(f"[Sequence Episode {episode_idx}] ======== SUBTASK {subtask_idx + 1} INITIAL STATE ========")
            obs = get_observations(env, subtask_episode)

            # **************************************
            # ***** Planner
            # **************************************
            if vlm != "" and vlm is not None:
                if not first_subtask:
                    memory = agent.history_memory.get_memory()
                    memory_frames = [x[1] for x in memory]
                    target_index, final_reason = call_planner(instruction=instruction_text, images=memory_frames, model=vlm, temporal=True)
                    _, _, _, _, target_step = (
                        memory[target_index][0], memory[target_index][2], memory[target_index][3], memory[target_index][5], memory[target_index][6])

                    go_to_waypoint_pos_list_ = [x[0] for x in global_steps[target_step:]][::-1]
                    init_steps, global_steps = navigate_using_compressed_trajectory(go_to_waypoint_pos_list_, env, init_steps, global_steps, dist_th=0.25)
                    obs = get_observations(env, subtask_episode)  # the agent moved: the panorama starts from the new view
                    print(f"[Waypoint Response:] {target_index}, [Reason:] {final_reason}")

                panorama, target_steps = [], []
                for step in range(12):
                    if env.episode_over:
                        break
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
                    global_steps.append(get_current_state(env))

                if not env.episode_over:
                    target_index, final_reason = call_planner(instruction=instruction_text, images=panorama, model=vlm, temporal=False)
                    if final_reason == "Call planner failed." or final_reason is None:
                        mllm_fail_count += 1
                        if mllm_fail_count >= 2:
                            raise RuntimeError(f"[{subtask_unique_id}] The VLM planner failed {mllm_fail_count} times in a row; "
                                               "stopping the evaluation. Unfinished sequences are re-run on resume.")
                    else:
                        mllm_fail_count = 0

                    left_steps, right_steps = target_steps[target_index], (12 - target_steps[target_index]) % 12
                    turn_action, n_turn = ({"action": 3}, right_steps) if right_steps <= left_steps else ({"action": 2}, left_steps)
                    for _ in range(n_turn):
                        if env.episode_over:
                            break
                        obs = env.step(turn_action)
                        init_steps += 1
                        global_steps.append(get_current_state(env))
                    print(f"[Heading Response:] {target_index}, [Reason:] {final_reason}")
                print("==============================================================")


            # **************************************
            # ***** Reactive Navigation
            # **************************************
            while not env.episode_over:
                info = env.get_metrics()
                dtg = info["distance_to_goal"]
                if dtg != last_dtg and not (np.isnan(dtg) and np.isnan(last_dtg)):  # NaN != NaN
                    last_dtg = dtg
                    continuse_rotation_count = 0
                else:
                    continuse_rotation_count += 1

                action = agent.act(obs, info, subtask_unique_id, episode_iter_step,
                                   agent_pos=global_steps[-1][0], agent_rot=global_steps[-1][1], global_step=len(global_steps) - 1)
                if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > (EARLY_STOP_STEPS - init_steps):
                    action = {"action": 0}  # STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT

                obs = env.step(action)
                obs = normalize_observations(obs)

                iter_step += 1
                global_steps.append(get_current_state(env))
                episode_iter_step += 1

            info = env.get_metrics()
            final_pos, final_rot = get_current_state(env)
            # Success: the final position is within SUCCESS_DISTANCE (geodesic) of a goal viewpoint, however the
            # subtask ended (STOP, early stop, or step limit). Subtasks whose goals are unreachable count as failures.
            final_goal_dist = nearest_goal_distance(env.sim, final_pos, goals)
            success = float(final_goal_dist < SUCCESS_DISTANCE)
            path_length = float(info["path_length"])
            spl = gt_goal_dist / max(path_length, gt_goal_dist) if success and math.isfinite(gt_goal_dist) else 0.0

            print(f"\n[Sequence Episode {episode_idx}] ======== SUBTASK {subtask_idx + 1} FINAL STATE ========")
            print(f"[Sequence Episode {episode_idx}] Subtask completed after {iter_step} steps")
            print(f"[Sequence Episode {episode_idx}] Start Position: {start_position}")
            print(f"[Sequence Episode {episode_idx}] Final Position: {list(final_pos)}")
            print(f"[Sequence Episode {episode_idx}] SR:  {success} (1=success, 0=failure)")
            print(f"[Sequence Episode {episode_idx}] SPL: {spl:.4f}")
            print(f"[Sequence Episode {episode_idx}] =======================================\n")

            # Save results for this subtask
            result_dict = {
                "success": success,
                "spl": spl,
                "path_length": path_length,
                "id": subtask_unique_id,
                "instruction": instruction_text,
                "navigation_type": task_type,
                "final_goal_dist": final_goal_dist if math.isfinite(final_goal_dist) else None,
            }
            result_dict = convert_numpy_types(result_dict)
            if "data" in exp_save:  # store to episode-level dict
                os.makedirs(log_dir, exist_ok=True)
                final_result_dicts[subtask_unique_id] = copy.deepcopy(result_dict)

            # Update agent position for next subtask (persist position across subtasks)
            # Store rotation as quaternion object (will convert to list when needed)
            agent_current_pos = final_pos
            agent_current_rot = final_rot
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # save after finishing an episode.
        for subtask_unique_id, result_dict in final_result_dicts.items():
            with open(os.path.join(log_dir, "stats_{}.json".format(subtask_unique_id)), "w") as f:
                json.dump(result_dict, f, indent=4)
        print(f"[Sequence Episode {episode_idx}] === MULTI_GOAL EPISODE COMPLETE, SAVED AT {log_dir} ===\n\n\n")

if __name__ == "__main__":
    main()

