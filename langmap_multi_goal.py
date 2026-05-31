import argparse
import gzip
import json
import os
import time
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import habitat_sim
from habitat_common.simulator import HabitatSimulator


parser = argparse.ArgumentParser()
parser.add_argument("--scene_path", default="/home/bo/Documents/VLN/temp_datasets/hm3d/val")
parser.add_argument("--langmap_path", default="/home/bo/Documents/VLN/AAA_LangMap_Models/PlaNavid/data/LangMap/annotations")
parser.add_argument("--result_dir", default="langmap_results")
parser.add_argument("--use_concise_description", type=int, default=1)
args = parser.parse_args()


def load_json(path):
    with gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r") as f:
        return json.load(f)


def get_sentence(task_type, cur_task, region_dict, goal_dict, concise_description=True):
    if task_type == "object":
        return cur_task["object_category"]

    if task_type == "room":
        return f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"

    if task_type == "region":
        region = region_dict[cur_task["region_id"]]
        region_desc = region["concise_description"] if concise_description else region["detailed_description"]
        return f"{cur_task['object_category']} in the {region['region_category'].lower()} that has {region_desc}"

    if task_type == "instance":
        goal = goal_dict[cur_task["instance_id"]]
        return goal["annot_unique_concise_description"] if concise_description else goal["annot_unique_detailed_description"]

    raise ValueError(task_type)


def evaluate_multi_goal(args):
    os.makedirs(args.result_dir, exist_ok=True)
    scene_files = sorted([x for x in os.listdir(args.langmap_path) if x.endswith(".json.gz") and not x.endswith(".json")])
    print(f"\nTotal {len(scene_files)} selected scenes: {scene_files}")

    for scene_file in tqdm(scene_files):
        scene_name = scene_file.replace(".json.gz", "").replace(".json", "")
        scene_data = load_json(os.path.join(args.langmap_path, scene_file))

        # Parse JSON.
        region_dict = scene_data["region_annotation"]
        goal_dict = {x["object_id"]: x for x in scene_data["goals"]}
        episode_dict = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }
        sequence_episodes = scene_data.get("episode_by_sequence", scene_data.get("episodes_by_sequence", []))

        # Run each multi-goal episode.
        for episode in sequence_episodes:
            episode_id = episode["episode_id"]
            start_position = episode['start_position']
            start_rotation = episode['start_rotation']
            episode_result = {
                "scene_name": scene_name,
                "episode_id": episode_id,
                "navigation_type": "sequence",
                "start_position": start_position,
                "start_rotation": start_rotation,
                "sequence": [],
            }


            ''' Load simulator. '''
            sim_settings = OmegaConf.load('habitat_config/langmap_sim_config.yaml')
            goat_agent_setting = OmegaConf.load('habitat_config/langmap_agent_config.yaml')
            sim_settings['scene'] = os.path.join(args.scene_path, scene_name, f"{scene_name.split('-')[-1]}.basis.glb")
            abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
            sim = abstract_sim.simulator
            agent = abstract_sim.agent
            agent_state = habitat_sim.AgentState()
            agent_state.position = start_position
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)
            pathfinder = sim.pathfinder


            agent_state = agent.get_state()
            previous_agent_state = agent_state
            subtask_start_position = agent_state.position
            for task_id, (task_type, task_index) in enumerate(episode["task_sequence"]):
                cur_task = episode_dict[task_type][task_index]
                instruction = get_sentence(task_type, cur_task, region_dict, goal_dict, concise_description=args.use_concise_description)
                target_ids = cur_task["target_object_ids"]
                cur_targets = [goal_dict[x] for x in target_ids]

                print(f"  task {task_id}: type={task_type}")
                print(f"  instruction: Find the {instruction}")

                step_count = 0
                traveled_distance = 0
                stop_trigger = False
                while step_count < 500:  # Limit each subgoal to 500 actions for fair comparison.
                    observations = sim.get_sensor_observations()
                    rgb = observations["color_sensor"][:, :, :3]
                    depth = observations["depth_sensor"]

                    #####################
                    # Use your model to predict actions, e.g. action_list, stop_trigger = model(rgb, depth, instruction)
                    action_list = ["turn_right", "turn_right", "move_forward"]
                    stop_trigger = True
                    #####################

                    for action in action_list:
                        observations = sim.step(action=action)
                        rgb = observations["color_sensor"][:, :, :3]
                        depth = observations["depth_sensor"]

                        agent_state = agent.get_state()
                        traveled_distance += np.linalg.norm(agent_state.position - previous_agent_state.position)
                        previous_agent_state = agent_state
                        step_count += 1  # Count both rotation and forward actions.

                    if stop_trigger:
                        break

                cur_agent_state = agent.get_state()
                tar_view_points = [
                    view_point["agent_state"]["position"] for tar in cur_targets for view_point in tar["view_points"]
                ]

                # Compute oracle geodesic distance.
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = subtask_start_position
                path.requested_ends = tar_view_points
                if pathfinder.find_path(path):
                    oracle_distance = path.geodesic_distance
                else:
                    oracle_distance = np.inf

                # Compute geodesic distance from the agent to the goal.
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = cur_agent_state.position
                path.requested_ends = tar_view_points
                if pathfinder.find_path(path):
                    goal_distance = path.geodesic_distance
                else:
                    goal_distance = np.inf

                sr = goal_distance <= 0.25  # Use 0.25 m for fair comparison.
                spl = 0 if not np.isfinite(oracle_distance) else sr * oracle_distance / max(oracle_distance, traveled_distance)

                episode_result["sequence"].append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "instruction": f"Find the {instruction}",
                    "sr": int(sr),
                    "spl": float(spl),
                    "step_count": step_count,
                })

                subtask_start_position = cur_agent_state.position

            # Save one JSON file for each multi-goal episode.
            output_path = os.path.join(args.result_dir, f"{scene_name}_sequence_{episode_id}.json")
            with open(output_path, "w") as f:
                json.dump(episode_result, f, indent=2)
            sim.close()

evaluate_multi_goal(args)
