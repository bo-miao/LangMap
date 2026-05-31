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
parser.add_argument("--result_dir", default="langmap_single_goal_results")
parser.add_argument("--use_concise_description", type=int, default=1)
args = parser.parse_args()

black_task_ids = \
   ['00800-TEEsavR23oF_room_74', '00810-CrMo8WxCyVb_object_30', '00810-CrMo8WxCyVb_object_31', '00810-CrMo8WxCyVb_region_75', '00810-CrMo8WxCyVb_region_76', '00810-CrMo8WxCyVb_room_53', '00810-CrMo8WxCyVb_room_54', '00820-mL8ThkuaVTM_object_29', '00820-mL8ThkuaVTM_region_47', '00820-mL8ThkuaVTM_room_39', '00823-7MXmsvcQjpJ_room_100', '00823-7MXmsvcQjpJ_room_101', '00823-7MXmsvcQjpJ_room_74', '00823-7MXmsvcQjpJ_room_98', '00823-7MXmsvcQjpJ_room_99', '00829-QaLdnwvtxbs_room_14', '00829-QaLdnwvtxbs_room_16', '00829-QaLdnwvtxbs_room_21', '00829-QaLdnwvtxbs_room_22', '00829-QaLdnwvtxbs_room_23', '00829-QaLdnwvtxbs_room_24', '00829-QaLdnwvtxbs_room_25', '00829-QaLdnwvtxbs_room_5', '00829-QaLdnwvtxbs_room_7', '00829-QaLdnwvtxbs_room_9', '00832-qyAac8rV8Zk_region_18', '00832-qyAac8rV8Zk_room_18', '00839-zt1RVoi7PcG_object_25', '00839-zt1RVoi7PcG_region_70', '00839-zt1RVoi7PcG_room_54', '00862-LT9Jq6dN3Ea_object_46', '00862-LT9Jq6dN3Ea_region_135', '00862-LT9Jq6dN3Ea_region_37', '00862-LT9Jq6dN3Ea_room_21', '00862-LT9Jq6dN3Ea_room_93', '00871-VBzV5z6i1WS_object_21', '00871-VBzV5z6i1WS_object_28', '00871-VBzV5z6i1WS_object_29', '00871-VBzV5z6i1WS_object_38', '00871-VBzV5z6i1WS_object_8', '00871-VBzV5z6i1WS_region_32', '00871-VBzV5z6i1WS_region_58', '00871-VBzV5z6i1WS_region_67', '00871-VBzV5z6i1WS_region_68', '00871-VBzV5z6i1WS_region_77', '00871-VBzV5z6i1WS_room_21', '00871-VBzV5z6i1WS_room_39', '00871-VBzV5z6i1WS_room_46', '00871-VBzV5z6i1WS_room_47', '00871-VBzV5z6i1WS_room_56', '00873-bxsVRursffK_object_29', '00873-bxsVRursffK_region_52', '00873-bxsVRursffK_room_39', '00876-mv2HUxq3B53_object_16', '00876-mv2HUxq3B53_object_31', '00876-mv2HUxq3B53_object_40', '00876-mv2HUxq3B53_region_33', '00876-mv2HUxq3B53_region_51', '00876-mv2HUxq3B53_region_69', '00876-mv2HUxq3B53_region_81', '00876-mv2HUxq3B53_region_92', '00876-mv2HUxq3B53_region_93', '00876-mv2HUxq3B53_room_19', '00876-mv2HUxq3B53_room_31', '00876-mv2HUxq3B53_room_45', '00876-mv2HUxq3B53_room_50', '00876-mv2HUxq3B53_room_59', '00877-4ok3usBNeis_object_44', '00877-4ok3usBNeis_region_58', '00877-4ok3usBNeis_region_8', '00877-4ok3usBNeis_room_52', '00877-4ok3usBNeis_room_9', '00878-XB4GS9ShBRE_object_18', '00878-XB4GS9ShBRE_region_33', '00878-XB4GS9ShBRE_room_32', '00880-Nfvxx8J5NCo_object_13', '00880-Nfvxx8J5NCo_object_30', '00880-Nfvxx8J5NCo_object_31', '00880-Nfvxx8J5NCo_region_22', '00880-Nfvxx8J5NCo_region_45', '00880-Nfvxx8J5NCo_region_46', '00880-Nfvxx8J5NCo_room_23', '00880-Nfvxx8J5NCo_room_44', '00880-Nfvxx8J5NCo_room_45', '00890-6s7QHgap2fW_region_19', '00890-6s7QHgap2fW_room_16', '00891-cvZr5TUy5C5_object_12', '00891-cvZr5TUy5C5_object_43', '00891-cvZr5TUy5C5_object_44', '00891-cvZr5TUy5C5_region_109', '00891-cvZr5TUy5C5_region_110', '00891-cvZr5TUy5C5_region_111', '00891-cvZr5TUy5C5_region_112', '00891-cvZr5TUy5C5_region_39', '00891-cvZr5TUy5C5_region_40', '00891-cvZr5TUy5C5_region_44', '00891-cvZr5TUy5C5_room_34', '00891-cvZr5TUy5C5_room_35', '00891-cvZr5TUy5C5_room_36', '00891-cvZr5TUy5C5_room_92', '00891-cvZr5TUy5C5_room_93', '00891-cvZr5TUy5C5_room_94', '00891-cvZr5TUy5C5_room_95', '00891-cvZr5TUy5C5_room_96', '00891-cvZr5TUy5C5_room_97', '00894-HY1NcmCgn3n_instance_7']

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


def evaluate_single_goal(args):
    os.makedirs(args.result_dir, exist_ok=True)
    scene_files = sorted([x for x in os.listdir(args.langmap_path) if x.endswith(".json.gz")])
    print(f"\nTotal {len(scene_files)} selected scenes: {scene_files}")

    for scene_file in tqdm(scene_files):
        scene_name = scene_file.replace(".json.gz", "")
        scene_data = load_json(os.path.join(args.langmap_path, scene_file))

        region_dict = scene_data["region_annotation"]
        goal_dict = {x["object_id"]: x for x in scene_data["goals"]}
        episode_dict = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }

        for task_type in ["object", "room", "region", "instance"]:
            for cur_task in episode_dict[task_type]:
                episode_id = cur_task["episode_id"]
                task_id = f"{scene_name}_{task_type}_{episode_id}"
                if task_id in black_task_ids:
                    print(f"{task_id} hit blacklist, skipped")
                    continue

                start_position = cur_task["start_position"]
                start_rotation = cur_task["start_rotation"]
                instruction = get_sentence(task_type, cur_task, region_dict, goal_dict, args.use_concise_description)
                target_ids = cur_task["target_object_ids"]
                cur_targets = [goal_dict[x] for x in target_ids]

                print(f"\n{task_id}")
                print(f"  instruction: Find the {instruction}")

                sim_settings = OmegaConf.load("habitat_config/langmap_sim_config.yaml")
                agent_setting = OmegaConf.load("habitat_config/langmap_agent_config.yaml")
                sim_settings["scene"] = os.path.join(args.scene_path, scene_name, f"{scene_name.split('-')[-1]}.basis.glb")
                abstract_sim = HabitatSimulator(sim_settings, agent_setting)
                sim = abstract_sim.simulator
                agent = abstract_sim.agent
                agent_state = habitat_sim.AgentState()
                agent_state.position = start_position
                agent_state.rotation = start_rotation
                agent.set_state(agent_state)
                pathfinder = sim.pathfinder

                step_count = 0
                traveled_distance = 0
                previous_agent_state = agent.get_state()
                start_agent_state = agent.get_state()

                while step_count < 500:
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
                        step_count += 1

                    if stop_trigger:
                        break

                cur_agent_state = agent.get_state()
                view_points = [
                    view_point["agent_state"]["position"] for tar in cur_targets for view_point in tar["view_points"]
                ]

                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = start_agent_state.position
                path.requested_ends = view_points
                if pathfinder.find_path(path):
                    oracle_distance = path.geodesic_distance
                else:
                    oracle_distance = np.inf

                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = cur_agent_state.position
                path.requested_ends = view_points
                if pathfinder.find_path(path):
                    goal_distance = path.geodesic_distance
                else:
                    goal_distance = np.inf

                sr = goal_distance <= 0.25
                spl = 0 if not np.isfinite(oracle_distance) else sr * oracle_distance / max(oracle_distance, traveled_distance)

                result = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "instruction": f"Find the {instruction}",
                    "sr": int(sr),
                    "spl": float(spl),
                    "step_count": step_count,
                }

                output_path = os.path.join(args.result_dir, f"{scene_name}_{task_type}_{episode_id}.json")
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                sim.close()


evaluate_single_goal(args)
