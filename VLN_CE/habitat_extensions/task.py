import gzip
import json
import os
from typing import Dict, List, Optional, Union
from tqdm import tqdm

import attr
# from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"
black_task_ids_ = \
   ['00800-TEEsavR23oF_room_74', '00810-CrMo8WxCyVb_object_30', '00810-CrMo8WxCyVb_object_31', '00810-CrMo8WxCyVb_region_75', '00810-CrMo8WxCyVb_region_76', '00810-CrMo8WxCyVb_room_53', '00810-CrMo8WxCyVb_room_54', '00820-mL8ThkuaVTM_object_29', '00820-mL8ThkuaVTM_region_47', '00820-mL8ThkuaVTM_room_39', '00823-7MXmsvcQjpJ_room_100', '00823-7MXmsvcQjpJ_room_101', '00823-7MXmsvcQjpJ_room_74', '00823-7MXmsvcQjpJ_room_98', '00823-7MXmsvcQjpJ_room_99', '00829-QaLdnwvtxbs_room_14', '00829-QaLdnwvtxbs_room_16', '00829-QaLdnwvtxbs_room_21', '00829-QaLdnwvtxbs_room_22', '00829-QaLdnwvtxbs_room_23', '00829-QaLdnwvtxbs_room_24', '00829-QaLdnwvtxbs_room_25', '00829-QaLdnwvtxbs_room_5', '00829-QaLdnwvtxbs_room_7', '00829-QaLdnwvtxbs_room_9', '00832-qyAac8rV8Zk_region_18', '00832-qyAac8rV8Zk_room_18', '00839-zt1RVoi7PcG_object_25', '00839-zt1RVoi7PcG_region_70', '00839-zt1RVoi7PcG_room_54', '00862-LT9Jq6dN3Ea_object_46', '00862-LT9Jq6dN3Ea_region_135', '00862-LT9Jq6dN3Ea_region_37', '00862-LT9Jq6dN3Ea_room_21', '00862-LT9Jq6dN3Ea_room_93', '00871-VBzV5z6i1WS_object_21', '00871-VBzV5z6i1WS_object_28', '00871-VBzV5z6i1WS_object_29', '00871-VBzV5z6i1WS_object_38', '00871-VBzV5z6i1WS_object_8', '00871-VBzV5z6i1WS_region_32', '00871-VBzV5z6i1WS_region_58', '00871-VBzV5z6i1WS_region_67', '00871-VBzV5z6i1WS_region_68', '00871-VBzV5z6i1WS_region_77', '00871-VBzV5z6i1WS_room_21', '00871-VBzV5z6i1WS_room_39', '00871-VBzV5z6i1WS_room_46', '00871-VBzV5z6i1WS_room_47', '00871-VBzV5z6i1WS_room_56', '00873-bxsVRursffK_object_29', '00873-bxsVRursffK_region_52', '00873-bxsVRursffK_room_39', '00876-mv2HUxq3B53_object_16', '00876-mv2HUxq3B53_object_31', '00876-mv2HUxq3B53_object_40', '00876-mv2HUxq3B53_region_33', '00876-mv2HUxq3B53_region_51', '00876-mv2HUxq3B53_region_69', '00876-mv2HUxq3B53_region_81', '00876-mv2HUxq3B53_region_92', '00876-mv2HUxq3B53_region_93', '00876-mv2HUxq3B53_room_19', '00876-mv2HUxq3B53_room_31', '00876-mv2HUxq3B53_room_45', '00876-mv2HUxq3B53_room_50', '00876-mv2HUxq3B53_room_59', '00877-4ok3usBNeis_object_44', '00877-4ok3usBNeis_region_58', '00877-4ok3usBNeis_region_8', '00877-4ok3usBNeis_room_52', '00877-4ok3usBNeis_room_9', '00878-XB4GS9ShBRE_object_18', '00878-XB4GS9ShBRE_region_33', '00878-XB4GS9ShBRE_room_32', '00880-Nfvxx8J5NCo_object_13', '00880-Nfvxx8J5NCo_object_30', '00880-Nfvxx8J5NCo_object_31', '00880-Nfvxx8J5NCo_region_22', '00880-Nfvxx8J5NCo_region_45', '00880-Nfvxx8J5NCo_region_46', '00880-Nfvxx8J5NCo_room_23', '00880-Nfvxx8J5NCo_room_44', '00880-Nfvxx8J5NCo_room_45', '00890-6s7QHgap2fW_region_19', '00890-6s7QHgap2fW_room_16', '00891-cvZr5TUy5C5_object_12', '00891-cvZr5TUy5C5_object_43', '00891-cvZr5TUy5C5_object_44', '00891-cvZr5TUy5C5_region_109', '00891-cvZr5TUy5C5_region_110', '00891-cvZr5TUy5C5_region_111', '00891-cvZr5TUy5C5_region_112', '00891-cvZr5TUy5C5_region_39', '00891-cvZr5TUy5C5_region_40', '00891-cvZr5TUy5C5_region_44', '00891-cvZr5TUy5C5_room_34', '00891-cvZr5TUy5C5_room_35', '00891-cvZr5TUy5C5_room_36', '00891-cvZr5TUy5C5_room_92', '00891-cvZr5TUy5C5_room_93', '00891-cvZr5TUy5C5_room_94', '00891-cvZr5TUy5C5_room_95', '00891-cvZr5TUy5C5_room_96', '00891-cvZr5TUy5C5_room_97', '00894-HY1NcmCgn3n_instance_7']

@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    """Loads the R2R VLN-CE dataset"""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            # cast integer IDs to strings
            episode["episode_id"] = str(episode["episode_id"])
            episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @staticmethod
    def check_config_paths_exist(config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.DATA_PATH.format(split=config.SPLIT, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.SPLIT
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config) -> bool:
        return all(
            os.path.exists(
                config.DATA_PATH.format(split=config.SPLIT, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        """Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language


@registry.register_dataset(name="LangMap-VLN-v1")
class LangMapDatasetV1(Dataset):
    """Loads the LangMap Object-Goal Navigation Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    navigation_types: List[str] = ["object", "room", "region", "instance"]
    black_task_ids = black_task_ids_

    def __init__(self, config = None) -> None:
        self.episodes = []
        self.config = config
        if config is None:
            return

        dataset_files = sorted([f for f in os.listdir(config.DATA_PATH) if f.endswith(".json.gz")])
        if config.CHECK == 1:
            dataset_files = ['00800-TEEsavR23oF.json.gz']

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            dataset_files = [
                f for f in dataset_files 
                if f.split(".")[0] in scenes_to_load
            ]

        # Filter by navigation types
        if hasattr(config, 'NAVIGATION_TYPES') and ALL_EPISODES_MASK not in config.NAVIGATION_TYPES:
            self.nav_types_to_load = set(config.NAVIGATION_TYPES)
        else:
            self.nav_types_to_load = ["object", "room", "region", "instance"]

        # load json
        for scene_file in dataset_files:
            scene_path = os.path.join(config.DATA_PATH, scene_file)
            with gzip.open(scene_path, 'rt', encoding='utf-8') as f:
                scene_data = json.load(f)
                self.from_json(
                    scene_data,
                    scenes_dir=config.SCENES_DIR,
                    scene_name=scene_file.split(".")[0]
                )

        self.episodes = [
            episode
            for episode in self.episodes if episode.info['navigation_type'] in self.nav_types_to_load
        ]

        # Create a simple instruction vocabulary
        self.instruction_vocab = VocabDict()

    def from_json(
        self, scene_data: dict, scenes_dir: Optional[str] = None, scene_name: str = ""
    ) -> None:
        # Get all goals dictionary
        all_goals_dict = {g['object_id']: g for g in scene_data["goals"]}
        region_dict = scene_data['region_annotation']
        # Combine all episode types
        all_episodes = []
        for nav_type in ["instance", "region", "room", "object"]:
            key = f"episodes_by_{nav_type}_level"
            if key in scene_data:
                for ep in scene_data[key]:
                    ep["navigation_type"] = nav_type
                    all_episodes.append(ep)

        # Convert each episode to VLN-CE format
        black_task_ids = self.black_task_ids
        for episode_data in all_episodes:
            episode_id = f"{scene_name}_{episode_data['navigation_type']}_{episode_data['episode_id']}"
            if episode_id in black_task_ids:
                continue

            # Generate instruction text based on navigation type
            nav_type = episode_data["navigation_type"]
            if nav_type in ["region", "room", "object"]:
                obj_category = episode_data['object_category']
            else:
                first_target_id = episode_data["target_object_ids"][0]
                obj_category = all_goals_dict[first_target_id].get("object_category", "object")

            if nav_type == "object":
                instruction_text = f"Find the {obj_category}."
            elif nav_type == "room":
                room_name = episode_data.get("room_name", "room")
                instruction_text = f"Find the {obj_category} in the {room_name}"
                if 'tie:' in room_name.lower():
                    instruction_text += ", 'Tie: A & B' means the area spans both rooms A and B"
            elif nav_type == "region":
                region_id = episode_data.get("region_id", "")
                if region_id in region_dict:
                    region_info = region_dict[region_id]
                    category = region_info.get('region_category', 'region').lower()
                    desc = region_info.get('concise_description', '') if len(self.nav_types_to_load) > 2 \
                        else region_info.get('detailed_description', '')
                    instruction_text = f"Find the {obj_category} in the {category} that has {desc}"
                    if 'tie:' in category.lower() or 'tie:' in desc.lower():
                        instruction_text += ", 'Tie: A & B' means the area spans both rooms A and B"
                else:
                    print("Error: cannot find region description.")
                    instruction_text = f"Find the {obj_category} in the target region"
            elif nav_type == "instance":
                inst_id = episode_data.get("instance_id", "")
                if inst_id in all_goals_dict:
                    inst_desc = all_goals_dict[inst_id].get('annot_unique_concise_description', 'object') if len(self.nav_types_to_load) > 2 \
                        else all_goals_dict[inst_id].get('annot_unique_detailed_description', 'object')
                    instruction_text = f"Find the {inst_desc}"
                else:
                    print("Error: cannot find instance description")
                    instruction_text = "Find the target instance"
            
            goals = []
            for obj_id in episode_data.get("target_object_ids", []):
                if obj_id in all_goals_dict:
                    goal_info = all_goals_dict[obj_id]
                    viewpoints = goal_info.get("view_points", [])
                    if viewpoints:
                        for vp in viewpoints:
                            agent_state = vp.get("agent_state", {})
                            vp_position = agent_state.get("position")
                            if vp_position:
                                goals.append(NavigationGoal(
                                    position=vp_position, radius=1.0
                                ))
                    else:
                        goals.append(NavigationGoal(
                            position=goal_info.get("position", [0, 0, 0]), radius=1.0
                        ))
            
            if "-" in scene_name:
                _, glb_name = scene_name.split("-", 1)
            else:
                glb_name = scene_name
            
            scene_id = f"{glb_name}.basis.glb"
            if scenes_dir is not None:
                scene_id = os.path.join(scenes_dir, scene_name, scene_id)
            
            # Skip episodes with no goals
            if not goals:
                continue
            
            # Validate episode data
            import math
            start_pos = episode_data.get("start_position", [0, 0, 0])
            start_rot = episode_data.get("start_rotation", [0, 0, 0, 1])
            
            # Check for NaN/Inf in positions
            if any(math.isnan(x) or math.isinf(x) for x in start_pos):
                print(f"Warning: Skipping episode {episode_id} - invalid start position: {start_pos}")
                continue
            
            if any(math.isnan(x) or math.isinf(x) for x in start_rot):
                print(f"Warning: Skipping episode {episode_id} - invalid start rotation: {start_rot}")
                continue
            
            # Check if rotation is a valid quaternion (length should be ~1)
            rot_magnitude = sum(x*x for x in start_rot) ** 0.5
            if abs(rot_magnitude - 1.0) > 0.1:  # Allow some tolerance
                print(f"Warning: Skipping episode {episode_id} - invalid rotation magnitude: {rot_magnitude}")
                continue
            
            # Create VLN episode
            episode = VLNExtendedEpisode(
                episode_id=episode_id, # task id
                scene_id=scene_id,
                start_position=episode_data.get("start_position", [0, 0, 0]),
                start_rotation=episode_data.get("start_rotation", [0, 0, 0, 1]),
                goals=goals,
                instruction=ExtendedInstructionData(
                    instruction_text=instruction_text,
                    instruction_id=episode_id,
                    language="en-US"
                ),
                trajectory_id=episode_id,
                info={
                    "navigation_type": nav_type,
                    "scene_name": scene_name,
                    "original_episode_id": episode_data["episode_id"],
                    "target_object_ids": episode_data["target_object_ids"]
                }
            )
            
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @classmethod
    def check_config_paths_exist(cls, config) -> bool:
        return os.path.exists(config.DATA_PATH) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def scene_from_scene_path(scene_path: str) -> str:
        """Extract scene name from scene path."""
        return os.path.splitext(os.path.basename(scene_path))[0]


@registry.register_dataset(name="LangMap-VLN-Seq-v1")
class LangMapDatasetV1Seq(Dataset):
    """Loads the LangMap Sequence Navigation Dataset (episodes with multiple subtasks)."""
    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        dataset_files = sorted([
            f for f in os.listdir(config.DATA_PATH) if f.endswith(".json.gz")
        ])
        if config.CHECK == 1:
            dataset_files = ['00800-TEEsavR23oF.json.gz']

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            dataset_files = [
                f for f in dataset_files
                if f.split(".")[0] in scenes_to_load
            ]

        for scene_file in dataset_files:
            scene_path = os.path.join(config.DATA_PATH, scene_file)
            with gzip.open(scene_path, 'rt', encoding='utf-8') as f:
                scene_data = json.load(f)
                self.from_json(
                    scene_data,
                    scenes_dir=config.SCENES_DIR,
                    scene_name=scene_file.split(".")[0]
                )

        self.instruction_vocab = VocabDict()

    def from_json(self, scene_data: dict, scenes_dir: Optional[str] = None, scene_name: str = "") -> None:
        # Get sequence episodes
        sequence_episodes = scene_data.get("episode_by_sequence", [])
        if not sequence_episodes:
            print(f"Warning: No sequence episodes found in scene {scene_name}")
            return

        # Process each sequence episode
        for seq_episode_data in tqdm(sequence_episodes):
            episode_id = seq_episode_data.get("episode_id")
            navigation_type = seq_episode_data.get("navigation_type", "sequence")
            episode_unique_id = f"{scene_name}_{navigation_type}_{episode_id}"

            # Get task sequence (list of [nav_type, index] tuples)
            task_sequence = seq_episode_data.get("task_sequence", [])
            if not task_sequence:
                print(f"Warning: Sequence episode {episode_unique_id} has no task_sequence, skipping")
                continue

            # Build scene path from scene name
            if "-" in scene_name:
                _, glb_name = scene_name.split("-", 1)
            else:
                glb_name = scene_name

            scene_id = f"{glb_name}.basis.glb"
            if scenes_dir is not None:
                scene_id = os.path.join(scenes_dir, scene_name, scene_id)

            import math
            start_pos = seq_episode_data.get("start_position", [0, 0, 0])
            start_rot = seq_episode_data.get("start_rotation", [0, 0, 0, 1])

            if any(math.isnan(x) or math.isinf(x) for x in start_pos):
                print(f"Warning: Skipping sequence episode {episode_unique_id} - invalid start position: {start_pos}")
                continue

            if any(math.isnan(x) or math.isinf(x) for x in start_rot):
                print(f"Warning: Skipping sequence episode {episode_unique_id} - invalid start rotation: {start_rot}")
                continue

            # Check if rotation is a valid quaternion
            rot_magnitude = sum(x * x for x in start_rot) ** 0.5
            if abs(rot_magnitude - 1.0) > 0.1:
                print(f"Warning: Skipping sequence episode {episode_unique_id} - invalid rotation magnitude: {rot_magnitude}")
                continue

            # The actual instruction for each subtask will be generated during evaluation
            instruction_text = f"Complete sequence of {len(task_sequence)} navigation tasks"

            # Store sequence metadata in episode info
            episode = VLNExtendedEpisode(
                episode_id=episode_unique_id,
                scene_id=scene_id,
                start_position=start_pos,
                start_rotation=start_rot,
                goals=[],  # Goals will be set per subtask during evaluation
                instruction=ExtendedInstructionData(
                    instruction_text=instruction_text,
                    instruction_id=episode_unique_id,
                    language="en-US"
                ),
                trajectory_id=episode_unique_id,
                info={
                    "navigation_type": navigation_type,
                    "scene_name": scene_name,
                    "original_episode_id": episode_id,
                    "task_sequence": task_sequence,
                    "is_sequence": True,
                }
            )

            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @classmethod
    def check_config_paths_exist(cls, config) -> bool:
        return os.path.exists(config.DATA_PATH) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def scene_from_scene_path(scene_path: str) -> str:
        """Extract scene name from scene path."""
        return os.path.splitext(os.path.basename(scene_path))[0]