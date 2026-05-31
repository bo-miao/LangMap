from habitat_sim import Simulator as Sim
import habitat_sim


def make_sim_cfg(settings, agent_settings):
    # sim cfg
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.default_agent_id = settings["default_agent"]
    sim_cfg.allow_sliding = settings["allow_sliding"]
    sim_cfg.random_seed = settings["seed"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    # sim_cfg.load_semantic_mesh = True
    # agent cfg
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space['turn_left'] = habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=agent_settings["turn_angle"]))
    agent_cfg.action_space['turn_right'] = habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=agent_settings["turn_angle"]))
    agent_cfg.action_space['move_forward'] = habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=agent_settings['step_size']))
    agent_cfg.action_space['look_up'] = habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=agent_settings["tilt_angle"]))
    agent_cfg.action_space['look_down'] = habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=agent_settings["tilt_angle"]))
    agent_cfg.height = agent_settings["height"]
    agent_cfg.radius = agent_settings["radius"]
    sim_sensors = []
    # build rgb sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [agent_settings['rgb_sensor']["height"], agent_settings['rgb_sensor']["width"]]
    rgb_sensor_spec.position = agent_settings['rgb_sensor']["position"]
    rgb_sensor_spec.hfov = agent_settings['rgb_sensor']["hfov"]
    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [agent_settings['depth_sensor']["height"], agent_settings['depth_sensor']["width"]]
    depth_sensor_spec.position = agent_settings['depth_sensor']["position"]
    depth_sensor_spec.hfov = agent_settings['depth_sensor']["hfov"]
    # Semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [agent_settings['rgb_sensor']["height"], agent_settings['rgb_sensor']["width"]]
    semantic_sensor_spec.position = agent_settings['rgb_sensor']["position"]
    semantic_sensor_spec.hfov = agent_settings['rgb_sensor']["hfov"]
    # add sensors
    sim_sensors.append(rgb_sensor_spec)
    sim_sensors.append(depth_sensor_spec)
    sim_sensors.append(semantic_sensor_spec)
    agent_cfg.sensor_specifications = sim_sensors
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_nav_mesh_cfg(settings):
    nav_mesh_cfg = habitat_sim.NavMeshSettings()
    nav_mesh_cfg.set_defaults()
    nav_mesh_cfg.agent_height = settings["agent_height"]
    nav_mesh_cfg.agent_radius = settings["agent_radius"]
    nav_mesh_cfg.agent_max_climb = settings["agent_max_climb"]
    nav_mesh_cfg.cell_height = settings["cell_height"]
    return nav_mesh_cfg

# current, we only support one agent
class HabitatSimulator:
    def __init__(self, sim_setting, agent_setting) -> None:
        # initialize environment
        self.config = make_sim_cfg(sim_setting, agent_setting)
        self.simulator = Sim(self.config)
        # recompute navigation mesh
        self.nav_mesh_config = make_nav_mesh_cfg(sim_setting)
        self.simulator.recompute_navmesh(self.simulator.pathfinder, self.nav_mesh_config)   
        # get agent
        self.agent = self.simulator.initialize_agent(sim_setting["default_agent"])

    