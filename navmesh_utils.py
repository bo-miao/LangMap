"""Navmesh presets for LangMap evaluation.

After every scene load, the navmesh is rebuilt in memory from the scene GLB with the preset in
SIMULATOR.NAVMESH.PRESET, before episode start positions and shortest-path distances are computed.
Habitat still reads the .navmesh file in the scene folder, but it is replaced immediately, so the HM3D
scene folders can be used as downloaded.

  stretch  Default. Hello Robot Stretch (radius 0.17 m, height 1.41 m), as in the Habitat 2023 ObjectNav
           challenge and MTU3D; LangMap tasks were generated on this navmesh.
  legacy   Habitat default agent (radius 0.10 m, height 1.50 m).
"""
NAVMESH_PRESETS = {
    "stretch": dict(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.10, cell_height=0.05, agent_max_slope=45.0),
    "legacy": dict(agent_radius=0.10, agent_height=1.50, agent_max_climb=0.20, cell_height=0.20, agent_max_slope=45.0),
}


def rebuild_navmesh(sim, preset):
    """Rebuild sim.pathfinder of a loaded habitat_sim.Simulator with the given preset."""
    import habitat_sim
    if preset not in NAVMESH_PRESETS:
        raise ValueError(f"unknown navmesh preset {preset!r}; choose from {sorted(NAVMESH_PRESETS)}")
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for key, value in NAVMESH_PRESETS[preset].items():
        setattr(settings, key, value)
    if not sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True) or not sim.pathfinder.is_loaded:
        raise RuntimeError(f"navmesh rebuild failed for preset {preset!r}")


def nearest_goal_distance(sim, position, goals):
    """Geodesic distance on the navmesh from position to the nearest goal viewpoint; inf if none is reachable."""
    dists = [sim.geodesic_distance(position, [goal.position]) for goal in goals]
    dists = [d for d in dists if d == d and d != float("inf")]  # drop NaN and inf
    return min(dists) if dists else float("inf")


def install_habitat_lab_navmesh():
    """Make habitat-lab rebuild the navmesh whenever HabitatSim loads a scene."""
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    original_init, original_reconfigure = HabitatSim.__init__, HabitatSim.reconfigure

    def ensure(sim):
        if getattr(sim, "_navmesh_scene", None) != sim._current_scene:  # reconfiguring the same scene keeps the mesh
            rebuild_navmesh(sim, sim.habitat_config.NAVMESH.PRESET)
            sim._navmesh_scene = sim._current_scene

    def __init__(self, config):
        original_init(self, config)
        ensure(self)

    def reconfigure(self, habitat_config):
        original_reconfigure(self, habitat_config)
        ensure(self)

    HabitatSim.__init__, HabitatSim.reconfigure = __init__, reconfigure
