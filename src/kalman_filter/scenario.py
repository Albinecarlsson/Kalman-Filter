from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RadarSettings:
    detection_range_km: float = 350.0
    measurement_noise_m: float = 75.0
    detection_probability: float = 0.98
    scan_period_s: float = 1.0
    coverage_angle_start_deg: float = 30.0
    coverage_angle_end_deg: float = 150.0
    radar_position_x_m: float = 0.0
    radar_position_y_m: float = 0.0

    @property
    def detection_range_m(self) -> float:
        return self.detection_range_km * 1000.0

    @property
    def coverage_angle_total_deg(self) -> float:
        return self.coverage_angle_end_deg - self.coverage_angle_start_deg


@dataclass(frozen=True)
class TargetSettings:
    max_targets: int = 20
    spawn_rate: float = 0.2
    min_velocity_ms: float = 150.0
    max_velocity_ms: float = 300.0
    arc_spawn_velocity_min_ms: float = 150.0
    arc_spawn_velocity_max_ms: float = 300.0
    side_spawn_velocity_min_ms: float = 150.0
    side_spawn_velocity_max_ms: float = 250.0
    front_spawn_velocity_min_ms: float = 100.0
    front_spawn_velocity_max_ms: float = 250.0
    min_acceleration_ms2: float = -0.5
    max_acceleration_ms2: float = 0.5
    spawn_distance_offset_km: float = 20.0
    spawn_locations: Tuple[str, ...] = ("arc", "left", "right", "front")
    spawn_weights: Tuple[float, ...] = (0.4, 0.2, 0.2, 0.2)
    out_of_bounds_km: float = 600.0

    @property
    def spawn_distance_offset_m(self) -> float:
        return self.spawn_distance_offset_km * 1000.0

    @property
    def out_of_bounds_m(self) -> float:
        return self.out_of_bounds_km * 1000.0


@dataclass(frozen=True)
class KalmanFilterSettings:
    acceleration_variance: float = 5.0
    initial_position_variance_m2: float = 1000.0
    initial_velocity_variance_m2s2: float = 100.0


@dataclass(frozen=True)
class TrackerSettings:
    max_track_age_s: float = 30.0
    min_hits_to_confirm: int = 3
    association_threshold_km: float = 15.0
    max_tracks_display: int = 10

    @property
    def association_threshold_m(self) -> float:
        return self.association_threshold_km * 1000.0


@dataclass(frozen=True)
class SimulationSettings:
    area_width_km: float = 500.0
    area_height_km: float = 500.0
    timestep_s: float = 1.0
    num_frames: int = 1000
    frame_interval_ms: int = 100
    random_seed: Optional[int] = None

    @property
    def area_width_m(self) -> float:
        return self.area_width_km * 1000.0

    @property
    def area_height_m(self) -> float:
        return self.area_height_km * 1000.0


@dataclass(frozen=True)
class VisualizationSettings:
    figure_width: int = 16
    figure_height: int = 8
    figure_dpi: int = 100
    plot_ylim_km: Tuple[float, float] = (-50.0, 332.5)
    range_rings_km: Tuple[float, ...] = (100.0, 200.0, 300.0)
    true_target_size: int = 80
    measurement_size: int = 60
    tracked_target_size: int = 120
    radar_center_size: int = 20
    true_target_color: str = "blue"
    true_target_marker: str = "o"
    true_target_alpha: float = 0.7
    measurement_color: str = "yellow"
    measurement_marker: str = "x"
    measurement_alpha: float = 0.8
    measurement_linewidth: int = 2
    tracked_target_color: str = "red"
    tracked_target_marker: str = "s"
    tracked_target_edgecolor: str = "darkred"
    tracked_target_linewidth: int = 2
    radar_color: str = "green"
    velocity_vector_color: str = "red"
    velocity_vector_linewidth: int = 2
    velocity_vector_alpha: float = 0.6
    velocity_vector_time_s: float = 60.0
    velocity_vector_min_speed_ms: float = 10.0
    radar_sweep_color: str = "lime"
    radar_sweep_alpha: float = 0.3
    radar_sweep_width_deg: float = 20.0
    radar_sweep_speed_deg_per_frame: float = 15.0
    illumination_color: str = "lime"
    illumination_alpha: float = 0.2
    grid_alpha: float = 0.3
    grid_linestyle: str = "--"
    legend_location: str = "upper left"
    legend_fontsize: int = 9
    legend_framealpha: float = 0.9
    stats_fontsize: int = 10
    stats_font_family: str = "monospace"

    def plot_xlim_km(self, detection_range_km: float) -> Tuple[float, float]:
        return (-detection_range_km * 0.9, detection_range_km * 0.9)

    def plot_ylim_km_for_range(self, detection_range_km: float) -> Tuple[float, float]:
        return (self.plot_ylim_km[0], detection_range_km * 0.95)


@dataclass(frozen=True)
class Scenario:
    name: str
    radar: RadarSettings
    target: TargetSettings
    kalman: KalmanFilterSettings
    tracker: TrackerSettings
    simulation: SimulationSettings
    visualization: VisualizationSettings


def _base_scenario() -> Scenario:
    return Scenario(
        name="default",
        radar=RadarSettings(),
        target=TargetSettings(),
        kalman=KalmanFilterSettings(),
        tracker=TrackerSettings(),
        simulation=SimulationSettings(),
        visualization=VisualizationSettings(),
    )


DEFAULT_SCENARIO = _base_scenario()


SCENARIOS: Dict[str, Scenario] = {
    "default": DEFAULT_SCENARIO,
    "high_traffic": replace(
        DEFAULT_SCENARIO,
        name="high_traffic",
        target=replace(DEFAULT_SCENARIO.target, max_targets=30, spawn_rate=0.3),
    ),
    "low_noise": replace(
        DEFAULT_SCENARIO,
        name="low_noise",
        radar=replace(DEFAULT_SCENARIO.radar, measurement_noise_m=25.0, detection_probability=0.99),
    ),
    "high_noise": replace(
        DEFAULT_SCENARIO,
        name="high_noise",
        radar=replace(DEFAULT_SCENARIO.radar, measurement_noise_m=150.0, detection_probability=0.9),
    ),
    "long_range": replace(
        DEFAULT_SCENARIO,
        name="long_range",
        radar=replace(DEFAULT_SCENARIO.radar, detection_range_km=450.0),
        visualization=replace(DEFAULT_SCENARIO.visualization, range_rings_km=(100.0, 200.0, 300.0, 400.0)),
    ),
    "fast_aircraft": replace(
        DEFAULT_SCENARIO,
        name="fast_aircraft",
        target=replace(DEFAULT_SCENARIO.target, min_velocity_ms=250.0, max_velocity_ms=600.0),
    ),
}


def get_scenario(name: str = "default") -> Scenario:
    key = (name or "default").strip().lower()
    if key not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS.keys()))
        raise ValueError(f"Unknown scenario '{name}'. Available: {valid}")
    return SCENARIOS[key]


def with_seed(scenario: Scenario, seed: Optional[int]) -> Scenario:
    if seed is None:
        return scenario
    return replace(scenario, simulation=replace(scenario.simulation, random_seed=seed))


def list_scenarios() -> List[str]:
    return sorted(SCENARIOS.keys())


def print_scenario(scenario: Scenario) -> None:
    print("\n" + "=" * 70)
    print(f"Scenario: {scenario.name}")
    print("=" * 70)

    print("\n[Radar]")
    print(f"  Detection Range: {scenario.radar.detection_range_km} km")
    print(f"  Measurement Noise: {scenario.radar.measurement_noise_m} m")
    print(f"  Detection Probability: {scenario.radar.detection_probability * 100:.1f}%")
    print(f"  Coverage Sector: {scenario.radar.coverage_angle_total_deg:.0f}°")

    print("\n[Target]")
    print(f"  Max Targets: {scenario.target.max_targets}")
    print(f"  Spawn Rate: {scenario.target.spawn_rate * 100:.1f}%")
    print(
        f"  Velocity Range: {scenario.target.min_velocity_ms * 3.6:.0f}-"
        f"{scenario.target.max_velocity_ms * 3.6:.0f} km/h"
    )

    print("\n[Kalman/Tracker]")
    print(f"  Acceleration Variance: {scenario.kalman.acceleration_variance}")
    print(f"  Max Track Age: {scenario.tracker.max_track_age_s} s")
    print(f"  Min Hits: {scenario.tracker.min_hits_to_confirm}")
    print(f"  Association Threshold: {scenario.tracker.association_threshold_km} km")

    print("\n[Simulation]")
    print(f"  Timestep: {scenario.simulation.timestep_s} s")
    print(f"  Frames: {scenario.simulation.num_frames}")
    print(f"  Frame Interval: {scenario.simulation.frame_interval_ms} ms")
    print(f"  Random Seed: {scenario.simulation.random_seed}")
    print("=" * 70 + "\n")
