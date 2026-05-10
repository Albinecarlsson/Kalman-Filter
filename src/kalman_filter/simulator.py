import random as rnd
import numpy as np
from typing import List, Tuple, Optional
from .kf import KF

from .scenario import DEFAULT_SCENARIO, Scenario


class Target:
    """Represents a moving target in the simulation."""

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 ax: float = 0, ay: float = 0, target_id: int = 0) -> None:
        """
        Initialize a target.

        Args:
            x, y: Initial position
            vx, vy: Initial velocity
            ax, ay: Acceleration
            target_id: Unique identifier
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.acceleration = np.array([ax, ay], dtype=float)
        self.id = target_id

    def move(self, dt: float) -> None:
        """Update position and velocity based on acceleration and time step."""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def get_position(self) -> np.ndarray:
        """Return current position."""
        return self.position.copy()


class Radar:
    """Simulates a radar system that detects targets with measurement noise."""

    def __init__(self, range_limit: float, measurement_noise: float, 
                 detection_probability: float = 1.0, scan_period: float = 0.1,
                 rng: Optional[np.random.Generator] = None) -> None:
        """
        Initialize the radar.
        Args:
            range_limit: Maximum detection range
            measurement_noise: Standard deviation of measurement noise
            detection_probability: Probability of detecting a target (0-1)
            scan_period: Time between radar scans
        """
        self.range_limit = range_limit
        self.measurement_noise = measurement_noise
        self.detection_probability = detection_probability
        self.scan_period = scan_period
        self.center = np.array([0.0, 0.0])
        self.rng = rng if rng is not None else np.random.default_rng()

    def detect(self, targets: List[Target]) -> List[Tuple[int, np.ndarray]]:
        """
        Detect targets within range with noise.

        Args:
            Targets: List of Target objects

        Returns:
            List of (target_id, measurement) tuples for detected targets
        """
        detections = []

        for target in targets:
            # Check if target is within range
            distance = np.linalg.norm(target.position - self.center)

            if distance <= self.range_limit:
                # Check detection probability
                if self.rng.random() < self.detection_probability:
                    # Add measurement noise
                    noise = self.rng.normal(0.0, self.measurement_noise, size=2)
                    measurement = target.position + noise
                    detections.append((target.id, measurement))

        return detections


class TrackedObject:
    """Represents a tracked object with an associated Kalman filter."""

    def __init__(self, obj_id: int, kf: KF, last_update_time: float) -> None:
        """
        Initialize a tracked object.

        Args:
            obj_id: Unique identifier
            kf: Kalman Filter instance
            last_update_time: Time of last update
        """
        self.id = obj_id
        self.kf = kf
        self.last_update_time = last_update_time
        self.time_since_update = 0.0
        self.consecutive_misses = 0
        self.measurements_received = 0

    def predict(self, dt: float) -> None:
        """Predict next state."""
        self.kf.prediction(dt)
        self.time_since_update += dt
        self.consecutive_misses += 1

    def update_with_measurement(self, measurement: np.ndarray,
                               measurement_variance: float, current_time: float) -> None:
        """Update with a measurement."""
        self.kf.update(measurement, measurement_variance)
        self.last_update_time = current_time
        self.time_since_update = 0.0
        self.consecutive_misses = 0
        self.measurements_received += 1

    def get_estimated_position(self) -> np.ndarray:
        """Get estimated position [x, y]."""
        return self.kf.position

    def get_estimated_velocity(self) -> np.ndarray:
        """Get estimated velocity [vx, vy]."""
        return self.kf.velocity


class RadarTracker:
    """Tracks multiple objects using Kalman filters and radar detections."""

    def __init__(self, radar: Radar, acceleration_variance: float = 0.1,
                 max_age: float = 2.0, min_hits: int = 3, 
                 association_threshold: float = 50.0) -> None:
        """
        Initialize the tracker.

        Args:
            radar: Radar instance
            acceleration_variance: Process noise variance for Kalman filters
            max_age: Maximum time to keep track of undetected object (seconds)
            min_hits: Minimum detections before confirming track
            association_threshold: Maximum distance for measurement-to-track association
        """
        self.radar = radar
        self.acceleration_variance = acceleration_variance
        self.max_age = max_age
        self.min_hits = min_hits
        self.association_threshold = association_threshold
        self.tracks: List[TrackedObject] = []
        self.next_id = 0
        self.current_time = 0.0

    def predict(self, dt: float) -> None:
        """Predict all tracks."""
        self.current_time += dt
        for track in self.tracks:
            track.predict(dt)

    def update(self, measurements: List[Tuple[int, np.ndarray]], 
               measurement_variance: float) -> None:
        """
        Update tracks with measurements using nearest neighbor association.

        Args:
            measurements: List of (target_id, measurement) tuples
            measurement_variance: Measurement noise variance
        """
        # Convert measurements to list and track associations
        unassociated_measurements = list(range(len(measurements)))

        for track in self.tracks:
            best_distance = self.association_threshold
            best_idx = -1

            # Find closest measurement
            for i, (_, measurement) in enumerate(measurements):
                if i not in unassociated_measurements:
                    continue

                predicted_pos = track.get_estimated_position()
                distance = np.linalg.norm(measurement - predicted_pos)

                if distance < best_distance:
                    best_distance = distance
                    best_idx = i

            # Update track if measurement found
            if best_idx >= 0:
                _, measurement = measurements[best_idx]
                track.update_with_measurement(measurement, measurement_variance, 
                                             self.current_time)
                unassociated_measurements.remove(best_idx)

        # Create new tracks for unassociated measurements
        for idx in unassociated_measurements:
            _, measurement = measurements[idx]
            self._create_track(measurement)

    def _create_track(self, initial_measurement: np.ndarray) -> None:
        """Create a new track from an initial measurement."""
        initial_x = float(initial_measurement[0])
        initial_y = float(initial_measurement[1])
        kf = KF(initial_x, 0.0,
            initial_y, 0.0,
                self.acceleration_variance)
        track = TrackedObject(self.next_id, kf, self.current_time)
        track.update_with_measurement(initial_measurement, self.radar.measurement_noise, 
                                     self.current_time)
        self.tracks.append(track)
        self.next_id += 1

    def cleanup_old_tracks(self) -> None:
        """Remove tracks that haven't been updated for too long."""
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get tracks with sufficient measurements."""
        return [t for t in self.tracks if t.measurements_received >= self.min_hits]

    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all active tracks."""
        return self.tracks.copy()


class RadarSimulationArea:
    """Simulates a radar coverage area with targets."""

    def __init__(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        scenario: Optional[Scenario] = None,
    ) -> None:
        """
        Initialize simulation area.

        Args:
            width: Width of area (meters) - defaults to config value
            height: Height of area (meters) - defaults to config value
        """
        self.scenario = scenario if scenario is not None else DEFAULT_SCENARIO
        self.rng = np.random.default_rng(self.scenario.simulation.random_seed)

        self.width = width if width is not None else self.scenario.simulation.area_width_m
        self.height = height if height is not None else self.scenario.simulation.area_height_m
        self.targets: List[Target] = []
        # Create radar with configuration parameters
        self.radar = Radar(
            range_limit=self.scenario.radar.detection_range_m,
            measurement_noise=self.scenario.radar.measurement_noise_m,
            detection_probability=self.scenario.radar.detection_probability,
            scan_period=self.scenario.radar.scan_period_s,
            rng=self.rng,
        )
        self.tracker = RadarTracker(
            self.radar,
            acceleration_variance=self.scenario.kalman.acceleration_variance,
            max_age=self.scenario.tracker.max_track_age_s,
            min_hits=self.scenario.tracker.min_hits_to_confirm,
            association_threshold=self.scenario.tracker.association_threshold_m,
        )
        self.time = 0.0
        self.dt = self.scenario.simulation.timestep_s
        self.last_measurements: List[Tuple[int, np.ndarray]] = []
        self.last_removed_targets: int = 0

    def add_target(self, x: float, y: float, vx: float, vy: float,
                  ax: float = 0.0, ay: float = 0.0) -> int:
        """Add a target to the simulation."""
        target_id = len(self.targets)
        target = Target(x, y, vx, vy, ax, ay, target_id)
        self.targets.append(target)
        return target_id

    def remove_out_of_bounds_targets(self, threshold_m: Optional[float] = None) -> int:
        """Remove targets outside the configured/provided radius and return count removed."""
        limit = threshold_m if threshold_m is not None else self.scenario.target.out_of_bounds_m
        before = len(self.targets)
        self.targets = [t for t in self.targets if np.linalg.norm(t.position) < limit]
        removed = before - len(self.targets)
        self.last_removed_targets = removed
        return removed

    def step(self) -> None:
        """Execute one simulation step."""
        # Move targets
        for target in self.targets:
            target.move(self.dt)

        # Remove targets that left simulation area
        self.remove_out_of_bounds_targets()

        # Get radar detections
        measurements = self.radar.detect(self.targets)
        self.last_measurements = measurements

        # Predict and update tracker
        self.tracker.predict(self.dt)
        self.tracker.update(measurements, self.radar.measurement_noise ** 2)
        self.tracker.cleanup_old_tracks()

        self.time += self.dt

    def run_simulation(self, num_steps: int) -> dict:
        """
        Run the simulation and collect data.

        Args:
            num_steps: Number of simulation steps

        Returns:
            Dictionary with simulation results
        """
        times = []
        true_positions = [[] for _ in self.targets]
        true_velocities = [[] for _ in self.targets]
        estimated_positions = [[] for _ in self.targets]
        estimated_velocities = [[] for _ in self.targets]
        measurements_log = []

        for _ in range(num_steps):
            self.step()
            times.append(self.time)

            # Log true target states
            for i, target in enumerate(self.targets):
                true_positions[i].append(target.position.copy())
                true_velocities[i].append(target.velocity.copy())

            # Log estimated states
            measurements_log.append(self.last_measurements)

            for track in self.tracker.get_all_tracks():
                if track.measurements_received <= len(self.targets):
                    track_idx = min(track.id, len(estimated_positions) - 1)
                    if track_idx < len(estimated_positions):
                        estimated_positions[track_idx].append(track.get_estimated_position().copy())
                        estimated_velocities[track_idx].append(track.get_estimated_velocity().copy())

        return {
            'times': np.array(times),
            'true_positions': true_positions,
            'true_velocities': true_velocities,
            'estimated_positions': estimated_positions,
            'estimated_velocities': estimated_velocities,
            'measurements': measurements_log,
            'tracker': self.tracker
        }
