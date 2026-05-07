"""
Configuration parameters for Saab GlobalEye Radar Simulation

This file contains all configurable parameters for the radar simulation,
including radar performance, target characteristics, tracking parameters,
and visualization settings.
"""

import numpy as np


class RadarConfig:
    """Radar system configuration parameters."""
    
    # Saab GlobalEye/Erieye Radar Performance
    DETECTION_RANGE_KM = 350.0  # Maximum detection range in kilometers
    DETECTION_RANGE_M = DETECTION_RANGE_KM * 1000.0  # In meters
    
    # Measurement accuracy (RMS error in meters)
    MEASUREMENT_NOISE_M = 75.0  # Range measurement noise standard deviation
    
    # Detection probability (0.0 to 1.0)
    DETECTION_PROBABILITY = 0.98  # 98% detection probability for fighter-sized targets
    
    # Scan/Update rate
    SCAN_PERIOD_S = 1.0  # Time between radar scans in seconds
    UPDATE_RATE_HZ = 1.0 / SCAN_PERIOD_S  # Update frequency in Hz
    
    # Coverage sector
    COVERAGE_ANGLE_START_DEG = 30.0  # Start angle of radar coverage (degrees)
    COVERAGE_ANGLE_END_DEG = 150.0   # End angle of radar coverage (degrees)
    COVERAGE_ANGLE_TOTAL_DEG = COVERAGE_ANGLE_END_DEG - COVERAGE_ANGLE_START_DEG
    
    # Radar position (relative to origin)
    RADAR_POSITION_X_M = 0.0
    RADAR_POSITION_Y_M = 0.0


class TargetConfig:
    """Target/Aircraft configuration parameters."""
    
    # Spawn settings
    MAX_TARGETS = 20  # Maximum number of simultaneous targets
    SPAWN_RATE = 0.2  # Probability of spawning new target per timestep (0.0-1.0)
    
    # Aircraft velocity ranges (in m/s)
    # Fighter aircraft: typically 400-900 km/h (111-250 m/s)
    # Commercial aircraft: typically 700-900 km/h (194-250 m/s)
    MIN_VELOCITY_MS = 150.0   # Minimum aircraft speed (540 km/h)
    MAX_VELOCITY_MS = 300.0   # Maximum aircraft speed (1080 km/h)
    
    # Specific velocity ranges for different spawn locations
    ARC_SPAWN_VELOCITY_MIN_MS = 150.0
    ARC_SPAWN_VELOCITY_MAX_MS = 300.0
    
    SIDE_SPAWN_VELOCITY_MIN_MS = 150.0
    SIDE_SPAWN_VELOCITY_MAX_MS = 250.0
    
    FRONT_SPAWN_VELOCITY_MIN_MS = 100.0
    FRONT_SPAWN_VELOCITY_MAX_MS = 250.0
    
    # Aircraft acceleration (m/s²)
    # Aircraft typically have minimal lateral acceleration in cruise
    MIN_ACCELERATION_MS2 = -0.5  # Slight deceleration
    MAX_ACCELERATION_MS2 = 0.5   # Slight acceleration
    
    # Spawn distance (how far outside radar range to spawn)
    SPAWN_DISTANCE_OFFSET_KM = 20.0  # Spawn 20 km outside radar range
    SPAWN_DISTANCE_OFFSET_M = SPAWN_DISTANCE_OFFSET_KM * 1000.0
    
    # Spawn location distribution weights
    SPAWN_LOCATIONS = ['arc', 'left', 'right', 'front']
    SPAWN_WEIGHTS = [0.4, 0.2, 0.2, 0.2]  # Probability weights for each location
    
    # Out-of-bounds distance (when to remove targets)
    OUT_OF_BOUNDS_KM = 600.0  # Remove targets beyond 600 km
    OUT_OF_BOUNDS_M = OUT_OF_BOUNDS_KM * 1000.0


class KalmanFilterConfig:
    """Kalman Filter configuration parameters."""
    
    # Process noise (acceleration variance)
    # Higher values = less smooth tracks, faster response to maneuvers
    # Lower values = smoother tracks, slower response to maneuvers
    ACCELERATION_VARIANCE = 5.0  # m/s² variance
    
    # Initial state covariance
    INITIAL_POSITION_VARIANCE_M2 = 1000.0  # 1000 m² initial position uncertainty
    INITIAL_VELOCITY_VARIANCE_M2S2 = 100.0  # 100 (m/s)² initial velocity uncertainty


class TrackerConfig:
    """Multi-target tracker configuration parameters."""
    
    # Track management
    MAX_TRACK_AGE_S = 30.0  # Maximum time without updates before deleting track (seconds)
    MIN_HITS_TO_CONFIRM = 3  # Minimum detections needed to confirm a track
    
    # Data association
    # Maximum distance between measurement and predicted position to associate
    ASSOCIATION_THRESHOLD_KM = 15.0  # 15 km
    ASSOCIATION_THRESHOLD_M = ASSOCIATION_THRESHOLD_KM * 1000.0
    
    # Track display settings
    MAX_TRACKS_DISPLAY = 10  # Maximum number of tracks to show in statistics panel


class SimulationConfig:
    """General simulation configuration."""
    
    # Simulation area
    AREA_WIDTH_KM = 500.0
    AREA_HEIGHT_KM = 500.0
    AREA_WIDTH_M = AREA_WIDTH_KM * 1000.0
    AREA_HEIGHT_M = AREA_HEIGHT_KM * 1000.0
    
    # Time step
    TIMESTEP_S = 1.0  # Simulation timestep in seconds
    
    # Animation settings
    NUM_FRAMES = 1000  # Number of animation frames
    FRAME_INTERVAL_MS = 100  # Milliseconds between frames (100 ms = 10x faster)
    
    # Random seed (set to None for random, or an integer for reproducible results)
    RANDOM_SEED = None  # e.g., 42 for reproducible simulations


class VisualizationConfig:
    """Visualization and display configuration."""
    
    # Figure settings
    FIGURE_WIDTH = 16
    FIGURE_HEIGHT = 8
    FIGURE_DPI = 100
    
    # Display ranges (in km for the plot)
    PLOT_XLIM_KM = (-RadarConfig.DETECTION_RANGE_KM * 0.9, 
                    RadarConfig.DETECTION_RANGE_KM * 0.9)
    PLOT_YLIM_KM = (-50, RadarConfig.DETECTION_RANGE_KM * 0.95)
    
    # Range rings to display (in km)
    RANGE_RINGS_KM = [100, 200, 300]
    
    # Marker sizes
    TRUE_TARGET_SIZE = 80
    MEASUREMENT_SIZE = 60
    TRACKED_TARGET_SIZE = 120
    RADAR_CENTER_SIZE = 20
    
    # Marker styles
    TRUE_TARGET_COLOR = 'blue'
    TRUE_TARGET_MARKER = 'o'
    TRUE_TARGET_ALPHA = 0.7
    
    MEASUREMENT_COLOR = 'yellow'
    MEASUREMENT_MARKER = 'x'
    MEASUREMENT_ALPHA = 0.8
    MEASUREMENT_LINEWIDTH = 2
    
    TRACKED_TARGET_COLOR = 'red'
    TRACKED_TARGET_MARKER = 's'
    TRACKED_TARGET_EDGECOLOR = 'darkred'
    TRACKED_TARGET_LINEWIDTH = 2
    
    RADAR_COLOR = 'green'
    
    # Velocity vector settings
    VELOCITY_VECTOR_COLOR = 'red'
    VELOCITY_VECTOR_LINEWIDTH = 2
    VELOCITY_VECTOR_ALPHA = 0.6
    VELOCITY_VECTOR_TIME_S = 60.0  # Show 60-second projection
    VELOCITY_VECTOR_MIN_SPEED_MS = 10.0  # Only show if speed > 10 m/s (36 km/h)
    
    # Radar sweep settings
    RADAR_SWEEP_COLOR = 'lime'  # Bright lime green for visibility
    RADAR_SWEEP_ALPHA = 0.3  # More visible illumination (was 0.1)
    RADAR_SWEEP_WIDTH_DEG = 20.0  # Width of sweep beam in degrees
    RADAR_SWEEP_SPEED_DEG_PER_FRAME = 15.0  # Faster sweep (was 10.0)
    # Illumination beam (bright active area)
    ILLUMINATION_COLOR = 'lime'
    ILLUMINATION_ALPHA = 0.2  # Subtle glow for currently illuminated area
    
    # Grid settings
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    
    # Legend settings
    LEGEND_LOCATION = 'upper left'
    LEGEND_FONTSIZE = 9
    LEGEND_FRAMEALPHA = 0.9
    
    # Statistics panel font
    STATS_FONTSIZE = 10
    STATS_FONT_FAMILY = 'monospace'


class PresetConfigs:
    """Predefined configuration presets for different scenarios."""
    
    @staticmethod
    def high_traffic():
        """High traffic scenario with many targets."""
        TargetConfig.MAX_TARGETS = 30
        TargetConfig.SPAWN_RATE = 0.3
        print("Loaded preset: High Traffic (30 max targets, 30% spawn rate)")
    
    @staticmethod
    def low_noise():
        """Low noise scenario for better tracking accuracy."""
        RadarConfig.MEASUREMENT_NOISE_M = 25.0
        RadarConfig.DETECTION_PROBABILITY = 0.99
        print("Loaded preset: Low Noise (25m noise, 99% detection)")
    
    @staticmethod
    def high_noise():
        """High noise scenario for challenging conditions."""
        RadarConfig.MEASUREMENT_NOISE_M = 150.0
        RadarConfig.DETECTION_PROBABILITY = 0.90
        print("Loaded preset: High Noise (150m noise, 90% detection)")
    
    @staticmethod
    def long_range():
        """Extended range scenario."""
        RadarConfig.DETECTION_RANGE_KM = 450.0
        RadarConfig.DETECTION_RANGE_M = 450000.0
        VisualizationConfig.RANGE_RINGS_KM = [100, 200, 300, 400]
        print("Loaded preset: Long Range (450 km detection)")   
    
    @staticmethod
    def fast_aircraft():
        """Scenario with faster aircraft (supersonic)."""
        TargetConfig.MIN_VELOCITY_MS = 250.0  # 900 km/h
        TargetConfig.MAX_VELOCITY_MS = 600.0  # 2160 km/h (Mach 1.8)
        print("Loaded preset: Fast Aircraft (900-2160 km/h)")
    
    @staticmethod
    def reset_defaults():
        """Reset all parameters to default values."""
        # Reload the module to get default values
        print("Reset all parameters to defaults")


# Helper function to print current configuration
def print_config():
    """Print current configuration parameters."""
    print("\n" + "="*70)
    print("Current Simulation Configuration")
    print("="*70)
    
    print("\n[Radar Configuration]")
    print(f"  Detection Range: {RadarConfig.DETECTION_RANGE_KM} km")
    print(f"  Measurement Noise: {RadarConfig.MEASUREMENT_NOISE_M} m")
    print(f"  Detection Probability: {RadarConfig.DETECTION_PROBABILITY*100:.1f}%")
    print(f"  Coverage Sector: {RadarConfig.COVERAGE_ANGLE_TOTAL_DEG}°")
    print(f"  Update Rate: {RadarConfig.UPDATE_RATE_HZ} Hz")
    
    print("\n[Target Configuration]")
    print(f"  Max Targets: {TargetConfig.MAX_TARGETS}")
    print(f"  Spawn Rate: {TargetConfig.SPAWN_RATE*100:.1f}%")
    print(f"  Velocity Range: {TargetConfig.MIN_VELOCITY_MS*3.6:.0f}-{TargetConfig.MAX_VELOCITY_MS*3.6:.0f} km/h")
    print(f"  Acceleration Range: {TargetConfig.MIN_ACCELERATION_MS2:.1f} to {TargetConfig.MAX_ACCELERATION_MS2:.1f} m/s²")
    
    print("\n[Kalman Filter Configuration]")
    print(f"  Acceleration Variance: {KalmanFilterConfig.ACCELERATION_VARIANCE}")
    
    print("\n[Tracker Configuration]")
    print(f"  Max Track Age: {TrackerConfig.MAX_TRACK_AGE_S} s")
    print(f"  Min Hits to Confirm: {TrackerConfig.MIN_HITS_TO_CONFIRM}")
    print(f"  Association Threshold: {TrackerConfig.ASSOCIATION_THRESHOLD_KM} km")
    
    print("\n[Simulation Configuration]")
    print(f"  Timestep: {SimulationConfig.TIMESTEP_S} s")
    print(f"  Simulation Frames: {SimulationConfig.NUM_FRAMES}")
    print(f"  Frame Interval: {SimulationConfig.FRAME_INTERVAL_MS} ms")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Print default configuration
    print_config()
    
    # Example: Load a preset
    print("\nExample: Loading 'High Traffic' preset...")
    PresetConfigs.high_traffic()
    print_config()
