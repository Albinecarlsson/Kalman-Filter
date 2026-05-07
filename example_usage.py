"""
Example: How to use configuration parameters

This script demonstrates how to customize the radar simulation
by modifying configuration parameters.
"""

from config import (RadarConfig, TargetConfig, KalmanFilterConfig, TrackerConfig,
                    SimulationConfig, VisualizationConfig, PresetConfigs, print_config)
from live_radar_sim import main

# Example 1: Run with default configuration
print("EXAMPLE 1: Default Configuration")
print("-" * 70)
main()

# Example 2: Customize specific parameters
# Uncomment the following to try different configurations:

# # High traffic scenario
# print("\nEXAMPLE 2: High Traffic Scenario")
# print("-" * 70)
# PresetConfigs.high_traffic()
# main()

# # Low noise, high precision radar
# print("\nEXAMPLE 3: Low Noise Radar")
# print("-" * 70)
# PresetConfigs.low_noise()
# main()

# # Custom configuration
# print("\nEXAMPLE 4: Custom Configuration")
# print("-" * 70)
# # Modify individual parameters
# RadarConfig.DETECTION_RANGE_KM = 400.0
# RadarConfig.DETECTION_RANGE_M = 400000.0
# RadarConfig.MEASUREMENT_NOISE_M = 50.0
# 
# TargetConfig.MAX_TARGETS = 15
# TargetConfig.SPAWN_RATE = 0.25
# TargetConfig.MIN_VELOCITY_MS = 200.0  # Faster aircraft
# TargetConfig.MAX_VELOCITY_MS = 400.0
# 
# VisualizationConfig.VELOCITY_VECTOR_TIME_S = 90.0  # Longer projection
# 
# print_config()  # Print the current configuration
# main()
