# Kalman Filter

Saab GlobalEye Airborne Radar Simulation with Kalman Filter Tracking

## Overview

This project simulates a Saab GlobalEye (Erieye) airborne early warning radar system with multi-target tracking using Kalman filters. The simulation demonstrates realistic radar detection, measurement noise, and target tracking for aircraft in a 350 km detection range.

## Features

- **Realistic Radar Simulation**: Models Saab GlobalEye/Erieye radar with 350 km range, 150° coverage
- **Kalman Filter Tracking**: Smooth tracking of multiple aircraft with position and velocity estimation
- **Multi-Target Tracking**: Automatic data association and track management
- **Live Visualization**: Real-time animated display with radar sweep, range rings, and velocity vectors
- **Configurable Parameters**: Comprehensive configuration system for all simulation parameters

## Project Layout

- `src/kalman_filter/` - Main package implementation
	- `kf.py` - Kalman Filter implementation
	- `simulator.py` - Radar and target simulation classes
	- `live_radar_sim.py` - Live animated radar display
	- `web_radar_sim.py` - Dash web simulation
	- `radar_demo.py` - Static visualization demo
	- `config.py` - Configuration parameters
	- `cli.py` - Unified CLI entrypoint
- Root `*.py` files are compatibility shims that forward to `src/kalman_filter`
- `test_kf.py` - Unit tests for Kalman filter

## Configuration System

All simulation parameters are centralized in `config.py`, organized into categories:

### RadarConfig
- Detection range (default: 350 km)
- Measurement noise (default: 75m RMS)
- Detection probability (default: 98%)
- Coverage sector (default: 150°)
- Update rate

### TargetConfig
- Maximum targets (default: 20)
- Spawn rate and locations
- Velocity ranges (540-1080 km/h)
- Acceleration ranges
- Spawn distance offsets

### KalmanFilterConfig
- Process noise (acceleration variance)
- Initial state covariances

### TrackerConfig
- Maximum track age (default: 30s)
- Minimum hits to confirm track (default: 3)
- Association threshold (default: 15 km)

### SimulationConfig
- Simulation area size
- Timestep (default: 1 second)
- Animation settings

### VisualizationConfig
- Figure dimensions and DPI
- Plot ranges and limits
- Marker sizes, colors, and styles
- Velocity vector projection time
- Radar sweep parameters
- Grid and legend settings

## Usage

### Basic Usage

Run the default live simulation:
```bash
python live_radar_sim.py
```

### CLI Usage (recommended)

After installing in editable mode:
```bash
python -m pip install -e .
```

Use the unified CLI:
```bash
kalman-filter live
kalman-filter web
kalman-filter demo
```

Run with scenario profiles:
```bash
kalman-filter live --scenario high_traffic
kalman-filter web --scenario low_noise
kalman-filter demo --scenario long_range
```

Deterministic replay + metadata export:
```bash
kalman-filter demo --scenario low_noise --seed 42 --metadata-out artifacts/demo_run.json
kalman-filter live --seed 123 --metadata-out artifacts/live_run.json
kalman-filter web --seed 123 --metadata-out artifacts/web_run.json
```

Verify two runs are identical:
```bash
kalman-filter verify artifacts/run_a.json artifacts/run_b.json
```

You can also run module mode directly from source:
```bash
PYTHONPATH=src python -m kalman_filter live
```

### Custom Configuration

Modify parameters before running:

```python
from config import RadarConfig, TargetConfig, print_config
from live_radar_sim import main

# Customize parameters
RadarConfig.DETECTION_RANGE_KM = 400.0
RadarConfig.DETECTION_RANGE_M = 400000.0
TargetConfig.MAX_TARGETS = 30
TargetConfig.SPAWN_RATE = 0.3

# Print configuration
print_config()

# Run simulation
main()
```

### Using Presets

Load predefined configuration presets:

```python
from config import PresetConfigs
from live_radar_sim import main

# Load high traffic preset
PresetConfigs.high_traffic()  # 30 targets, 30% spawn rate

# Or low noise preset
PresetConfigs.low_noise()  # 25m noise, 99% detection

# Or long range preset
PresetConfigs.long_range()  # 450 km range

# Run simulation
main()
```

### View Current Configuration

```python
from config import print_config

print_config()  # Displays all current parameter values
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Simulation Parameters You Can Adjust

**Radar Performance:**
- Detection range
- Measurement noise
- Detection probability
- Coverage angle

**Target Characteristics:**
- Maximum number of targets
- Spawn rate and locations
- Velocity ranges (by spawn location)
- Acceleration ranges

**Tracking:**
- Kalman filter process noise
- Track confirmation thresholds
- Data association distance
- Track lifetime

**Visualization:**
- Display ranges
- Marker sizes and colors
- Velocity vector projection time
- Animation speed
- Grid and legend styling

## Examples

See `example_usage.py` for complete examples of:
- Running with default configuration
- Using predefined presets
- Creating custom configurations
- Combining multiple parameter changes

## License

MIT
