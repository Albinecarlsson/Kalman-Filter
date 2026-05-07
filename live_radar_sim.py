"""
Live Radar System Simulation with Random Target Spawning

Real-time animated simulation showing:
- Randomly spawning targets
- Radar illumination/coverage area
- Detected measurements
- Tracked objects with estimated positions
- Real-time statistics
"""

import numpy as np
import matplotlib
# Configure matplotlib backend for Windows
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from simulator import RadarSimulationArea, Target
import random

# Import configuration parameters
from config import (RadarConfig, TargetConfig, KalmanFilterConfig, TrackerConfig,
                    SimulationConfig, VisualizationConfig, print_config)


class LiveRadarSimulation:
    """Live animated radar simulation with random target spawning - Saab GlobalEye configuration."""

    def __init__(self, spawn_rate=None, max_targets=None):
        """
        Initialize the live simulation with Saab GlobalEye/Erieye radar parameters.

        Args:
            spawn_rate: Probability of spawning a new target each step (None = use config)
            max_targets: Maximum number of active targets (None = use config)
        """
        self.area = RadarSimulationArea()
        self.spawn_rate = spawn_rate if spawn_rate is not None else TargetConfig.SPAWN_RATE
        self.max_targets = max_targets if max_targets is not None else TargetConfig.MAX_TARGETS
        self.radar_sweep_angle = RadarConfig.COVERAGE_ANGLE_START_DEG
        self.step_count = 0
        self.coverage_angle_start = RadarConfig.COVERAGE_ANGLE_START_DEG
        self.coverage_angle_end = RadarConfig.COVERAGE_ANGLE_END_DEG

        # Configure figure
        self.fig, (self.ax_radar, self.ax_stats) = plt.subplots(
            1, 2, figsize=(VisualizationConfig.FIGURE_WIDTH, VisualizationConfig.FIGURE_HEIGHT),
            dpi=VisualizationConfig.FIGURE_DPI
        )
        self.fig.suptitle('Saab GlobalEye Airborne Radar System - Kalman Filter Tracking', 
                         fontsize=16, fontweight='bold')

        # Set up radar axis (150° sector, scaled for km)
        range_km = RadarConfig.DETECTION_RANGE_KM
        self.ax_radar.set_xlim(VisualizationConfig.PLOT_XLIM_KM)
        self.ax_radar.set_ylim(VisualizationConfig.PLOT_YLIM_KM)
        self.ax_radar.set_xlabel('X Position (km)')
        self.ax_radar.set_ylabel('Y Position (km)')
        self.ax_radar.set_title(f'GlobalEye Radar Display ({RadarConfig.COVERAGE_ANGLE_TOTAL_DEG:.0f}° Coverage)')
        self.ax_radar.grid(True, alpha=VisualizationConfig.GRID_ALPHA, 
                          linestyle=VisualizationConfig.GRID_LINESTYLE)
        self.ax_radar.set_aspect('equal')

        # Radar elements - coverage sector
        self.radar_center = self.ax_radar.plot(
            0, 0,
            marker='*',
            color=VisualizationConfig.RADAR_COLOR,
            markersize=VisualizationConfig.RADAR_CENTER_SIZE,
            label='GlobalEye Platform',
            zorder=5
        )[0]
        # Create sector using Wedge
        self.radar_circle = Wedge(
            (0, 0), range_km, self.coverage_angle_start, self.coverage_angle_end,
            fill=False, linestyle='--', color=VisualizationConfig.RADAR_COLOR, 
            linewidth=2, alpha=0.5
        )
        self.ax_radar.add_patch(self.radar_circle)
        
        # Add range rings (like real radar displays)
        for range_ring in VisualizationConfig.RANGE_RINGS_KM:
            if range_ring <= range_km:
                ring = Wedge(
                    (0, 0), range_ring, self.coverage_angle_start, self.coverage_angle_end,
                    fill=False, linestyle=':', color='gray', linewidth=1, alpha=0.3
                )
                self.ax_radar.add_patch(ring)

        # Plot elements (initialize empty, will be updated)
        self.true_targets_scatter = None
        self.measurements_scatter = None
        self.tracked_positions_scatter = None
        self.tracked_velocities = None
        self.tracked_trails = []
        self.measurement_trails = []

        # Stats axis
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', transform=self.ax_stats.transAxes,
            fontsize=VisualizationConfig.STATS_FONTSIZE, 
            verticalalignment='top', 
            fontfamily=VisualizationConfig.STATS_FONT_FAMILY,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        self.ax_radar.legend(loc=VisualizationConfig.LEGEND_LOCATION, 
                            fontsize=VisualizationConfig.LEGEND_FONTSIZE)

    def spawn_random_target(self):
        """Spawn a random aircraft target at the edge of the radar coverage."""
        if len(self.area.targets) >= self.max_targets:
            return

        if np.random.rand() > self.spawn_rate:
            return

        # Spawn aircraft with realistic velocities using config parameters
        side = np.random.choice(TargetConfig.SPAWN_LOCATIONS, p=TargetConfig.SPAWN_WEIGHTS)

        if side == 'arc':
            # Spawn on the curved arc edge
            angle = np.random.uniform(self.coverage_angle_start + 10, self.coverage_angle_end - 10)
            angle_rad = np.radians(angle)
            radius = RadarConfig.DETECTION_RANGE_M + TargetConfig.SPAWN_DISTANCE_OFFSET_M
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            # Velocity toward radar (inbound aircraft)
            speed = np.random.uniform(TargetConfig.ARC_SPAWN_VELOCITY_MIN_MS, 
                                     TargetConfig.ARC_SPAWN_VELOCITY_MAX_MS)
            heading = np.random.uniform(-180, -90)  # Generally toward center
            heading_rad = np.radians(angle + heading)
            vx = speed * np.cos(heading_rad)
            vy = speed * np.sin(heading_rad)
        elif side == 'left':
            # Spawn on left side
            x = -300000  # -300 km
            y = np.random.uniform(50000, 300000)
            vx = np.random.uniform(TargetConfig.SIDE_SPAWN_VELOCITY_MIN_MS, 
                                  TargetConfig.SIDE_SPAWN_VELOCITY_MAX_MS)
            vy = np.random.uniform(-50, 50)
        elif side == 'right':
            # Spawn on right side
            x = 300000  # 300 km
            y = np.random.uniform(50000, 300000)
            vx = np.random.uniform(-TargetConfig.SIDE_SPAWN_VELOCITY_MAX_MS, 
                                  -TargetConfig.SIDE_SPAWN_VELOCITY_MIN_MS)
            vy = np.random.uniform(-50, 50)
        else:  # front
            # Spawn in front, flying across
            x = np.random.uniform(-200000, 200000)
            y = 370000  # 370 km away
            vx = np.random.uniform(-200, 200)
            vy = np.random.uniform(-TargetConfig.FRONT_SPAWN_VELOCITY_MAX_MS, 
                                  -TargetConfig.FRONT_SPAWN_VELOCITY_MIN_MS)

        # Minimal acceleration for aircraft from config
        ax = np.random.uniform(TargetConfig.MIN_ACCELERATION_MS2, 
                              TargetConfig.MAX_ACCELERATION_MS2)
        ay = np.random.uniform(TargetConfig.MIN_ACCELERATION_MS2, 
                              TargetConfig.MAX_ACCELERATION_MS2)

        self.area.add_target(x, y, vx, vy, ax, ay)

    def remove_out_of_bounds_targets(self):
        """Remove targets that have moved far outside the area."""
        self.area.targets = [t for t in self.area.targets
                            if np.linalg.norm(t.position) < TargetConfig.OUT_OF_BOUNDS_M]

    def update_frame(self, frame):
        """Update simulation and plot for one frame."""
        # Spawn new targets randomly
        self.spawn_random_target()

        # Run simulation step
        self.area.step()
        self.step_count += 1

        # Remove out-of-bounds targets
        self.remove_out_of_bounds_targets()

        # Update radar sweep visual effect (coverage angle range)
        self.radar_sweep_angle = self.coverage_angle_start + \
                                ((self.radar_sweep_angle - self.coverage_angle_start + 
                                  VisualizationConfig.RADAR_SWEEP_SPEED_DEG_PER_FRAME) % 
                                 (self.coverage_angle_end - self.coverage_angle_start))

        # Clear previous scatter plots safely
        try:
            if self.true_targets_scatter:
                self.true_targets_scatter.remove()
                self.true_targets_scatter = None
        except (ValueError, AttributeError):
            pass
        
        try:
            if self.measurements_scatter:
                self.measurements_scatter.remove()
                self.measurements_scatter = None
        except (ValueError, AttributeError):
            pass
        
        try:
            if self.tracked_positions_scatter:
                self.tracked_positions_scatter.remove()
                self.tracked_positions_scatter = None
        except (ValueError, AttributeError):
            pass
        
        try:
            if self.tracked_velocities:
                self.tracked_velocities.remove()
                self.tracked_velocities = None
        except (ValueError, AttributeError):
            pass

        # Get radar detections
        measurements = self.area.radar.detect(self.area.targets)

        # Plot true targets (convert to km for display)
        if self.area.targets:
            target_positions = np.array([t.position / 1000.0 for t in self.area.targets])
            self.true_targets_scatter = self.ax_radar.scatter(
                target_positions[:, 0], target_positions[:, 1],
                c=VisualizationConfig.TRUE_TARGET_COLOR, 
                s=VisualizationConfig.TRUE_TARGET_SIZE, 
                marker=VisualizationConfig.TRUE_TARGET_MARKER, 
                label='True Targets', zorder=4, 
                alpha=VisualizationConfig.TRUE_TARGET_ALPHA
            )
        else:
            self.true_targets_scatter = None

        # Plot measurements (noisy detections, in km)
        if measurements:
            measurement_positions = np.array([m[1] / 1000.0 for m in measurements])
            self.measurements_scatter = self.ax_radar.scatter(
                measurement_positions[:, 0], measurement_positions[:, 1],
                c=VisualizationConfig.MEASUREMENT_COLOR, 
                s=VisualizationConfig.MEASUREMENT_SIZE, 
                marker=VisualizationConfig.MEASUREMENT_MARKER, 
                linewidths=VisualizationConfig.MEASUREMENT_LINEWIDTH, 
                label='Detections', zorder=3, 
                alpha=VisualizationConfig.MEASUREMENT_ALPHA
            )
        else:
            self.measurements_scatter = None

        # Plot tracked objects (in km)
        confirmed_tracks = self.area.tracker.get_confirmed_tracks()
        if confirmed_tracks:
            track_positions = np.array([t.get_estimated_position() / 1000.0 for t in confirmed_tracks])
            self.tracked_positions_scatter = self.ax_radar.scatter(
                track_positions[:, 0], track_positions[:, 1],
                c=VisualizationConfig.TRACKED_TARGET_COLOR, 
                s=VisualizationConfig.TRACKED_TARGET_SIZE, 
                marker=VisualizationConfig.TRACKED_TARGET_MARKER, 
                label='Tracked Aircraft', zorder=4, 
                edgecolors=VisualizationConfig.TRACKED_TARGET_EDGECOLOR, 
                linewidths=VisualizationConfig.TRACKED_TARGET_LINEWIDTH
            )

            # Draw velocity vectors for tracked objects (in km scale)
            velocity_segments = []
            velocity_colors = []
            for track in confirmed_tracks:
                pos = track.get_estimated_position() / 1000.0  # Convert to km
                vel = track.get_estimated_velocity()
                speed_ms = np.linalg.norm(vel)
                if speed_ms > VisualizationConfig.VELOCITY_VECTOR_MIN_SPEED_MS:
                    # Scale factor: show velocity vector for projection time
                    scale = VisualizationConfig.VELOCITY_VECTOR_TIME_S / 1000.0  # Convert m/s to km
                    velocity_segments.append([(pos[0], pos[1]), 
                                             (pos[0] + vel[0]*scale, pos[1] + vel[1]*scale)])
                    velocity_colors.append(VisualizationConfig.VELOCITY_VECTOR_COLOR)

            if velocity_segments:
                lc = LineCollection(velocity_segments, colors=velocity_colors, 
                                   linewidths=VisualizationConfig.VELOCITY_VECTOR_LINEWIDTH, 
                                   alpha=VisualizationConfig.VELOCITY_VECTOR_ALPHA)
                self.tracked_velocities = self.ax_radar.add_collection(lc)
        else:
            self.tracked_velocities = None

        # Draw radar sweep (in km)
        range_km = RadarConfig.DETECTION_RANGE_KM
        
        # Add bright illumination wedge (primary beam - wider and more prominent)
        illumination = Wedge((0, 0), range_km, self.radar_sweep_angle,
                            self.radar_sweep_angle + VisualizationConfig.RADAR_SWEEP_WIDTH_DEG * 2.5,
                            alpha=VisualizationConfig.ILLUMINATION_ALPHA,
                            facecolor=VisualizationConfig.ILLUMINATION_COLOR, zorder=2, linewidth=0.5, edgecolor='white')
        self.ax_radar.add_patch(illumination)
        
        # Draw trailing sweep wedge (secondary beam - shows recent scan path)
        sweep = Wedge((0, 0), range_km, self.radar_sweep_angle,
                      self.radar_sweep_angle + VisualizationConfig.RADAR_SWEEP_WIDTH_DEG, 
                      alpha=VisualizationConfig.RADAR_SWEEP_ALPHA, 
                      color=VisualizationConfig.RADAR_SWEEP_COLOR, zorder=1)
        self.ax_radar.add_patch(sweep)

        # Update legend
        self.ax_radar.legend(loc=VisualizationConfig.LEGEND_LOCATION, 
                            fontsize=VisualizationConfig.LEGEND_FONTSIZE, 
                            framealpha=VisualizationConfig.LEGEND_FRAMEALPHA)

        # Update statistics
        self._update_stats(measurements)

        return []

    def _update_stats(self, measurements):
        """Update statistics display."""
        stats = f"{'Saab GlobalEye Radar System':^55}\n"
        stats += "="*55 + "\n\n"
        stats += f"Simulation Time:        {self.area.time:>10.1f} s ({self.area.time/60:>5.1f} min)\n"
        stats += f"Step Number:            {self.step_count:>10}\n"
        stats += f"Active Aircraft:        {len(self.area.targets):>10}\n"
        stats += f"Confirmed Tracks:       {len(self.area.tracker.get_confirmed_tracks()):>10}\n"
        stats += f"All Tracks:             {len(self.area.tracker.get_all_tracks()):>10}\n"
        stats += f"Total Created:          {self.area.tracker.next_id:>10}\n"
        stats += f"Current Detections:     {len(measurements):>10}\n"

        stats += f"\n{'GlobalEye/Erieye Radar Specs':^55}\n"
        stats += "-"*55 + "\n"
        stats += f"Detection Range:        {self.area.radar.range_limit/1000:>10.0f} km\n"
        stats += f"Coverage Sector:        {self.coverage_angle_end - self.coverage_angle_start:>10.0f}°\n"
        stats += f"Measurement Noise σ:    {self.area.radar.measurement_noise:>10.0f} m\n"
        stats += f"Detection Probability:  {self.area.radar.detection_probability*100:>9.1f}%\n"
        stats += f"Update Rate:            {1.0/self.area.dt:>10.1f} Hz\n"

        stats += f"\n{'Tracker Configuration':^55}\n"
        stats += "-"*55 + "\n"
        stats += f"Max Track Age:          {self.area.tracker.max_age:>10.1f} s\n"
        stats += f"Min Hits to Confirm:    {self.area.tracker.min_hits:>10}\n"
        stats += f"Association Threshold:  {self.area.tracker.association_threshold/1000:>10.1f} km\n"

        stats += f"\n{'Active Aircraft Tracks':^55}\n"
        stats += "-"*55 + "\n"

        if self.area.tracker.get_all_tracks():
            for i, track in enumerate(self.area.tracker.get_all_tracks()[:10]):  # Show first 10
                status = "✓" if track.measurements_received >= self.area.tracker.min_hits else "⋯"
                pos = track.get_estimated_position() / 1000.0  # km
                vel = track.get_estimated_velocity()
                speed_ms = np.linalg.norm(vel)
                speed_kmh = speed_ms * 3.6
                heading = np.degrees(np.arctan2(vel[1], vel[0]))
                
                stats += f"\n{status} Track {track.id:03d}:\n"
                stats += f"  Pos: ({pos[0]:7.1f}, {pos[1]:7.1f}) km\n"
                stats += f"  Speed: {speed_kmh:6.0f} km/h  Hdg: {heading:6.1f}°\n"
                stats += f"  Age: {track.time_since_update:4.1f}s  Hits: {track.measurements_received:3d}\n"
            
            if len(self.area.tracker.get_all_tracks()) > 10:
                stats += f"\n... and {len(self.area.tracker.get_all_tracks()) - 10} more tracks\n"
        else:
            stats += "\nNo active tracks\n"

        self.stats_text.set_text(stats)

    def run(self, num_frames=None):
        """Run the live simulation animation."""
        frames = num_frames if num_frames is not None else SimulationConfig.NUM_FRAMES
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                           interval=SimulationConfig.FRAME_INTERVAL_MS, repeat=False, blit=False)
        plt.tight_layout()
        return ani


def main():
    """Main entry point for Saab GlobalEye radar simulation."""
    print("="*70)
    print("Saab GlobalEye Airborne Early Warning Radar System")
    
    # Print current configuration
    print_config()
    
    print("\nStarting live simulation...")
    print(f"  System: Erieye AEW&C Radar")
    print(f"  Detection Range: {RadarConfig.DETECTION_RANGE_KM} km")
    print(f"  Coverage: {RadarConfig.COVERAGE_ANGLE_TOTAL_DEG:.0f}° sector")
    print(f"  Max Tracking Capacity: {TargetConfig.MAX_TARGETS} targets displayed")
    print("\nVisualization:")
    print(f"  - {VisualizationConfig.TRUE_TARGET_COLOR.capitalize()} circles: True aircraft positions")
    print(f"  - {VisualizationConfig.MEASUREMENT_COLOR.capitalize()} X marks: Radar detections (with ~{RadarConfig.MEASUREMENT_NOISE_M:.0f}m noise)")
    print(f"  - {VisualizationConfig.TRACKED_TARGET_COLOR.capitalize()} squares: Tracked aircraft with Kalman filter estimates")
    print(f"  - {VisualizationConfig.VELOCITY_VECTOR_COLOR.capitalize()} arrows: Velocity vectors ({VisualizationConfig.VELOCITY_VECTOR_TIME_S:.0f}-second projection)")
    print(f"  - {VisualizationConfig.ILLUMINATION_COLOR.capitalize()} bright cone: Current radar illumination beam (active)")
    print(f"  - {VisualizationConfig.RADAR_SWEEP_COLOR.capitalize()} sweep: Radar scanning history")
    print(f"\nAircraft velocities: {TargetConfig.MIN_VELOCITY_MS*3.6:.0f}-{TargetConfig.MAX_VELOCITY_MS*3.6:.0f} km/h")
    print("\nClose the window to exit.\n")

    # Create and run simulation with config parameters
    sim = LiveRadarSimulation()
    ani = sim.run()

    plt.show()


if __name__ == "__main__":
    main()
