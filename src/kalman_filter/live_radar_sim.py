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
from typing import Optional
import matplotlib
# Configure matplotlib backend for Windows
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from .simulator import RadarSimulationArea, Target
import random

from .scenario import DEFAULT_SCENARIO, Scenario, get_scenario, print_scenario, with_seed
from .replay import build_run_metadata, write_run_metadata


class LiveRadarSimulation:
    """Live animated radar simulation with random target spawning - Saab GlobalEye configuration."""

    def __init__(self, spawn_rate=None, max_targets=None, scenario: Scenario = DEFAULT_SCENARIO):
        """
        Initialize the live simulation with Saab GlobalEye/Erieye radar parameters.

        Args:
            spawn_rate: Probability of spawning a new target each step (None = use config)
            max_targets: Maximum number of active targets (None = use config)
        """
        self.scenario = scenario
        self.radar_cfg = scenario.radar
        self.target_cfg = scenario.target
        self.sim_cfg = scenario.simulation
        self.vis_cfg = scenario.visualization

        self.area = RadarSimulationArea(scenario=scenario)
        self.spawn_rate = spawn_rate if spawn_rate is not None else self.target_cfg.spawn_rate
        self.max_targets = max_targets if max_targets is not None else self.target_cfg.max_targets
        self.radar_sweep_angle = self.radar_cfg.coverage_angle_start_deg
        self.step_count = 0
        self.coverage_angle_start = self.radar_cfg.coverage_angle_start_deg
        self.coverage_angle_end = self.radar_cfg.coverage_angle_end_deg

        # Configure figure
        self.fig, (self.ax_radar, self.ax_stats) = plt.subplots(
            1, 2, figsize=(self.vis_cfg.figure_width, self.vis_cfg.figure_height),
            dpi=self.vis_cfg.figure_dpi
        )
        self.fig.suptitle('Saab GlobalEye Airborne Radar System - Kalman Filter Tracking', 
                         fontsize=16, fontweight='bold')

        # Set up radar axis (150° sector, scaled for km)
        range_km = self.radar_cfg.detection_range_km
        self.ax_radar.set_xlim(self.vis_cfg.plot_xlim_km(range_km))
        self.ax_radar.set_ylim(self.vis_cfg.plot_ylim_km_for_range(range_km))
        self.ax_radar.set_xlabel('X Position (km)')
        self.ax_radar.set_ylabel('Y Position (km)')
        self.ax_radar.set_title(f'GlobalEye Radar Display ({self.radar_cfg.coverage_angle_total_deg:.0f}° Coverage)')
        self.ax_radar.grid(True, alpha=self.vis_cfg.grid_alpha,
                  linestyle=self.vis_cfg.grid_linestyle)
        self.ax_radar.set_aspect('equal')

        # Radar elements - coverage sector
        self.radar_center = self.ax_radar.plot(
            0, 0,
            marker='*',
            color=self.vis_cfg.radar_color,
            markersize=self.vis_cfg.radar_center_size,
            label='GlobalEye Platform',
            zorder=5
        )[0]
        # Create sector using Wedge
        self.radar_circle = Wedge(
            (0, 0), range_km, self.coverage_angle_start, self.coverage_angle_end,
            fill=False, linestyle='--', color=self.vis_cfg.radar_color,
            linewidth=2, alpha=0.5
        )
        self.ax_radar.add_patch(self.radar_circle)
        
        # Add range rings (like real radar displays)
        for range_ring in self.vis_cfg.range_rings_km:
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
            fontsize=self.vis_cfg.stats_fontsize,
            verticalalignment='top', 
            fontfamily=self.vis_cfg.stats_font_family,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        self.ax_radar.legend(loc=self.vis_cfg.legend_location,
                            fontsize=self.vis_cfg.legend_fontsize)

    def spawn_random_target(self):
        """Spawn a random aircraft target at the edge of the radar coverage."""
        rng = self.area.rng

        if len(self.area.targets) >= self.max_targets:
            return

        if rng.random() > self.spawn_rate:
            return

        # Spawn aircraft with realistic velocities using config parameters
        side = rng.choice(self.target_cfg.spawn_locations, p=self.target_cfg.spawn_weights)

        if side == 'arc':
            # Spawn on the curved arc edge
            angle = rng.uniform(self.coverage_angle_start + 10, self.coverage_angle_end - 10)
            angle_rad = np.radians(angle)
            radius = self.radar_cfg.detection_range_m + self.target_cfg.spawn_distance_offset_m
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            # Velocity toward radar (inbound aircraft)
            speed = rng.uniform(self.target_cfg.arc_spawn_velocity_min_ms,
                                     self.target_cfg.arc_spawn_velocity_max_ms)
            heading = rng.uniform(-180, -90)  # Generally toward center
            heading_rad = np.radians(angle + heading)
            vx = speed * np.cos(heading_rad)
            vy = speed * np.sin(heading_rad)
        elif side == 'left':
            # Spawn on left side
            x = -300000  # -300 km
            y = rng.uniform(50000, 300000)
            vx = rng.uniform(self.target_cfg.side_spawn_velocity_min_ms,
                                  self.target_cfg.side_spawn_velocity_max_ms)
            vy = rng.uniform(-50, 50)
        elif side == 'right':
            # Spawn on right side
            x = 300000  # 300 km
            y = rng.uniform(50000, 300000)
            vx = rng.uniform(-self.target_cfg.side_spawn_velocity_max_ms,
                                  -self.target_cfg.side_spawn_velocity_min_ms)
            vy = rng.uniform(-50, 50)
        else:  # front
            # Spawn in front, flying across
            x = rng.uniform(-200000, 200000)
            y = 370000  # 370 km away
            vx = rng.uniform(-200, 200)
            vy = rng.uniform(-self.target_cfg.front_spawn_velocity_max_ms,
                                  -self.target_cfg.front_spawn_velocity_min_ms)

        # Minimal acceleration for aircraft from config
        ax = rng.uniform(self.target_cfg.min_acceleration_ms2,
                      self.target_cfg.max_acceleration_ms2)
        ay = rng.uniform(self.target_cfg.min_acceleration_ms2,
                      self.target_cfg.max_acceleration_ms2)

        self.area.add_target(x, y, vx, vy, ax, ay)

    def remove_out_of_bounds_targets(self):
        """Remove targets that have moved far outside the area."""
        self.area.targets = [t for t in self.area.targets
                            if np.linalg.norm(t.position) < self.target_cfg.out_of_bounds_m]

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
                                                                    self.vis_cfg.radar_sweep_speed_deg_per_frame) %
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
        measurements = self.area.last_measurements

        # Plot true targets (convert to km for display)
        if self.area.targets:
            target_positions = np.array([t.position / 1000.0 for t in self.area.targets])
            self.true_targets_scatter = self.ax_radar.scatter(
                target_positions[:, 0], target_positions[:, 1],
                c=self.vis_cfg.true_target_color,
                s=self.vis_cfg.true_target_size,
                marker=self.vis_cfg.true_target_marker,
                label='True Targets', zorder=4, 
                alpha=self.vis_cfg.true_target_alpha
            )
        else:
            self.true_targets_scatter = None

        # Plot measurements (noisy detections, in km)
        if measurements:
            measurement_positions = np.array([m[1] / 1000.0 for m in measurements])
            self.measurements_scatter = self.ax_radar.scatter(
                measurement_positions[:, 0], measurement_positions[:, 1],
                c=self.vis_cfg.measurement_color,
                s=self.vis_cfg.measurement_size,
                marker=self.vis_cfg.measurement_marker,
                linewidths=self.vis_cfg.measurement_linewidth,
                label='Detections', zorder=3, 
                alpha=self.vis_cfg.measurement_alpha
            )
        else:
            self.measurements_scatter = None

        # Plot tracked objects (in km)
        confirmed_tracks = self.area.tracker.get_confirmed_tracks()
        if confirmed_tracks:
            track_positions = np.array([t.get_estimated_position() / 1000.0 for t in confirmed_tracks])
            self.tracked_positions_scatter = self.ax_radar.scatter(
                track_positions[:, 0], track_positions[:, 1],
                c=self.vis_cfg.tracked_target_color,
                s=self.vis_cfg.tracked_target_size,
                marker=self.vis_cfg.tracked_target_marker,
                label='Tracked Aircraft', zorder=4, 
                edgecolors=self.vis_cfg.tracked_target_edgecolor,
                linewidths=self.vis_cfg.tracked_target_linewidth
            )

            # Draw velocity vectors for tracked objects (in km scale)
            velocity_segments = []
            velocity_colors = []
            for track in confirmed_tracks:
                pos = track.get_estimated_position() / 1000.0  # Convert to km
                vel = track.get_estimated_velocity()
                speed_ms = np.linalg.norm(vel)
                if speed_ms > self.vis_cfg.velocity_vector_min_speed_ms:
                    # Scale factor: show velocity vector for projection time
                    scale = self.vis_cfg.velocity_vector_time_s / 1000.0  # Convert m/s to km
                    velocity_segments.append([(pos[0], pos[1]), 
                                             (pos[0] + vel[0]*scale, pos[1] + vel[1]*scale)])
                    velocity_colors.append(self.vis_cfg.velocity_vector_color)

            if velocity_segments:
                lc = LineCollection(velocity_segments, colors=velocity_colors, 
                                   linewidths=self.vis_cfg.velocity_vector_linewidth,
                                   alpha=self.vis_cfg.velocity_vector_alpha)
                self.tracked_velocities = self.ax_radar.add_collection(lc)
        else:
            self.tracked_velocities = None

        # Draw radar sweep (in km)
        range_km = self.radar_cfg.detection_range_km
        
        # Add bright illumination wedge (primary beam - wider and more prominent)
        illumination = Wedge((0, 0), range_km, self.radar_sweep_angle,
                            self.radar_sweep_angle + self.vis_cfg.radar_sweep_width_deg * 2.5,
                            alpha=self.vis_cfg.illumination_alpha,
                            facecolor=self.vis_cfg.illumination_color, zorder=2, linewidth=0.5, edgecolor='white')
        self.ax_radar.add_patch(illumination)
        
        # Draw trailing sweep wedge (secondary beam - shows recent scan path)
        sweep = Wedge((0, 0), range_km, self.radar_sweep_angle,
                      self.radar_sweep_angle + self.vis_cfg.radar_sweep_width_deg,
                      alpha=self.vis_cfg.radar_sweep_alpha,
                      color=self.vis_cfg.radar_sweep_color, zorder=1)
        self.ax_radar.add_patch(sweep)

        # Update legend
        self.ax_radar.legend(loc=self.vis_cfg.legend_location,
                    fontsize=self.vis_cfg.legend_fontsize,
                    framealpha=self.vis_cfg.legend_framealpha)

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
        frames = num_frames if num_frames is not None else self.sim_cfg.num_frames
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                           interval=self.sim_cfg.frame_interval_ms, repeat=False, blit=False)
        plt.tight_layout()
        return ani


def main(scenario_name: str = "default", seed: Optional[int] = None, metadata_out: Optional[str] = None):
    """Main entry point for Saab GlobalEye radar simulation."""
    scenario = with_seed(get_scenario(scenario_name), seed)

    print("="*70)
    print("Saab GlobalEye Airborne Early Warning Radar System")
    
    # Print current configuration
    print_scenario(scenario)
    
    print("\nStarting live simulation...")
    print(f"  System: Erieye AEW&C Radar")
    print(f"  Detection Range: {scenario.radar.detection_range_km} km")
    print(f"  Coverage: {scenario.radar.coverage_angle_total_deg:.0f}° sector")
    print(f"  Max Tracking Capacity: {scenario.target.max_targets} targets displayed")
    print("\nVisualization:")
    print(f"  - {scenario.visualization.true_target_color.capitalize()} circles: True aircraft positions")
    print(f"  - {scenario.visualization.measurement_color.capitalize()} X marks: Radar detections (with ~{scenario.radar.measurement_noise_m:.0f}m noise)")
    print(f"  - {scenario.visualization.tracked_target_color.capitalize()} squares: Tracked aircraft with Kalman filter estimates")
    print(f"  - {scenario.visualization.velocity_vector_color.capitalize()} arrows: Velocity vectors ({scenario.visualization.velocity_vector_time_s:.0f}-second projection)")
    print(f"  - {scenario.visualization.illumination_color.capitalize()} bright cone: Current radar illumination beam (active)")
    print(f"  - {scenario.visualization.radar_sweep_color.capitalize()} sweep: Radar scanning history")
    print(f"\nAircraft velocities: {scenario.target.min_velocity_ms*3.6:.0f}-{scenario.target.max_velocity_ms*3.6:.0f} km/h")
    print("\nClose the window to exit.\n")

    metadata = build_run_metadata(mode="live", scenario=scenario, seed_override=seed)
    if metadata_out:
        write_run_metadata(metadata_out, metadata)
        print(f"Run metadata written to: {metadata_out}")

    # Create and run simulation with config parameters
    sim = LiveRadarSimulation(scenario=scenario)
    ani = sim.run()

    plt.show()


if __name__ == "__main__":
    main()
