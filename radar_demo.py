"""
Radar System Simulation with Kalman Filter Tracking

This demo simulates a radar system that detects moving objects and tracks them
using Kalman filters. Multiple targets can move with constant or accelerated motion.
"""

import numpy as np
import matplotlib
# Configure matplotlib backend for Windows
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from simulator import RadarSimulationArea
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def create_static_visualization(simulation_area, num_steps=500):
    """
    Run simulation and create static plots showing tracking performance.
    
    Args:
        simulation_area: RadarSimulationArea instance
        num_steps: Number of simulation steps to run
    """
    # Run simulation
    results = simulation_area.run_simulation(num_steps)
    
    times = results['times']
    true_positions = results['true_positions']
    estimated_positions = results['estimated_positions']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: 2D trajectory
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('2D Trajectories', fontsize=12, fontweight='bold')
    
    for i, true_pos in enumerate(true_positions):
        if true_pos:
            true_pos = np.array(true_pos)
            ax1.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label=f'True Target {i}' if i == 0 else '', alpha=0.7)
            ax1.plot(true_pos[0, 0], true_pos[0, 1], 'bo', markersize=8)  # Start
            ax1.plot(true_pos[-1, 0], true_pos[-1, 1], 'bs', markersize=10)  # End
    
    for i, est_pos in enumerate(estimated_positions):
        if est_pos:
            est_pos = np.array(est_pos)
            ax1.plot(est_pos[:, 0], est_pos[:, 1], 'r--', label=f'Estimated Track {i}' if i == 0 else '', alpha=0.7)
            ax1.plot(est_pos[0, 0], est_pos[0, 1], 'ro', markersize=6)
            ax1.plot(est_pos[-1, 0], est_pos[-1, 1], 'rs', markersize=8)
    
    # Draw radar range circle
    circle = Circle((0, 0), simulation_area.radar.range_limit, fill=False, 
                   linestyle='--', color='green', linewidth=2, label='Radar Range')
    ax1.add_patch(circle)
    ax1.plot(0, 0, 'g*', markersize=15, label='Radar Center')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-800, 800)
    ax1.set_ylim(-800, 800)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    
    # Plot 2: X position tracking error
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('X Position Tracking Error', fontsize=12, fontweight='bold')
    
    for i in range(min(len(true_positions), len(estimated_positions))):
        if true_positions[i] and estimated_positions[i]:
            true_x = np.array([p[0] for p in true_positions[i]])
            est_x = np.array([p[0] for p in estimated_positions[i][:len(true_x)]])
            
            if len(est_x) > 0:
                error = true_x[:len(est_x)] - est_x
                ax2.plot(error, label=f'Target {i}', alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Y position tracking error
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Y Position Tracking Error', fontsize=12, fontweight='bold')
    
    for i in range(min(len(true_positions), len(estimated_positions))):
        if true_positions[i] and estimated_positions[i]:
            true_y = np.array([p[1] for p in true_positions[i]])
            est_y = np.array([p[1] for p in estimated_positions[i][:len(true_y)]])
            
            if len(est_y) > 0:
                error = true_y[:len(est_y)] - est_y
                ax3.plot(error, label=f'Target {i}', alpha=0.7)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Tracking statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = "Simulation Statistics\n" + "="*30 + "\n\n"
    stats_text += f"Total Targets: {len(simulation_area.targets)}\n"
    stats_text += f"Active Tracks: {len(simulation_area.tracker.get_confirmed_tracks())}\n"
    stats_text += f"Total Tracks Created: {simulation_area.tracker.next_id}\n"
    stats_text += f"Simulation Time: {simulation_area.time:.1f}s\n"
    stats_text += f"Radar Range: {simulation_area.radar.range_limit}m\n"
    stats_text += f"Measurement Noise σ: {simulation_area.radar.measurement_noise}m\n"
    stats_text += f"Detection Probability: {simulation_area.radar.detection_probability*100:.0f}%\n"
    
    stats_text += "\n\nTrack Information:\n" + "-"*30 + "\n"
    for track in simulation_area.tracker.get_all_tracks():
        stats_text += f"Track ID {track.id}:\n"
        stats_text += f"  Measurements: {track.measurements_received}\n"
        stats_text += f"  Age: {track.time_since_update:.2f}s\n"
        stats_text += f"  Pos: ({track.get_estimated_position()[0]:.1f}, {track.get_estimated_position()[1]:.1f})\n"
        stats_text += f"  Vel: ({track.get_estimated_velocity()[0]:.2f}, {track.get_estimated_velocity()[1]:.2f})\n\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """Main simulation runner."""
    print("="*60)
    print("Radar System Simulation with Kalman Filter Tracking")
    print("="*60)
    
    # Create simulation area
    area = RadarSimulationArea(width=1000, height=1000)
    
    # Add targets with different motion profiles
    print("\nAdding targets:")
    
    # Target 1: Constant velocity motion
    target1_id = area.add_target(x=-400, y=-300, vx=100, vy=80, ax=0, ay=0)
    print(f"  Target 1: Constant velocity (-400, -300) -> (100, 80) m/s")
    
    # Target 2: Accelerating motion
    target2_id = area.add_target(x=300, y=-200, vx=-50, vy=50, ax=10, ay=5)
    print(f"  Target 2: Accelerating motion (300, -200) -> (-50, 50) m/s with acceleration")
    
    # Target 3: Another constant velocity target
    target3_id = area.add_target(x=0, y=400, vx=0, vy=-100, ax=0, ay=0)
    print(f"  Target 3: Constant velocity (0, 400) -> (0, -100) m/s")
    
    print(f"\nRadar Configuration:")
    print(f"  Range: {area.radar.range_limit}m")
    print(f"  Measurement Noise σ: {area.radar.measurement_noise}m")
    print(f"  Detection Probability: {area.radar.detection_probability*100:.0f}%")
    
    print(f"\nTracker Configuration:")
    print(f"  Max Track Age: {area.tracker.max_age}s")
    print(f"  Min Hits to Confirm: {area.tracker.min_hits}")
    print(f"  Association Threshold: {area.tracker.association_threshold}m")
    
    # Run simulation
    print("\nRunning simulation...")
    num_steps = 500
    fig = create_static_visualization(area, num_steps)
    
    print(f"Simulation completed: {area.time:.1f} seconds")
    print(f"Total active tracks: {len(area.tracker.get_confirmed_tracks())}")
    
    print("\nDisplaying results...")
    plt.show()


if __name__ == "__main__":
    main()
