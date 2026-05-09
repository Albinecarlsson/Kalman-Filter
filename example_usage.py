"""Example usage for immutable scenario profiles."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kalman_filter.scenario import list_scenarios
from live_radar_sim import main


print("AVAILABLE SCENARIOS")
print("-" * 70)
for scenario in list_scenarios():
    print(f"- {scenario}")

print("\nRUNNING DEFAULT SCENARIO")
print("-" * 70)
main("default")

# Examples:
# main("high_traffic")
# main("low_noise")
# main("long_range")
