import argparse
from .scenario import list_scenarios
from .replay import compare_run_metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kalman-filter",
        description="Run Kalman radar simulation modes",
    )
    subparsers = parser.add_subparsers(dest="command")
    scenario_choices = list_scenarios()

    live_parser = subparsers.add_parser("live", help="Run live matplotlib simulation")
    live_parser.add_argument("--scenario", default="default", choices=scenario_choices)
    live_parser.add_argument("--seed", type=int, default=None, help="Override random seed for deterministic replay")
    live_parser.add_argument("--metadata-out", default=None, help="Write run metadata JSON to this path")

    web_parser = subparsers.add_parser("web", help="Run Dash web simulation")
    web_parser.add_argument("--scenario", default="default", choices=scenario_choices)
    web_parser.add_argument("--seed", type=int, default=None, help="Override random seed for deterministic replay")
    web_parser.add_argument("--metadata-out", default=None, help="Write run metadata JSON to this path")

    demo_parser = subparsers.add_parser("demo", help="Run static radar demo")
    demo_parser.add_argument("--scenario", default="default", choices=scenario_choices)
    demo_parser.add_argument("--seed", type=int, default=None, help="Override random seed for deterministic replay")
    demo_parser.add_argument("--metadata-out", default=None, help="Write run metadata JSON to this path")

    verify_parser = subparsers.add_parser("verify", help="Compare two replay metadata JSON files")
    verify_parser.add_argument("metadata_a", help="Path to first metadata JSON")
    verify_parser.add_argument("metadata_b", help="Path to second metadata JSON")

    args = parser.parse_args(argv)

    if args.command == "live":
        from .live_radar_sim import main as run_live

        run_live(scenario_name=args.scenario, seed=args.seed, metadata_out=args.metadata_out)
        return 0

    if args.command == "web":
        from .web_radar_sim import main as run_web

        run_web(scenario_name=args.scenario, seed=args.seed, metadata_out=args.metadata_out)
        return 0

    if args.command == "demo":
        from .radar_demo import main as run_demo

        run_demo(scenario_name=args.scenario, seed=args.seed, metadata_out=args.metadata_out)
        return 0

    if args.command == "verify":
        result = compare_run_metadata(args.metadata_a, args.metadata_b)
        if result["match"]:
            print("Replay metadata match: no differences found.")
            return 0

        print("Replay metadata drift detected:")
        for diff in result["differences"]:
            print(f"- {diff}")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
