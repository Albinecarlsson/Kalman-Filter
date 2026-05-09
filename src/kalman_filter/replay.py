import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scenario import Scenario


def build_run_metadata(mode: str, scenario: Scenario, seed_override: Optional[int] = None) -> Dict[str, Any]:
    return {
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.name,
        "seed": scenario.simulation.random_seed,
        "seed_override": seed_override,
        "config": asdict(scenario),
    }


def write_run_metadata(path: str, metadata: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_run_metadata(path: str) -> Dict[str, Any]:
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _diff_values(left: Any, right: Any, path: str, diffs: List[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        for missing in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{missing}: present in A, missing in B")
        for missing in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{missing}: missing in A, present in B")
        for key in sorted(left_keys & right_keys):
            child_path = f"{path}.{key}" if path else key
            _diff_values(left[key], right[key], child_path, diffs)
        return

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            diffs.append(f"{path}: list length {len(left)} != {len(right)}")
            return
        for index, (lval, rval) in enumerate(zip(left, right)):
            _diff_values(lval, rval, f"{path}[{index}]", diffs)
        return

    if left != right:
        diffs.append(f"{path}: {left!r} != {right!r}")


def compare_run_metadata(path_a: str, path_b: str) -> Dict[str, Any]:
    meta_a = read_run_metadata(path_a)
    meta_b = read_run_metadata(path_b)
    diffs: List[str] = []
    _diff_values(meta_a, meta_b, "", diffs)

    return {
        "match": len(diffs) == 0,
        "differences": diffs,
        "path_a": str(path_a),
        "path_b": str(path_b),
    }
