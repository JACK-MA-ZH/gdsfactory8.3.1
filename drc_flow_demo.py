"""Comprehensive integration test for the DRC environment and tools."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image

from drc import DRCToolEnv
from drc_data_preprocess import create_drc_dataset
from drc_tool import MovePolygonTool, SplitPolygonTool

MOVE_DX = -0.08
MOVE_DY = 0.0


def _format_tool_calls(calls: list[dict]) -> str:
    return f"<tool_code>{json.dumps(calls)}</tool_code>"


def _save_image(image: Image.Image, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{label}.png"
    image.save(file_path)
    print(f"Saved image '{label}' to {file_path.resolve()}")
    return file_path


def _assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_tool_success(successes: Iterable[bool], tool_name: str) -> None:
    flags = list(successes)
    _assert_true(flags, f"{tool_name} did not run (no success flags recorded)")
    _assert_true(all(flags), f"{tool_name} reported failure: {flags}")


def main() -> None:
    data_root = Path("demo_drc_data")
    if data_root.exists():
        shutil.rmtree(data_root)

    print("Creating demo dataset...")
    dataset = create_drc_dataset(str(data_root), num_samples=1, split="demo")
    _assert_true(len(dataset) == 1, "Expected dataset to contain exactly one record")
    record = dataset[0]
    gds_rel_path = record["initial_gds_path"]

    artifacts_dir = data_root / "demo_artifacts"

    env = DRCToolEnv(
        tools=[SplitPolygonTool(), MovePolygonTool()],
        max_tool_response_length=2048,
        data_root_dir=str(data_root),
    )
    env.reset(gds_paths=[gds_rel_path])

    print("Running initial DRC check...")
    initial_drc = env.get_drc_violations()[0]
    _assert_true(initial_drc["count"] == 0, f"Expected clean layout, found: {initial_drc['errors_text']}")

    _save_image(env.get_image(0), artifacts_dir, "00_initial_layout")

    split_calls = [
        {
            "name": "op_split_polygon",
            "arguments": {
                "polygon_name": "p1",
                "split_line": {"axis": "x", "value": 0.55},
                "layer": [1, 0],
            },
        }
    ]
    print("Testing SplitPolygonTool...")
    formatted_split = _format_tool_calls(split_calls)
    split_responses, split_images, split_successes, split_active = env.step(formatted_split)
    _assert_true(split_active[0], "Environment reported inactive batch entry during split test")
    _assert_tool_success(split_successes[0], "SplitPolygonTool")
    print("Split response:", split_responses[0])
    _save_image(split_images[0], artifacts_dir, "01_after_split")

    component = env.components[0]
    named_instances = getattr(component, "named_instances", {}) or {}
    _assert_true("p1" not in named_instances, "Original polygon reference still present after split")
    for part_name in ("p1_part1", "p1_part2"):
        _assert_true(part_name in named_instances, f"Missing split output '{part_name}'")

    pre_move_center = tuple(named_instances["p1_part1"].center)

    move_calls = [
        {
            "name": "op_move_polygon",
            "arguments": {
                "polygon_name": "p1_part1",
                "dx": MOVE_DX,
                "dy": MOVE_DY,
            },
        }
    ]
    print("Testing MovePolygonTool...")
    formatted_move = _format_tool_calls(move_calls)
    move_responses, move_images, move_successes, move_active = env.step(formatted_move)
    _assert_true(move_active[0], "Environment reported inactive batch entry during move test")
    _assert_tool_success(move_successes[0], "MovePolygonTool")
    print("Move response:", move_responses[0])
    _save_image(move_images[0], artifacts_dir, "02_after_move")

    post_move_center = tuple(named_instances["p1_part1"].center)
    dx_applied = post_move_center[0] - pre_move_center[0]
    dy_applied = post_move_center[1] - pre_move_center[1]
    _assert_true(
        math.isclose(dx_applied, MOVE_DX, rel_tol=1e-6, abs_tol=1e-6)
        and math.isclose(dy_applied, MOVE_DY, rel_tol=1e-6, abs_tol=1e-6),
        f"MovePolygonTool delta mismatch: expected ({MOVE_DX}, {MOVE_DY}) got ({dx_applied}, {dy_applied})",
    )

    print("Running DRC check after tool operations...")
    final_drc = env.get_drc_violations()[0]
    _assert_true(
        final_drc["count"] > 0,
        "Expected at least one DRC violation after split and move operations",
    )
    print("DRC errors detected:")
    print(final_drc["errors_text"])

    _save_image(env.get_image(0), artifacts_dir, "03_final_layout")
    if final_drc["bboxes"]:
        violation_zoom = env.get_image(0, bbox=final_drc["bboxes"][0])
        _save_image(violation_zoom, artifacts_dir, "04_violation_zoom")

    print("All tool tests completed successfully.")


if __name__ == "__main__":
    main()
