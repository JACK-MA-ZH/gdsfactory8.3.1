"""Comprehensive integration test for the DRC environment and tools."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

from drc import DRCToolEnv
from drc_data_preprocess import create_drc_dataset
from drc_tool import (
    DeletePolygonTool,
    MovePolygonTool,
    OffsetPolygonTool,
    SplitPolygonTool,
)

MOVE_DX = -0.12
MOVE_DY = 0.0
OFFSET_DISTANCE = 0.04


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


def _bbox_dimensions_um(reference, dbu: float) -> Tuple[float, float]:
    bbox = reference.bbox()
    width = (bbox.right - bbox.left) * dbu
    height = (bbox.top - bbox.bottom) * dbu
    return float(width), float(height)


def _log_demo_references(component, stage: str) -> None:
    named_instances = getattr(component, "named_instances", None)
    if isinstance(named_instances, dict):
        names = [name for name, ref in named_instances.items() if ref is not None]
    else:
        references = getattr(component, "references", [])
        names = [getattr(ref, "name", "") for ref in references if getattr(ref, "name", None)]
    print(f"[drc_flow_demo] {stage} active references: {sorted(names)}")


def _execute_tool(
    env: DRCToolEnv,
    tool_calls: list[dict],
    *,
    tool_name: str,
    artifact_label: str,
    artifacts_dir: Path,
):
    formatted_calls = _format_tool_calls(tool_calls)
    responses, images, successes, active = env.step(formatted_calls)
    _assert_true(active[0], f"Environment reported inactive batch entry during {tool_name}")
    _assert_tool_success(successes[0], tool_name)
    print(f"{tool_name} response:", responses[0])
    _save_image(images[0], artifacts_dir, artifact_label)
    return responses[0]


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
        tools=[
            SplitPolygonTool(),
            MovePolygonTool(),
            OffsetPolygonTool(),
            DeletePolygonTool(),
        ],
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
    _execute_tool(
        env,
        split_calls,
        tool_name="SplitPolygonTool",
        artifact_label="01_after_split",
        artifacts_dir=artifacts_dir,
    )

    component = env.components[0]
    _log_demo_references(component, "after split")
    named_instances = getattr(component, "named_instances", None)
    _assert_true(isinstance(named_instances, dict), "Component missing named reference map")
    dbu = component.kcl.dbu
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
    _execute_tool(
        env,
        move_calls,
        tool_name="MovePolygonTool",
        artifact_label="02_after_move",
        artifacts_dir=artifacts_dir,
    )
    _log_demo_references(component, "after move")

    post_move_center = tuple(named_instances["p1_part1"].center)
    dx_applied = post_move_center[0] - pre_move_center[0]
    dy_applied = post_move_center[1] - pre_move_center[1]
    _assert_true(
        math.isclose(dx_applied, MOVE_DX, rel_tol=1e-6, abs_tol=1e-6)
        and math.isclose(dy_applied, MOVE_DY, rel_tol=1e-6, abs_tol=1e-6),
        f"MovePolygonTool delta mismatch: expected ({MOVE_DX}, {MOVE_DY}) got ({dx_applied}, {dy_applied})",
    )

    pre_offset_width, pre_offset_height = _bbox_dimensions_um(named_instances["p1_part2"], dbu)
    offset_calls = [
        {
            "name": "op_offset_polygon",
            "arguments": {
                "polygon_name": "p1_part2",
                "distance": OFFSET_DISTANCE,
                "layer": [1, 0],
            },
        }
    ]
    print("Testing OffsetPolygonTool...")
    _execute_tool(
        env,
        offset_calls,
        tool_name="OffsetPolygonTool",
        artifact_label="03_after_offset",
        artifacts_dir=artifacts_dir,
    )
    _log_demo_references(component, "after offset")

    post_offset_width, post_offset_height = _bbox_dimensions_um(named_instances["p1_part2"], dbu)
    expected_delta = 2.0 * OFFSET_DISTANCE
    width_delta = post_offset_width - pre_offset_width
    height_delta = post_offset_height - pre_offset_height
    _assert_true(
        math.isclose(width_delta, expected_delta, rel_tol=1e-6, abs_tol=5e-3)
        and math.isclose(height_delta, expected_delta, rel_tol=1e-6, abs_tol=5e-3),
        (
            "OffsetPolygonTool did not expand geometry by the expected amount: "
            f"Δw={width_delta}, Δh={height_delta}, expected {expected_delta}"
        ),
    )

    print("Running DRC check after move/offset operations...")
    mid_drc = env.get_drc_violations()[0]
    _assert_true(mid_drc["count"] > 0, "Expected DRC violations after move/offset operations")
    print("Intermediate DRC errors detected:")
    print(mid_drc["errors_text"])

    delete_calls = [
        {"name": "op_delete_polygon", "arguments": {"polygon_name": "p1_part2"}}
    ]
    print("Testing DeletePolygonTool...")
    _execute_tool(
        env,
        delete_calls,
        tool_name="DeletePolygonTool",
        artifact_label="04_after_delete",
        artifacts_dir=artifacts_dir,
    )
    _log_demo_references(component, "after delete")

    _assert_true(
        "p1_part2" not in named_instances,
        "DeletePolygonTool failed to remove the target reference",
    )

    print("Running final DRC check after delete...")
    final_drc = env.get_drc_violations()[0]
    _assert_true(final_drc["count"] == 0, "Expected clean layout after delete operation")
    print("Final DRC errors text (should be empty):")
    print(final_drc["errors_text"])

    _save_image(env.get_image(0), artifacts_dir, "05_final_layout")
    print("All tool tests completed successfully.")


if __name__ == "__main__":
    main()
