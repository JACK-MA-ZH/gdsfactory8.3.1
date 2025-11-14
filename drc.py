from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image

try:  # pragma: no cover - optional dependency when running outside the agent stack
    from agent_r1.tool.base import BaseImageToolEnv, BaseTool
except ImportError:  # pragma: no cover - light-weight fallbacks for local testing
    class BaseTool:  # type: ignore[override]
        name = "base_tool"

        def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError("BaseTool.execute must be implemented by subclasses.")

    class BaseImageToolEnv:  # type: ignore[override]
        def __init__(self) -> None:
            pass

try:
    import gdsfactory as gf
    from gdsfactory.generic_tech import get_generic_pdk

    gf.config.rich_output()
    PDK = get_generic_pdk()
    PDK.activate()
    GDS_INSTALLED = True
    GDSComponent = gf.Component
except ImportError:  # pragma: no cover - defensive fallback
    gf = None
    GDS_INSTALLED = False
    GDSComponent = Any

if GDS_INSTALLED:
    from drc_tool import component_to_pil_image
else:  # pragma: no cover - fallback when gdsfactory missing
    component_to_pil_image = None


def _iter_references(comp: GDSComponent) -> Iterable[Any]:  # pragma: no cover - helper
    if hasattr(comp, "insts"):
        for inst in comp.insts:
            yield inst
    elif hasattr(comp, "references"):
        for ref in comp.references:
            yield ref


def _ensure_reference_names(comp: GDSComponent) -> None:
    if not hasattr(comp, "named_instances") or comp.named_instances is None:
        comp.named_instances = {}

    for index, reference in enumerate(_iter_references(comp)):
        if not getattr(reference, "name", None):
            reference.name = f"p{index}"
        comp.named_instances[reference.name] = reference


class DRCToolEnv(BaseImageToolEnv):
    """Environment that applies geometry tools to gdsfactory components."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        max_tool_response_length: int = 2048,
        data_root_dir: str | os.PathLike[str] = ".",
    ) -> None:
        super().__init__()
        self.tools: List[BaseTool] = list(tools)
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.max_tool_response_length = max_tool_response_length
        self.data_root_dir = Path(data_root_dir)

        self.batch_size: int = 0
        self.components: List[GDSComponent] = []
        self.op_counts: List[int] = []

    # ------------------------------------------------------------------
    # Environment lifecycle helpers
    # ------------------------------------------------------------------
    def reset(
        self,
        components: Sequence[GDSComponent] | None = None,
        gds_paths: Sequence[str] | None = None,
    ) -> None:
        if gf is None:
            raise RuntimeError("gdsfactory is required to load or manipulate GDS files.")

        loaded_components: List[GDSComponent] = []
        if gds_paths:
            for rel_path in gds_paths:
                abs_path = self.data_root_dir / rel_path
                if not abs_path.exists():
                    raise FileNotFoundError(f"GDS file not found at {abs_path}")
                component = gf.import_gds(
                    str(abs_path), rename_duplicated_cells=True
                )
                _ensure_reference_names(component)
                loaded_components.append(component)
        elif components:
            for component in components:
                _ensure_reference_names(component)
                loaded_components.append(component)
        else:
            raise ValueError("Must provide either 'components' or 'gds_paths'.")

        self.components = loaded_components
        self.batch_size = len(self.components)
        self.op_counts = [0] * self.batch_size

    # ------------------------------------------------------------------
    def get_drc_violations(self) -> List[Dict[str, Any]]:
        if gf is None:
            raise RuntimeError("gdsfactory is required to perform DRC checks.")

        try:  # Import lazily to keep klayout optional until a check is requested
            import klayout.db as kdb
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise RuntimeError(
                "klayout is required to run the DRC checks. Please install klayout."
            ) from exc

        def _bbox_to_tuple(box: Any, dbu: float) -> Tuple[float, float, float, float]:
            return (
                float(box.left) * dbu,
                float(box.bottom) * dbu,
                float(box.right) * dbu,
                float(box.top) * dbu,
            )

        min_spacing = 0.1
        min_width = 0.12

        batch_results: List[Dict[str, Any]] = []
        for component in self.components:
            if component is None:
                batch_results.append({
                    "count": 0,
                    "errors_text": "No component loaded.",
                    "errors_json": [],
                    "bboxes": [],
                    "component": component,
                })
                continue

            try:
                layout = component.kcl.layout
                dbu = float(getattr(layout, "dbu", 1.0) or 1.0)
                layer_index = int(layout.layer(1, 0))

                if layer_index < 0:
                    errors: List[Dict[str, Any]] = []
                else:
                    region = kdb.Region(component.kdb_cell.begin_shapes_rec(layer_index))
                    errors = []

                    spacing_pairs = list(region.space_check(min_spacing / dbu).each())
                    for pair in spacing_pairs:
                        bbox = _bbox_to_tuple(pair.bbox(), dbu)
                        errors.append({"type": "min_spacing", "bbox": bbox})

                    width_pairs = list(region.width_check(min_width / dbu).each())
                    for pair in width_pairs:
                        bbox = _bbox_to_tuple(pair.bbox(), dbu)
                        errors.append({"type": "min_width", "bbox": bbox})

                errors_text = (
                    "\n".join([f"ERROR: {e['type']} at {e['bbox']}" for e in errors])
                    if errors
                    else "No DRC errors found."
                )

                batch_results.append({
                    "count": len(errors),
                    "errors_text": errors_text,
                    "errors_json": errors,
                    "bboxes": [e["bbox"] for e in errors],
                    "component": component,
                })
            except Exception as exc:  # pragma: no cover - defensive
                batch_results.append({
                    "count": -1,
                    "errors_text": f"DRC check failed: {exc}",
                    "errors_json": [],
                    "bboxes": [],
                    "component": component,
                })
        return batch_results

    # ------------------------------------------------------------------
    def get_schematic(self, item_index: int) -> str:
        if item_index >= self.batch_size:
            raise IndexError(f"Index {item_index} is out of range for batch size {self.batch_size}.")
        component = self.components[item_index]
        if component is None:
            return "{}"
        try:
            return component.netlist()
        except AttributeError:
            return "{}"

    # ------------------------------------------------------------------
    def get_image(
        self,
        item_index: int,
        bbox: Tuple[float, float, float, float] | None = None,
    ) -> Image.Image:
        if item_index >= self.batch_size:
            raise IndexError(f"Index {item_index} is out of range for batch size {self.batch_size}.")

        component = self.components[item_index]
        if gf is None or component is None or component_to_pil_image is None:
            return Image.new("RGB", (100, 100), color="white")

        bbox_to_plot: Tuple[float, float, float, float] | None = None
        if bbox:
            dx = (bbox[2] - bbox[0]) * 0.5
            dy = (bbox[3] - bbox[1]) * 0.5
            bbox_to_plot = (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)

        return component_to_pil_image(
            component,
            title=f"component_{item_index}",
            bbox=bbox_to_plot,
        )

    # ------------------------------------------------------------------
    def step(
        self,
        raw_responses: Sequence[str] | str,
    ) -> Tuple[List[str], List[Image.Image], List[List[bool]], List[bool]]:
        if isinstance(raw_responses, str):
            raw_responses = [raw_responses]

        if self.batch_size == 0:
            raise RuntimeError("Environment has not been reset with any components.")

        if len(raw_responses) not in {1, self.batch_size}:
            raise ValueError(
                f"Expected {self.batch_size} responses (or 1 broadcast response) but received {len(raw_responses)}."
            )

        if len(raw_responses) == 1 and self.batch_size > 1:
            raw_responses = list(raw_responses) * self.batch_size

        batch_formatted_responses: List[str] = []
        batch_images: List[Image.Image] = []
        batch_successes: List[List[bool]] = []
        batch_active: List[bool] = []

        for index, raw_response in enumerate(raw_responses):
            component = self.components[index]
            tool_calls = self.extract_tool_calls(raw_response)

            if not tool_calls:
                batch_formatted_responses.append("")
                batch_images.append(self.get_image(index))
                batch_successes.append([])
                batch_active.append(False)
                continue

            tool_responses: List[str] = []
            tool_successes: List[bool] = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {}) or {}

                tool = self.tool_map.get(tool_name)
                if tool is None:
                    tool_responses.append(f"Error: Tool '{tool_name}' not found.")
                    tool_successes.append(False)
                    continue

                try:
                    result = tool.execute(args=tool_args, component=component)
                    response_text = result.get("content", json.dumps(result))
                    success = bool(result.get("success", True))
                except Exception as exc:  # pragma: no cover - runtime safety
                    response_text = f"Error executing tool '{tool_name}': {exc}"
                    success = False

                tool_responses.append(response_text)
                tool_successes.append(success)
                if success:
                    self.op_counts[index] += 1

            batch_formatted_responses.append(self.format_tool_response(tool_responses))
            batch_images.append(self.get_image(index))
            batch_successes.append(tool_successes)
            batch_active.append(True)

        return batch_formatted_responses, batch_images, batch_successes, batch_active

    # ------------------------------------------------------------------
    def stop(self, raw_responses: Sequence[str]) -> List[bool]:
        return ["<answer>" in response or not self.extract_tool_calls(response) for response in raw_responses]

    # ------------------------------------------------------------------
    def extract_tool_calls(self, raw_response: str) -> List[Dict[str, Any]]:
        try:
            match = re.search(r"<tool_code>(.*?)</tool_code>", raw_response, re.DOTALL)
            if not match:
                return []
            tool_calls_str = match.group(1).strip()
            tool_calls = json.loads(tool_calls_str)
            return tool_calls if isinstance(tool_calls, list) else [tool_calls]
        except (json.JSONDecodeError, AttributeError):
            return []

    # ------------------------------------------------------------------
    def format_tool_response(self, tool_responses: Sequence[str]) -> str:
        if not tool_responses:
            return ""
        response_data = json.dumps({"results": [{"content": resp} for resp in tool_responses]})
        if len(response_data) > self.max_tool_response_length:
            response_data = response_data[: self.max_tool_response_length] + "..."
        return f"<tool_response>{response_data}</tool_response>"


__all__ = ["DRCToolEnv", "GDS_INSTALLED"]
