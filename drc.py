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
    from drc_tool import component_to_pil_image, get_ref_shapes
else:  # pragma: no cover - fallback when gdsfactory missing
    component_to_pil_image = None
    get_ref_shapes = None


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


def _get_kdb_cell(component: GDSComponent) -> Any:
    """Return the underlying :class:`klayout.db.Cell` for ``component``."""

    if hasattr(component, "kdb_cell"):
        return component.kdb_cell
    if hasattr(component, "_kdb_cell"):
        return component._kdb_cell
    return getattr(component, "cell", None)


def _format_netlist_payload(component: GDSComponent) -> str | None:
    """Serialize ``component.netlist`` output to a JSON string when available."""

    try:
        netlist_payload = component.netlist()
    except AttributeError:
        return None

    if isinstance(netlist_payload, str):
        netlist_payload = netlist_payload.strip()
        return netlist_payload or None

    if isinstance(netlist_payload, dict):
        if not netlist_payload:
            return None
        return json.dumps(netlist_payload, ensure_ascii=False, indent=2, sort_keys=True)

    if netlist_payload:
        try:
            return json.dumps(netlist_payload, ensure_ascii=False, default=str)
        except TypeError:
            return str(netlist_payload)

    return None


def _reference_bbox(reference: Any) -> Dict[str, float] | None:
    """Return a JSON-serializable bbox description for ``reference``."""

    bbox_candidate = None
    bbox_method = getattr(reference, "bbox", None)
    if callable(bbox_method):
        try:
            bbox_candidate = bbox_method()
        except Exception:
            bbox_candidate = None
    elif bbox_method is not None:
        bbox_candidate = bbox_method

    if bbox_candidate is None:
        return None

    def _extract_value(obj: Any, *names: str) -> float | None:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    continue
        return None

    xmin = _extract_value(bbox_candidate, "xmin", "left")
    xmax = _extract_value(bbox_candidate, "xmax", "right")
    ymin = _extract_value(bbox_candidate, "ymin", "bottom")
    ymax = _extract_value(bbox_candidate, "ymax", "top")

    coords = {k: v for k, v in {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}.items() if v is not None}
    return coords or None


def _reference_center(reference: Any) -> Tuple[float, float] | None:
    center = getattr(reference, "center", None)
    if center is None:
        bbox = _reference_bbox(reference)
        if not bbox:
            return None
        try:
            cx = (bbox["xmin"] + bbox["xmax"]) / 2.0
            cy = (bbox["ymin"] + bbox["ymax"]) / 2.0
            return (cx, cy)
        except Exception:
            return None

    try:
        return (float(center[0]), float(center[1]))
    except Exception:
        return None


def _build_reference_snapshot(component: GDSComponent) -> Dict[str, Any]:
    """Create a lightweight JSON-serializable schematic for ``component``."""

    info = component.info if isinstance(getattr(component, "info", None), dict) else {}
    labels = info.get("polygon_labels") if isinstance(info, dict) else None
    if not isinstance(labels, list):
        labels = []

    label_lookup: Dict[str, Dict[str, Any]] = {}
    for entry in labels:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        label_lookup[str(name)] = {
            "layer": entry.get("layer_index"),
            "centroid": entry.get("centroid"),
        }

    named_instances = getattr(component, "named_instances", None)
    if isinstance(named_instances, dict) and named_instances:
        reference_iter: Iterable[Tuple[str | None, Any]] = named_instances.items()
    else:
        reference_iter = ((getattr(ref, "name", None), ref) for ref in _iter_references(component))

    references: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw_name, reference in reference_iter:
        if reference is None:
            continue
        ref_id = id(reference)
        if ref_id in seen_ids:
            continue
        seen_ids.add(ref_id)

        name = raw_name or getattr(reference, "name", None)
        if not name:
            name = f"ref_{len(references)}"

        bbox = _reference_bbox(reference)
        center = _reference_center(reference)
        references.append(
            {
                "name": str(name),
                "bbox": bbox,
                "center": center,
                "label": label_lookup.get(str(name)),
            }
        )

    snapshot = {
        "component": getattr(component, "name", "component"),
        "reference_count": len(references),
        "references": references,
    }

    if info:
        snapshot["info_keys"] = sorted(str(key) for key in info.keys())

    return snapshot


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
                    region = kdb.Region()
                    kdb_cell = _get_kdb_cell(component)
                    if kdb_cell is not None:
                        region += kdb.Region(kdb_cell.shapes(layer_index))

                    references = getattr(component, "named_instances", None)
                    if isinstance(references, dict) and references:
                        refs_iter: Iterable[Any] = references.values()
                    else:
                        refs_iter = _iter_references(component)

                    if get_ref_shapes is None:
                        raise RuntimeError("get_ref_shapes helper unavailable; check drc_tool import")

                    for ref in refs_iter:
                        try:
                            region += get_ref_shapes(ref, layer_index)
                        except Exception:
                            continue
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
        netlist_payload = _format_netlist_payload(component)
        if netlist_payload:
            return netlist_payload
        snapshot = _build_reference_snapshot(component)
        return json.dumps(snapshot, ensure_ascii=False)

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
