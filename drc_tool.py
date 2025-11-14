from __future__ import annotations

from typing import Dict, Any, List
import sys
sys.path.append('.')

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

# REPO_ROOT = Path(__file__).resolve().parents[2]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.append(str(REPO_ROOT))

import gdsfactory as gf
from gdsfactory import get_layer
from gdsfactory.boolean import get_ref_shapes
from gdsfactory.typings import component
import klayout.db as kdb

try:  # pragma: no cover - optional dependency when running outside the agent stack
    from agent_r1.tool.base import BaseTool
except ImportError:  # pragma: no cover - simple fallback for local testing
    class BaseTool:  # type: ignore[override]
        name = "base_tool"

        def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError


_COLOR_CYCLE = plt.rcParams.get("axes.prop_cycle", None)
POLYGON_LABELS_KEY = "polygon_labels"


def _iter_references(comp: component) -> Iterable[gf.ComponentReference]:
    """Yield component references stored on the component."""

    if hasattr(comp, "insts"):
        for inst in comp.insts:
            yield inst
    elif hasattr(comp, "references"):
        for ref in comp.references:
            yield ref


def _ensure_named_instance_map(component: component) -> Dict[str, gf.ComponentReference]:
    if not hasattr(component, "named_instances") or component.named_instances is None:
        component.named_instances = {}
    return component.named_instances


def _register_reference_name(component: component, name: str, reference: gf.ComponentReference) -> None:
    mapping = _ensure_named_instance_map(component)
    mapping[name] = reference


def _remove_reference_name(component: component, name: str) -> None:
    mapping = _ensure_named_instance_map(component)
    mapping.pop(name, None)


def _remove_reference(component: component, reference: gf.ComponentReference) -> None:
    if hasattr(component, "insts") and reference in component.insts:
        component.insts.remove(reference)
    elif hasattr(component, "references") and reference in component.references:
        component.references.remove(reference)
    else:
        raise ValueError("Component reference not found; cannot remove.")


def polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Return the centroid of a polygon described by ``points``."""

    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        raise ValueError("Cannot compute centroid of empty polygon.")

    x = pts[:, 0]
    y = pts[:, 1]
    area_accumulator = 0.0
    cx_accumulator = 0.0
    cy_accumulator = 0.0

    for idx in range(len(pts)):
        jdx = (idx + 1) % len(pts)
        cross = x[idx] * y[jdx] - x[jdx] * y[idx]
        area_accumulator += cross
        cx_accumulator += (x[idx] + x[jdx]) * cross
        cy_accumulator += (y[idx] + y[jdx]) * cross

    area = area_accumulator / 2.0
    if abs(area) < 1e-9:
        centroid = pts.mean(axis=0)
        return float(centroid[0]), float(centroid[1])

    cx = cx_accumulator / (6.0 * area)
    cy = cy_accumulator / (6.0 * area)
    return float(cx), float(cy)


def _build_label_lookup(component_to_plot: component) -> List[Dict[str, object]]:
    labels = getattr(component_to_plot, "info", {}).get(POLYGON_LABELS_KEY, [])
    if isinstance(labels, Iterable):
        return [dict(entry) for entry in labels]
    return []


def _find_polygon_label(
    label_entries: List[Dict[str, object]],
    layer_index: int,
    centroid: Tuple[float, float],
) -> str | None:
    for entry in label_entries:
        if entry.get("layer_index") != layer_index:
            continue
        stored_centroid = entry.get("centroid")
        if stored_centroid is None:
            continue
        stored_centroid = np.asarray(stored_centroid, dtype=float)
        if np.allclose(stored_centroid, np.asarray(centroid, dtype=float), atol=1e-6):
            candidate = entry.get("name")
            if candidate:
                return str(candidate)
    return None


def _build_reference_centroids(component_to_plot: component) -> Dict[str, Tuple[float, float]]:
    centroids: Dict[str, Tuple[float, float]] = {}
    named_instances = getattr(component_to_plot, "named_instances", None)
    if not isinstance(named_instances, dict):
        return centroids

    for name, reference in named_instances.items():
        if not reference:
            continue
        try:
            center = reference.center
        except Exception:
            try:
                bbox = reference.bbox
                center = ((bbox.xmin + bbox.xmax) / 2.0, (bbox.ymin + bbox.ymax) / 2.0)
            except Exception:
                continue
        centroids[name] = (float(center[0]), float(center[1]))
    return centroids


def _match_reference_name(
    reference_centroids: Dict[str, Tuple[float, float]],
    centroid: Tuple[float, float],
) -> str | None:
    if not reference_centroids:
        return None

    target = np.asarray(centroid, dtype=float)
    best_name: str | None = None
    best_distance = float("inf")
    for name, ref_centroid in reference_centroids.items():
        diff = target - np.asarray(ref_centroid, dtype=float)
        distance = float(np.dot(diff, diff))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


@dataclass(slots=True)
class _ReferenceMatch:
    reference: gf.ComponentReference
    index: int


def _find_reference_by_name(comp: component, name: str) -> _ReferenceMatch:
    mapping = getattr(comp, "named_instances", None)
    if isinstance(mapping, dict) and name in mapping:
        reference = mapping[name]
        return _ReferenceMatch(reference=reference, index=-1)

    for index, ref in enumerate(_iter_references(comp)):
        if ref.name == name:
            return _ReferenceMatch(reference=ref, index=index)
    raise ValueError(f"Component does not contain a reference named {name!r}.")


def plot_with_labels_and_vertices(
    component_to_plot: component,
    title: str,
    *,
    bbox: Tuple[float, float, float, float] | None = None,
):
    """Plot component polygons, annotating vertices and polygon names."""

    polygons_by_layer = component_to_plot.get_polygons_points()
    fig, ax = plt.subplots()
    colors = (_COLOR_CYCLE.by_key()["color"] if _COLOR_CYCLE else ["tab:blue"])
    color_count = len(colors)
    color_index = 0

    label_entries = _build_label_lookup(component_to_plot)
    reference_centroids = _build_reference_centroids(component_to_plot)

    for layer_index, polygons in polygons_by_layer.items():
        for poly in polygons:
            color = colors[color_index % color_count]
            color_index += 1

            points_array = np.asarray(poly, dtype=float)
            patch = Polygon(
                points_array,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.2,
                linewidth=1.0,
            )
            ax.add_patch(patch)

            for x, y in points_array:
                ax.text(
                    x,
                    y,
                    f"({x:.3f}, {y:.3f})",
                    fontsize=7,
                    color="red",
                    ha="left",
                    va="bottom",
                )

            centroid = polygon_centroid(points_array)
            label = _find_polygon_label(label_entries, int(layer_index), centroid)
            if label is None:
                label = _match_reference_name(reference_centroids, centroid)
            if label:
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="blue",
                    weight="bold",
                )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_facecolor("white")

    if bbox:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    else:
        ax.autoscale()
    return fig, ax


def component_to_pil_image(
    component_to_plot: component,
    *,
    title: str = "layout",
    bbox: Tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Render ``component_to_plot`` to a PIL image with polygon annotations."""

    fig, _ = plot_with_labels_and_vertices(component_to_plot, title, bbox=bbox)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = Image.frombuffer(
        "RGBA",
        (width, height),
        canvas.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1,
    )
    image = image.copy()
    plt.close(fig)
    return image


class DRCBaseTool(BaseTool):
    """Base class for tools that operate on gdsfactory components."""

    def execute(
        self, args: Dict[str, Any], component: component | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if component is None:
            raise ValueError("A component instance must be provided to execute the tool.")
        if not isinstance(args, dict):
            raise TypeError("Tool arguments must be provided as a dictionary.")

        result = self._execute(args, component)
        if result is None:
            result = {}
        return {"success": True, "status": "success", **result}

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        raise NotImplementedError

    @staticmethod
    def _get_reference(component: component, name: str) -> gf.ComponentReference:
        return _find_reference_by_name(component, name).reference


class MovePolygonTool(DRCBaseTool):
    """Move a polygon (component reference) by a delta."""

    name = "op_move_polygon"
    description = "通过给定的增量 (dx, dy) 移动布局中的指定“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要移动的‘多边形’（实例）的名称。"},
            "dx": {"type": "number", "description": "x 方向的移动距离 (um)。"},
            "dy": {"type": "number", "description": "y 方向的移动距离 (um)。"},
        },
        "required": ["polygon_name", "dx", "dy"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        dx = float(args["dx"])
        dy = float(args["dy"])

        reference = self._get_reference(component, polygon_name)
        reference.dmove((dx, dy))
        return {
            "content": f"Moved polygon {polygon_name} by dx={dx}, dy={dy}.",
            "polygon": polygon_name,
            "dx": dx,
            "dy": dy,
        }


class DeletePolygonTool(DRCBaseTool):
    """Delete a polygon (component reference) from the component."""

    name = "op_delete_polygon"
    description = "从布局中删除指定的“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要删除的‘多边形’（实例）的名称。"}
        },
        "required": ["polygon_name"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        reference_match = _find_reference_by_name(component, polygon_name)
        _remove_reference(component, reference_match.reference)
        _remove_reference_name(component, polygon_name)
        return {"content": f"Deleted polygon {polygon_name}."}


class OffsetPolygonTool(DRCBaseTool):
    """Offset a polygon by a specific distance."""

    name = "op_offset_polygon"
    description = "通过给定距离偏移（收缩或增长）指定的“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要偏移的‘多边形’（实例）的名称。"},
            "distance": {
                "type": "number",
                "description": "偏移距离 (um)。负数表示收缩，正数表示增长。",
            },
            "layer": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "GDS Layer [layer, purpose] 列表 (例如 [1, 0])。",
            },
        },
        "required": ["polygon_name", "distance", "layer"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        distance = float(args["distance"])
        layer_tuple = tuple(args["layer"])
        if len(layer_tuple) != 2:
            raise ValueError("Layer must be specified as [layer, purpose].")

        reference = self._get_reference(component, polygon_name)
        distance_dbu = component.kcl.to_dbu(distance)

        layer_index = get_layer(layer_tuple)
        region = get_ref_shapes(reference, layer_index)
        region = region.size(distance_dbu)

        _remove_reference(component, reference)
        _remove_reference_name(component, polygon_name)

        new_component = gf.Component(name=f"{polygon_name}_offset")
        new_component.kdb_cell.shapes(layer_index).insert(region)

        new_reference = component.add_ref(new_component, name=polygon_name)
        _register_reference_name(component, polygon_name, new_reference)
        return {
            "content": f"Offset polygon {polygon_name} by distance {distance} on layer {layer_tuple}.",
            "polygon": polygon_name,
            "distance": distance,
        }


class SplitPolygonTool(DRCBaseTool):
    """Split a polygon using an infinite axis-aligned line."""

    name = "op_split_polygon"
    description = (
        "使用一条无限长且与坐标轴平行的直线 (x=value 或 y=value) 来分割一个“多边形”（实例）。"
        "原始“多边形”被替换为由该直线切割得到的两个新的“多边形”。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要分割的‘多边形’（实例）的名称。"},
            "split_line": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["x", "y"],
                        "description": "决定分割直线是 x=value 还是 y=value。",
                    },
                    "value": {"type": "number", "description": "直线常数值 (um)。"},
                },
                "required": ["axis", "value"],
                "description": "定义分割直线的轴向与常数值。",
            },
            "layer": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "GDS Layer [layer, purpose] 列表 (例如 [1, 0])。",
            },
        },
        "required": ["polygon_name", "split_line", "layer"],
    }

    @staticmethod
    def _build_half_plane_masks(
        reference: gf.ComponentReference,
        axis: str,
        value: float,
        layer: tuple[int, int],
    ) -> tuple[gf.Component, gf.Component]:
        bbox = reference.bbox()
        xmin = float(bbox.left)
        xmax = float(bbox.right)
        ymin = float(bbox.bottom)
        ymax = float(bbox.top)
        span_x = abs(xmax - xmin)
        span_y = abs(ymax - ymin)
        margin = max(span_x, span_y, 1.0) * 2.0

        low = gf.Component(name="split_half_low")
        high = gf.Component(name="split_half_high")

        if axis == "x":
            x_low = min(xmin - margin, value - margin)
            x_high = max(xmax + margin, value + margin)
            y_lo = ymin - margin
            y_hi = ymax + margin
            low.add_polygon(
                [(x_low, y_lo), (value, y_lo), (value, y_hi), (x_low, y_hi)],
                layer=layer,
            )
            high.add_polygon(
                [(value, y_lo), (x_high, y_lo), (x_high, y_hi), (value, y_hi)],
                layer=layer,
            )
        else:  # axis == "y"
            y_low = min(ymin - margin, value - margin)
            y_high = max(ymax + margin, value + margin)
            x_lo = xmin - margin
            x_hi = xmax + margin
            low.add_polygon(
                [(x_lo, y_low), (x_hi, y_low), (x_hi, value), (x_lo, value)],
                layer=layer,
            )
            high.add_polygon(
                [(x_lo, value), (x_hi, value), (x_hi, y_high), (x_lo, y_high)],
                layer=layer,
            )

        return low, high

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        split_line = args["split_line"]
        axis = split_line.get("axis")
        if axis not in {"x", "y"}:
            raise ValueError("split_line.axis must be either 'x' or 'y'.")
        try:
            value = float(split_line["value"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("split_line.value must be a valid number.") from exc

        layer_tuple = tuple(args["layer"])
        if len(layer_tuple) != 2:
            raise ValueError("Layer must be specified as [layer, purpose].")

        reference = self._get_reference(component, polygon_name)

        low_mask, high_mask = self._build_half_plane_masks(reference, axis, value, layer_tuple)
        first_half = gf.boolean(reference, low_mask, operation="and", layer=layer_tuple)
        second_half = gf.boolean(reference, high_mask, operation="and", layer=layer_tuple)

        _remove_reference(component, reference)
        _remove_reference_name(component, polygon_name)

        new_refs: List[str] = []
        if first_half.get_polygons():
            ref_first = component.add_ref(first_half, name=f"{polygon_name}_part1")
            _register_reference_name(component, ref_first.name, ref_first)
            new_refs.append(ref_first.name)
        if second_half.get_polygons():
            ref_second = component.add_ref(second_half, name=f"{polygon_name}_part2")
            _register_reference_name(component, ref_second.name, ref_second)
            new_refs.append(ref_second.name)

        return {
            "content": (
                f"Split polygon {polygon_name} with {axis}={value} into {', '.join(new_refs)}."
            ),
            "original_polygon": polygon_name,
            "new_references": new_refs,
            "split_axis": axis,
            "split_value": value,
        }


def _create_demo_component() -> component:
    """Create a demo component with a single rectangle reference."""

    demo = gf.Component("drc_demo")
    rectangle = gf.components.rectangle(size=(40.0, 20.0), layer=(1, 0))
    demo.add_ref(rectangle, name="demo_rect")
    return demo


def _run_tool_and_plot(
    tool: DRCBaseTool,
    args: Dict[str, Any],
    comp: component,
    title: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute a tool, plot the result, and persist the figure."""

    result = tool.execute(args=args, component=comp)
    fig, _ax = plot_with_labels_and_vertices(comp, title=title)
    output_path = output_dir / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tool.name}] {title}: {result}")
    print(f"Saved plot to: {output_path.resolve()}")
    return result


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output_directory = Path("drc_demo_outputs")
    output_directory.mkdir(parents=True, exist_ok=True)

    demo_component = _create_demo_component()
    fig, _ = plot_with_labels_and_vertices(demo_component, "Initial component state")
    initial_path = output_directory / "initial_component_state.png"
    fig.savefig(initial_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {initial_path.resolve()}")

    move_tool = MovePolygonTool()
    _run_tool_and_plot(
        move_tool,
        {"polygon_name": "demo_rect", "dx": 10.0, "dy": 5.0},
        demo_component,
        "After move",
        output_directory,
    )

    offset_tool = OffsetPolygonTool()
    _run_tool_and_plot(
        offset_tool,
        {"polygon_name": "demo_rect", "distance": 2.0, "layer": [1, 0]},
        demo_component,
        "After offset",
        output_directory,
    )

    split_tool = SplitPolygonTool()
    split_result = _run_tool_and_plot(
        split_tool,
        {
            "polygon_name": "demo_rect",
            "split_line": {"axis": "y", "value": 12.5},
            "layer": [1, 0],
        },
        demo_component,
        "After split",
        output_directory,
    )

    new_refs = split_result.get("new_references", [])
    if new_refs:
        delete_tool = DeletePolygonTool()
        _run_tool_and_plot(
            delete_tool,
            {"polygon_name": new_refs[0]},
            demo_component,
            "After delete",
            output_directory,
        )

    print(
        "Demo complete. Generated plots demonstrate each tool operation on the sample"
        " component."
    )