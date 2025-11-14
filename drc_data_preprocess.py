"""Script to build a DRC dataset with polygon name annotations."""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from typing import Dict, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency for HuggingFace datasets
    import datasets
except ImportError:
    datasets = None

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drc_tool import POLYGON_LABELS_KEY, polygon_centroid, plot_with_labels_and_vertices

POLYGON_NAME_PROPERTY_ID = 1


class _FallbackDataset(list):
    """Light-weight dataset replacement used when :mod:`datasets` is unavailable."""

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "_FallbackDataset":
        return cls(df.to_dict(orient="records"))

    def to_parquet(self, path: str) -> None:
        _write_dataframe_with_fallback(pd.DataFrame(list(self)), path)


def _ensure_named_instance_maps(
    component: gf.Component,
) -> Tuple[Dict[str, gf.Component], Dict[str, gf.ComponentReference]]:
    """Ensure ``component`` exposes dictionaries for named references and instances."""

    if not hasattr(component, "named_references") or component.named_references is None:
        component.named_references = {}
    if not hasattr(component, "named_instances") or component.named_instances is None:
        component.named_instances = {}
    return component.named_references, component.named_instances


def add_named_polygon(
    component: gf.Component,
    points: Sequence[Sequence[float]],
    *,
    layer: Tuple[int, int] = (1, 0),
    name: str,
) -> gf.Component:
    """Add a polygon reference to ``component`` and remember its name for plotting."""

    polygon_component = gf.Component()
    polygon_component.add_polygon(points, layer=layer)
    reference = component.add_ref(polygon_component, name=name)

    named_refs, named_instances = _ensure_named_instance_maps(component)
    named_refs[name] = polygon_component
    named_instances[name] = reference

    points_array = np.asarray(points, dtype=float)
    centroid = polygon_centroid(points_array)

    # Remember the polygon label so we can annotate during plotting.
    labels: List[Dict[str, object]] = list(component.info.get(POLYGON_LABELS_KEY, []))
    layer_index = int(component.kcl.layout.layer(layer[0], layer[1]))
    labels.append(
        {
            "name": name,
            "layer_index": layer_index,
            "centroid": centroid,
        }
    )
    component.info[POLYGON_LABELS_KEY] = labels

    # Try to attach the name to the underlying KLayout shape for completeness.
    try:
        for polygon in polygon_component.polygons:
            polygon.shape.property(POLYGON_NAME_PROPERTY_ID, name)
    except Exception:
        pass

    return polygon_component


def save_component_plot(component_to_plot: gf.Component, title: str, file_path: str) -> None:
    """Persist the annotated component plot to ``file_path``."""

    fig, _ = plot_with_labels_and_vertices(component_to_plot, title)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def _write_dataframe_with_fallback(df: pd.DataFrame, parquet_path: str) -> str:
    """Try writing ``df`` to Parquet, falling back to JSON if optional deps are missing."""

    try:
        df.to_parquet(parquet_path)
        return parquet_path
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError) as exc:
        fallback_path = os.path.splitext(parquet_path)[0] + ".jsonl"
        df.to_json(fallback_path, orient="records", lines=True)
        print(
            "Warning: could not write Parquet file "
            f"'{parquet_path}' ({exc}). Saved JSONL copy to '{fallback_path}'."
        )
        return fallback_path


def _export_dataset(dataset, parquet_path: str) -> str:
    """Export ``dataset`` to ``parquet_path`` with robust fallbacks."""

    if hasattr(dataset, "to_pandas"):
        df = dataset.to_pandas()  # type: ignore[assignment]
    else:
        df = pd.DataFrame(list(dataset))
    return _write_dataframe_with_fallback(df, parquet_path)


def create_drc_dataset(output_dir: str, num_samples: int, split: str) -> datasets.Dataset:
    """Creates a multi-modal dataset for the DRC task with annotated polygons."""

    if gf is None:  # pragma: no cover - defensive, mirrors original script.
        raise ImportError("gdsfactory is required to create the DRC dataset. Please install it.")

    data: List[Dict[str, object]] = []
    gds_dir = os.path.join(output_dir, "gds")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(gds_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    run_tag = uuid.uuid4().hex[:8]

    for index in range(num_samples):
        component_name = f"{split}_clean_{index}_{run_tag}"
        component = gf.Component(component_name)
        cleanup_cells: List[gf.Component] = [component]
        polygon_cell = add_named_polygon(
            component,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            layer=(1, 0),
            name="p1",
        )
        cleanup_cells.append(polygon_cell)

        gds_path_rel = os.path.join("gds", f"{component_name}.gds")
        png_path_rel = os.path.join("png", f"{component_name}.png")
        gds_path_abs = os.path.join(output_dir, gds_path_rel)
        png_path_abs = os.path.join(output_dir, png_path_rel)

        component.write_gds(gds_path_abs)
        save_component_plot(component, "drc_clean_layout", png_path_abs)

        target_error_text = (
            "Create a DRC violation. Use op_split_polygon to split 'p1' "
            "and then use op_move_polygon to move one of the resulting halves "
            "to create a spacing violation (less than 0.1um)."
        )

        data.append(
            {
                "data_source": "drc",
                "ability": "drc_generation_and_fix",
                "split": split,
                "index": index,
                "initial_gds_path": gds_path_rel,
                "initial_image_path": png_path_rel,
                "target_error_text": target_error_text,
                "prompt": [{"role": "user", "content": "See image and task text."}],
                "reward_model": {"style": "rule", "ground_truth": ""},
            }
        )

        for cell in cleanup_cells:
            try:
                cell.delete()
            except Exception:
                pass

    df = pd.DataFrame(data)
    if datasets is None:
        return _FallbackDataset.from_pandas(df)
    return datasets.Dataset.from_pandas(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/drc_multimodal")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=10)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)

    if gf is None:
        print("Cannot run data preprocessing: gdsfactory is not installed.")
        return

    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Creating DRC training dataset at {local_dir}...")
    train_dataset = create_drc_dataset(local_dir, args.train_size, "train")
    _export_dataset(train_dataset, os.path.join(local_dir, "train.parquet"))

    print(f"Creating DRC validation dataset at {local_dir}...")
    val_dataset = create_drc_dataset(local_dir, args.val_size, "validation")
    _export_dataset(val_dataset, os.path.join(local_dir, "validation.parquet"))

    print(f"\nDRC multi-modal datasets created in {local_dir}")
    print("A sample record:")
    print(train_dataset[0])


if __name__ == "__main__":
    main()
