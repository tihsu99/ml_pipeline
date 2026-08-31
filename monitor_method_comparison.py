#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

import math
from generate_event_info_yaml import parse_feature_config
from common import build_classification_lookup, channel_latex_label, process_latex_label, post_calibrate_tau_tau, event_preselection_mask
from evaluation_config import post_calibration_enabled
from plot_style import plot_data_mc_histogram_from_counts, plot_truth_prediction_bundle
from quantum.observables_builder import build_observables, get_observable_names

DEFAULT_SAMPLE_LIMIT = 200_000
QUANTUM_RANGES = {
    "theta_cm": (0.0, 1.0),
    "mtautau": (0.0, 150.0),
}
for _axis in ("n", "r", "k"):
    QUANTUM_RANGES[f"cos_theta_A_{_axis}"] = (-1.0, 1.0)
    QUANTUM_RANGES[f"cos_theta_B_{_axis}"] = (-1.0, 1.0)
for _axis_a in ("n", "r", "k"):
    for _axis_b in ("n", "r", "k"):
        QUANTUM_RANGES[f"cos_theta_A_{_axis_a}_times_cos_theta_B_{_axis_b}"] = (-1.0, 1.0)
del _axis, _axis_a, _axis_b


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("Install pyyaml or use JSON config.")
        return yaml.safe_load(text)
    return json.loads(text)


def parquet_files(paths: list[Path]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if path.is_file() and path.suffix == ".parquet":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(p) for p in sorted(path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {paths}")
    return files


def build_sample_map(paths: list[Path], default_label: str, combine: bool) -> dict[str, list[str]]:
    if combine:
        return {default_label: parquet_files(paths)}

    samples: dict[str, list[str]] = {}
    for index, path in enumerate(paths):
        label = path.stem if path.is_file() else path.name
        if not label:
            label = f"{default_label}_{index}"
        if label in samples:
            label = f"{label}_{index}"
        samples[label] = parquet_files([path])
    return samples


def display_process_label(label: str) -> str:
    if label in {"Data", "data", "data94"}:
        return "Data"
    if label.startswith("Ztautau_"):
        return channel_latex_label(label)
    return process_latex_label(label)


def category_lookup_payload(config: dict[str, Any]) -> dict[str, Any]:
    lookup = build_classification_lookup(config)
    return {
        "sample_default_label": lookup.sample_default_label,
        "sample_event_category_to_label": lookup.sample_event_category_to_label,
    }


def neutrino_prediction_labels(config: dict[str, Any]) -> set[str]:
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    labels: set[str] = set()
    for raw_value in prediction_cfg.values():
        if isinstance(raw_value, dict):
            for nested_value in raw_value.values():
                if isinstance(nested_value, list):
                    labels.update(str(item) for item in nested_value)
                elif nested_value is not None:
                    labels.add(str(nested_value))
        elif isinstance(raw_value, list):
            labels.update(str(item) for item in raw_value)
        elif raw_value is not None:
            labels.add(str(raw_value))
    return labels


def event_process_labels(
    events: ak.Array,
    sample_name: str,
    is_data: bool,
    label_lookup: dict[str, Any],
) -> np.ndarray:
    if is_data:
        return np.asarray(["Data"] * len(events), dtype=object)
    if "classification_target_name" in events.fields:
        labels = np.asarray(ak.to_numpy(events["classification_target_name"], allow_missing=False), dtype=str)
        return np.asarray([display_process_label(label) for label in labels], dtype=object)

    category_map = label_lookup.get("sample_event_category_to_label", {}).get(sample_name)
    if category_map and "event_category" in events.fields:
        categories = to_numpy(events["event_category"], np.int64)
        labels = [category_map.get(int(category), sample_name) for category in categories]
        return np.asarray([display_process_label(label) for label in labels], dtype=object)

    default_label = label_lookup.get("sample_default_label", {}).get(sample_name, sample_name)
    return np.asarray([display_process_label(default_label)] * len(events), dtype=object)


def raw_event_process_labels(
    events: ak.Array,
    sample_name: str,
    label_lookup: dict[str, Any],
) -> np.ndarray:
    if "classification_target_name" in events.fields:
        return np.asarray(ak.to_numpy(events["classification_target_name"], allow_missing=False), dtype=str)

    category_map = label_lookup.get("sample_event_category_to_label", {}).get(sample_name)
    if category_map and "event_category" in events.fields:
        categories = to_numpy(events["event_category"], np.int64)
        return np.asarray([category_map.get(int(category), "") for category in categories], dtype=str)

    default_label = label_lookup.get("sample_default_label", {}).get(sample_name, sample_name)
    return np.asarray([default_label] * len(events), dtype=str)


def available_columns(file_name: str) -> set[str] | None:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(file_name).schema_arrow.names)
    except Exception:
        return None


def row_group_chunks(file_name: str, row_groups_per_chunk: int) -> list[list[int] | None]:
    if row_groups_per_chunk <= 0:
        return [None]
    try:
        import inspect

        if "row_groups" not in inspect.signature(ak.from_parquet).parameters:
            return [None]
    except Exception:
        return [None]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_name)
        num_row_groups = parquet_file.num_row_groups
    except Exception:
        return [None]

    chunks: list[list[int] | None] = []
    for start in range(0, num_row_groups, row_groups_per_chunk):
        chunks.append(list(range(start, min(start + row_groups_per_chunk, num_row_groups))))
    return chunks or [None]


def read_parquet_chunk(file_name: str, columns: list[str], row_groups: list[int] | None) -> ak.Array | None:
    present = available_columns(file_name)
    requested = list(dict.fromkeys(columns))
    selected = requested if present is None else [column for column in requested if column in present]
    if not selected:
        return None
    if row_groups is not None:
        try:
            return ak.from_parquet(file_name, columns=selected, row_groups=row_groups)
        except TypeError:
            pass
    return ak.from_parquet(file_name, columns=selected)


def to_numpy(array: Any, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(ak.to_numpy(array, allow_missing=False), dtype=dtype)
    except TypeError:
        return np.asarray(ak.to_numpy(array), dtype=dtype)
    except Exception:
        return np.asarray(ak.to_list(array), dtype=dtype)


def event_weights(events: ak.Array, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in events.fields:
        return None
    weights = to_numpy(events[weight_column], np.float64).reshape(-1)
    if len(weights) != len(events):
        return None
    return weights


def condition_values(events: ak.Array, global_idx: int) -> np.ndarray:
    conditions = to_numpy(events["conditions"], np.float64)
    if conditions.ndim != 2:
        conditions = np.asarray(conditions, dtype=np.float64).reshape((len(events), -1))
    if global_idx >= conditions.shape[1]:
        raise IndexError(f"conditions only has {conditions.shape[1]} columns; requested index {global_idx}.")
    return conditions[:, global_idx]

def evenet_columns() -> list[str]:
    return [
        "evenet_invisible_a_valid",
        "evenet_invisible_a_pt",
        "evenet_invisible_a_theta",
        "evenet_invisible_a_phi",
        "evenet_invisible_b_valid",
        "evenet_invisible_b_pt",
        "evenet_invisible_b_theta",
        "evenet_invisible_b_phi",
    ]

def sequential_values_and_weights(
    events: ak.Array,
    sequential_idx: int,
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    x = to_numpy(events["x"], np.float64)
    x_mask = to_numpy(events["x_mask"], bool)
    if x.ndim != 3:
        raise ValueError(f"x must be rank-3, got shape {x.shape}")
    if x_mask.shape != x.shape[:2]:
        raise ValueError(f"x_mask shape {x_mask.shape} does not match x leading shape {x.shape[:2]}")
    if sequential_idx >= x.shape[2]:
        raise IndexError(f"x only has {x.shape[2]} features; requested index {sequential_idx}.")

    values = x[:, :, sequential_idx][x_mask]
    weights = event_weights(events, weight_column)
    if weights is None:
        return values, None
    expanded_weights = np.broadcast_to(weights[:, None], x_mask.shape)[x_mask]
    return values, expanded_weights


def update_summary(summary: dict[str, Any], values: np.ndarray, sample_limit: int) -> None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return

    summary["count"] += int(finite_values.size)
    summary["min"] = min(summary["min"], float(np.min(finite_values)))
    summary["max"] = max(summary["max"], float(np.max(finite_values)))
    summary["integer"] = bool(summary["integer"] and np.allclose(finite_values, np.round(finite_values)))

    sampled = summary["sampled"]
    if sampled.size < sample_limit:
        need = sample_limit - sampled.size
        if finite_values.size > need:
            indices = np.linspace(0, finite_values.size - 1, need, dtype=np.int64)
            finite_values = finite_values[indices]
        summary["sampled"] = np.concatenate([sampled, finite_values])


def make_bins(summary: dict[str, Any], bins: int, max_integer_bins: int) -> np.ndarray:
    if summary["count"] == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    low = float(summary["min"])
    high = float(summary["max"])
    if summary["integer"]:
        int_low = math.floor(low)
        int_high = math.ceil(high)
        if int_high - int_low + 1 <= max_integer_bins:
            return np.arange(int_low - 0.5, int_high + 1.5, 1.0)

    sampled = summary["sampled"]
    if sampled.size:
        low, high = np.quantile(sampled, [0.005, 0.995])

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(summary["min"])
        high = float(summary["max"])
    if low == high:
        pad = abs(low) if low else 1.0
        low -= 0.5 * pad
        high += 0.5 * pad

    pad = 0.05 * (high - low)
    return np.linspace(low - pad, high + pad, bins + 1)


def empty_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
        "integer": True,
        "sampled": np.array([], dtype=np.float64),
    }


def sample_feature_values(
    events: ak.Array,
    feature_kind: str,
    feature_idx: int,
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if feature_kind == "global":
        return condition_values(events, feature_idx), event_weights(events, weight_column)
    if feature_kind == "sequential":
        return sequential_values_and_weights(events, feature_idx, weight_column)
    raise ValueError(f"Unsupported feature kind: {feature_kind}")


def needed_columns(feature_kind: str, weight_column: str | None) -> list[str]:
    columns = ["conditions"] if feature_kind == "global" else ["x", "x_mask"]
    columns.extend(["classification_target_name", "event_category", "baseline_theta_cm"])
    if weight_column:
        columns.append(weight_column)
    return columns


def scan_feature_range(
    sample_files: dict[str, list[str]],
    feature_kind: str,
    feature_idx: int,
    weight_column: str | None,
    max_events: int | None,
    row_groups_per_chunk: int,
) -> dict[str, Any]:
    summary = empty_summary()
    rows_by_sample: dict[str, int] = {}
    columns = needed_columns(feature_kind, weight_column)
    for sample_name, files in sample_files.items():
        remaining = max_events
        rows_by_sample[sample_name] = 0
        for file_name in files:
            for row_groups in row_group_chunks(file_name, row_groups_per_chunk):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                values, _ = sample_feature_values(events, feature_kind, feature_idx, weight_column)
                rows_by_sample[sample_name] += int(len(events))
                update_summary(summary, values, DEFAULT_SAMPLE_LIMIT)
                if remaining is not None:
                    remaining -= int(len(events))
                del events, values
                gc.collect()
            if remaining is not None and remaining <= 0:
                break
    return {"summary": summary, "rows_by_sample": rows_by_sample}


def histogram_feature(
    sample_files: dict[str, list[str]],
    feature_kind: str,
    feature_idx: int,
    bins: np.ndarray,
    weight_column: str | None,
    max_events: int | None,
    row_groups_per_chunk: int,
    *,
    is_data: bool,
    label_lookup: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    histograms: dict[str, dict[str, Any]] = {}
    columns = needed_columns(feature_kind, weight_column)
    for sample_name, files in sample_files.items():
        rows = 0
        remaining = max_events
        for file_name in files:
            for row_groups in row_group_chunks(file_name, row_groups_per_chunk):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                values, weights = sample_feature_values(events, feature_kind, feature_idx, weight_column)
                event_labels = event_process_labels(events, sample_name, is_data, label_lookup)
                if feature_kind == "sequential":
                    x_mask = to_numpy(events["x_mask"], bool)
                    labels = np.broadcast_to(event_labels[:, None], x_mask.shape)[x_mask]
                else:
                    labels = event_labels

                finite = np.isfinite(values)
                if weights is not None:
                    finite &= np.isfinite(weights)
                finite_values = values[finite]
                finite_labels = labels[finite]
                finite_weights = None if weights is None else weights[finite]
                for label in np.unique(finite_labels):
                    label_mask = finite_labels == label
                    state = histograms.setdefault(
                        str(label),
                        {
                            "counts": np.zeros(len(bins) - 1, dtype=np.float64),
                            "sumw2": np.zeros(len(bins) - 1, dtype=np.float64),
                            "rows": 0,
                            "weighted": False,
                        },
                    )
                    if finite_weights is None:
                        counts_chunk = np.histogram(finite_values[label_mask], bins=bins)[0].astype(np.float64)
                        state["counts"] += counts_chunk
                        state["sumw2"] += counts_chunk
                    else:
                        label_weights = finite_weights[label_mask]
                        state["weighted"] = True
                        state["counts"] += np.histogram(finite_values[label_mask], bins=bins, weights=label_weights)[0]
                        state["sumw2"] += np.histogram(
                            finite_values[label_mask],
                            bins=bins,
                            weights=label_weights * label_weights,
                        )[0]
                    state["rows"] += int(np.sum(label_mask))

                rows += int(len(events))
                if remaining is not None:
                    remaining -= int(len(events))
                del events, values, weights, labels
                gc.collect()
            if remaining is not None and remaining <= 0:
                break
        if rows == 0:
            continue
    return histograms


def plot_control_histograms(
    feature_name: str,
    bins: np.ndarray,
    data_histograms: dict[str, dict[str, Any]],
    mc_histograms: dict[str, dict[str, Any]],
    output_dir: Path,
    title_prefix: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    data_sumw2 = np.zeros(len(bins) - 1, dtype=np.float64)
    for payload in data_histograms.values():
        data_counts += payload["counts"]
        data_sumw2 += payload["sumw2"]
    mc_counts = {sample_name: payload["counts"] for sample_name, payload in mc_histograms.items()}
    plot_path = output_dir / f"{feature_name.replace('/', '_')}.png"
    return plot_data_mc_histogram_from_counts(
        bins,
        data_counts,
        data_sumw2,
        mc_counts,
        plot_path,
        title=f"{title_prefix}: {feature_name}",
        xlabel=feature_name,
    )


def process_control_feature(payload: dict[str, Any]) -> dict[str, Any]:
    feature_idx = payload["feature_idx"]
    feature_name = payload["feature_name"]
    feature_kind = payload["feature_kind"]
    sample_files = {**payload["mc_files"]}
    range_result = scan_feature_range(
        sample_files,
        feature_kind,
        feature_idx,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
    )
    bins = make_bins(range_result["summary"], payload["bins"], payload["max_integer_bins"])
    data_histograms = histogram_feature(
        payload["data_files"],
        feature_kind,
        feature_idx,
        bins,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
        is_data=True,
        label_lookup=payload["label_lookup"],
    )
    mc_histograms = histogram_feature(
        payload["mc_files"],
        feature_kind,
        feature_idx,
        bins,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
        is_data=False,
        label_lookup=payload["label_lookup"],
    )
    plot_path = plot_control_histograms(
        feature_name,
        bins,
        data_histograms,
        mc_histograms,
        Path(payload["output_dir"]),
        payload["title_prefix"],
    )
    return {
        "feature_kind": feature_kind,
        "feature_name": feature_name,
        "plot": plot_path,
        "rows_by_sample": range_result["rows_by_sample"],
    }


def p4_field_candidates(prefixes: tuple[str, ...], component: str) -> list[str]:
    return [f"{prefix}_{component}" for prefix in prefixes]


def first_existing_field(events: ak.Array, candidates: list[str]) -> str | None:
    fields = set(events.fields)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def component_array(events: ak.Array, candidates: list[str], slot: int | None = None) -> np.ndarray | None:
    field = first_existing_field(events, candidates)
    if field is None:
        return None
    values = to_numpy(events[field], np.float64)
    if slot is not None and values.ndim == 2 and values.shape[1] > slot:
        return values[:, slot]
    return values


def p4_from_component_prefixes(
    events: ak.Array,
    prefixes: tuple[str, ...],
    slot: int | None = None,
) -> ak.Array | None:
    components = {
        "px": component_array(events, p4_field_candidates(prefixes, "px"), slot),
        "py": component_array(events, p4_field_candidates(prefixes, "py"), slot),
        "pz": component_array(events, p4_field_candidates(prefixes, "pz"), slot),
        "E": component_array(events, p4_field_candidates(prefixes, "E"), slot),
    }
    if any(value is None for value in components.values()):
        return None
    return ak.zip(
        {
            "px": components["px"],
            "py": components["py"],
            "pz": components["pz"],
            "E": components["E"],
        },
        with_name="Momentum4D",
    )


def target_visible_columns() -> list[str]:
    columns = []
    for leg in ("a", "b"):
        visible_prefixes = (
            f"lead_{leg}_visible",
            f"visible_{leg}",
            "visible",
        )
        target_prefixes = (
            f"target_{leg}_invisible",
            f"target_{leg}_missing",
            f"target_missing_{leg}",
            f"lead_{leg}_missing",
            "target_missing",
            "target_invisible",
        )
        for component in ("px", "py", "pz", "E"):
            columns.extend(p4_field_candidates(visible_prefixes, component))
            columns.extend(p4_field_candidates(target_prefixes, component))
    return columns


def build_target_observables(events: ak.Array) -> dict[str, np.ndarray] | None:
    try:
        import vector

        vector.register_awkward()
    except Exception:
        pass

    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1)
    target_a = p4_from_component_prefixes(
        events,
        ("target_a_invisible", "target_a_missing", "target_missing_a", "lead_a_missing", "target_missing", "target_invisible"),
        slot=0,
    )
    target_b = p4_from_component_prefixes(
        events,
        ("target_b_invisible", "target_b_missing", "target_missing_b", "lead_b_missing", "target_missing", "target_invisible"),
        slot=1,
    )
    if visible_a is None or visible_b is None or target_a is None or target_b is None:
        return None
    observables = build_observables(
        tau_a_p4=visible_a + target_a,
        tau_b_p4=visible_b + target_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}


TAU_MASS = 1.777  # GeV
CM_ENERGY = 91.2 # GeV



def build_evenet_observables(
    events: ak.Array,
    post_calibration: bool,
) -> dict[str, np.ndarray] | None:
    try:
        import vector
        vector.register_awkward()
    except Exception:
        pass

    required = {
        "evenet_invisible_a_theta",
        "evenet_invisible_a_phi",
        "evenet_invisible_b_theta",
        "evenet_invisible_b_phi",
    }

    missing = required - set(events.fields)
    if missing:
        print(f"[monitor-input] missing EveNet columns: {sorted(missing)}", flush=True)
        return None

    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1)

    if visible_a is None or visible_b is None:
        print("[monitor-input] missing visible four-vector columns", flush=True)
        return None

    delta_visible_a_theta = events["evenet_invisible_a_theta"]
    delta_visible_a_phi = events["evenet_invisible_a_phi"]
    delta_visible_b_theta = events["evenet_invisible_b_theta"]
    delta_visible_b_phi = events["evenet_invisible_b_phi"]

    tau_a_theta = visible_a.theta + delta_visible_a_theta
    tau_b_theta = visible_b.theta + delta_visible_b_theta
    tau_a_phi = (visible_a.phi + delta_visible_a_phi + math.pi) % (2 * math.pi) - math.pi
    tau_b_phi = (visible_b.phi + delta_visible_b_phi + math.pi) % (2 * math.pi) - math.pi

    energy = CM_ENERGY / 2
    mass = TAU_MASS

    p = np.sqrt( energy ** 2 - mass ** 2)

    tau_a = ak.zip(
        {
            "pt": p * np.sin(tau_a_theta),
            "theta": tau_a_theta,
            "phi": tau_a_phi,
            "m": ak.ones_like(tau_a_theta) * mass,
        },
        with_name="Momentum4D",
    )

    tau_b = ak.zip(
        {
            "pt": p * np.sin(tau_b_theta),
            "theta": tau_b_theta,
            "phi": tau_b_phi,
            "m": ak.ones_like(tau_b_theta) * mass,
        },
        with_name="Momentum4D",
    )


    if post_calibration:
        tau_a, tau_b = post_calibrate_tau_tau(tau_a, tau_b)

    observables = build_observables(
        tau_a_p4=tau_a,
        tau_b_p4=tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )



    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}

def quantum_columns(observable: str, weight_column: str | None) -> list[str]:
    columns = target_visible_columns()
    columns.extend(evenet_columns())
    columns.extend([
        f"truth_{observable}",
        f"baseline_{observable}",
        "baseline_flags_valid",
        "baseline_mmc_likelihood",
        "mmc_likelihood",
    ])
    if weight_column:
        columns.append(weight_column)
    return list(dict.fromkeys(columns))


def observable_values(
    events: ak.Array,
    source: str,
    observable: str,
    derived_observables: dict[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    if source in {"target", "evenet"}:
        if derived_observables is None:
            return None
        return derived_observables.get(observable)

    if source == "truth":
        field = f"truth_{observable}"
    elif source == "baseline":
        field = f"baseline_{observable}"
    else:
        field = f"{source}_{observable}"

    if field not in events.fields:
        return None

    return to_numpy(events[field], np.float64)

def baseline_valid_mask(events: ak.Array) -> np.ndarray:
    return events["baseline_theta_cm"] > 0


def process_quantum_observable(payload: dict[str, Any]) -> dict[str, Any]:
    observable = payload["observable"]
    low, high = QUANTUM_RANGES.get(observable, (-1.0, 1.0))
    bins = np.linspace(low, high, payload["bins_2d"] + 1)
    comparisons = {
        "truth_vs_target": ("truth", "target", "Stored truth", "Target", {"Baseline": "baseline"}),
        "target_vs_baseline": ("target", "baseline", "Target", "Baseline", {"Stored truth": "truth"}),
        "truth_vs_baseline": ("truth", "baseline", "Stored truth", "Baseline", {"Target": "target"}),
        "truth_vs_evenet": ("truth", "evenet", "Stored truth", "Evenet", {"Baseline": "baseline"}),
        "target_vs_evenet": ("target", "evenet", "Target", "Evenet", {"Baseline": "baseline"}),
        "baseline_vs_evenet": ("baseline", "evenet", "Baseline", "Evenet", {"Stored truth": "truth"}),

    }
    process_values: dict[str, dict[str, Any]] = {}
    columns = quantum_columns(observable, payload["weight_column"])
    columns.extend(["classification_target_name", "event_category", "baseline_theta_cm"])
    sample_files = {**payload["mc_files"]}
    allowed_labels = set(payload["neutrino_prediction_labels"])

    for sample_name, files in sample_files.items():
        remaining = payload["max_events"]
        for file_name in files:
            for row_groups in row_group_chunks(file_name, payload["row_groups_per_chunk"]):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                labels = raw_event_process_labels(events, sample_name, payload["label_lookup"])
                keep = np.asarray([True for label in labels], dtype=bool) # TODO, to fix later
                if not np.any(keep):
                    if remaining is not None:
                        remaining -= int(len(events))
                    del events, labels
                    gc.collect()
                    continue
                target_observables = build_target_observables(events)
                evenet_observables = build_evenet_observables(
                    events,
                    post_calibration=payload["post_calibration"],
                )
                weights = event_weights(events, payload["weight_column"])
                source_values = {
                    "truth": observable_values(events, "truth", observable),
                    "target": observable_values(events, "target", observable, target_observables),
                    "baseline": observable_values(events, "baseline", observable),
                    "evenet": observable_values(events, "evenet", observable, evenet_observables),
                }
                if source_values["baseline"] is not None:
                    baseline_values = source_values["baseline"].copy()
                    baseline_values[~baseline_valid_mask(events)] = np.nan
                    source_values["baseline"] = baseline_values

                for process_name in np.unique(labels[keep]):
                    process_mask = keep & (labels == process_name)
                    state = process_values.setdefault(
                        str(process_name),
                        {"truth": [], "target": [], "baseline": [], "evenet": [], "weight": [], "total": 0},
                    )
                    state["total"] += int(np.sum(process_mask))
                    for source_name, values_array in source_values.items():
                        if values_array is not None:
                            state[source_name].append(values_array[process_mask])
                    if weights is not None:
                        state["weight"].append(weights[process_mask])
                if remaining is not None:
                    remaining -= int(len(events))
                del events, target_observables, weights, labels, keep, source_values
                gc.collect()
            if remaining is not None and remaining <= 0:
                break

    results = {}
    output_dir = Path(payload["output_dir"])
    for comparison_name, (x_source, y_source, x_label, y_label, extra_sources) in comparisons.items():
        for process_name, state in process_values.items():
            if not state[x_source] or not state[y_source]:
                continue

            x_values = np.concatenate(state[x_source])
            y_values = np.concatenate(state[y_source])
            weights = np.concatenate(state["weight"]) if state["weight"] else None
            extra_predictions = {
                extra_label: np.concatenate(state[source_name])
                for extra_label, source_name in extra_sources.items()
                if state[source_name]
            }
            process_result = plot_truth_prediction_bundle(
                x_values,
                y_values,
                output_dir / comparison_name / process_name,
                observable,
                bins=bins,
                weight=weights,
                truth_label=x_label,
                pred_label=y_label,
                xaxis_label=observable,
                title=f"{comparison_name}: {observable}",
                summary_title=f"{display_process_label(process_name)} summary",
                total_entries=state["total"],
                extra_predictions=extra_predictions,
            )
            results.setdefault(comparison_name, {})[process_name] = process_result
    return {
        "observable": observable,
        "plots": {
            comparison_name: {
                process_name: result["plots"]
                for process_name, result in process_results.items()
            }
            for comparison_name, process_results in results.items()
        },
        "metrics": {
            comparison_name: {
                process_name: result["metrics"]
                for process_name, result in process_results.items()
            }
            for comparison_name, process_results in results.items()
        },
    }


def run_tasks(tasks: list[dict[str, Any]], workers: int, worker_func, log_key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            print(f"[monitor-input] plotting {task[log_key]}", flush=True)
            results.append(worker_func(task))
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        future_map = {executor.submit(worker_func, task): task[log_key] for task in tasks}
        for future in as_completed(future_map):
            name = future_map[future]
            print(f"[monitor-input] finished {name}", flush=True)
            results.append(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser("Monitor EveNet input parquet files")
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--mc-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("monitor_plots"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--bins-2d", type=int, default=70)
    parser.add_argument("--max-integer-bins", type=int, default=120)
    parser.add_argument("--weight-column", default="event_weight")
    parser.add_argument("--row-groups-per-chunk", type=int, default=1)
    parser.add_argument("--skip-quantum", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    post_calibration = post_calibration_enabled(config)
    feature_config = parse_feature_config(config)

    # data_files = build_sample_map(args.data_dir, "data", combine=True)
    mc_files = build_sample_map(args.mc_dir, "mc", combine=False)
    common_payload = {
        # "data_files": data_files,
        "mc_files": mc_files,
        "max_events": args.max_events,
        "bins": args.bins,
        "max_integer_bins": args.max_integer_bins,
        "weight_column": args.weight_column or None,
        "row_groups_per_chunk": args.row_groups_per_chunk,
        "label_lookup": category_lookup_payload(config),
        "post_calibration": post_calibration,
    }

    all_results: dict[str, Any] = {
        # "data_samples": {name: len(files) for name, files in data_files.items()},
        "mc_samples": {name: len(files) for name, files in mc_files.items()},
        "max_events": args.max_events,
        "weight_column": args.weight_column,
        "post_calibration": post_calibration,
        "neutrino_prediction_labels": sorted(neutrino_prediction_labels(config)),
    }

    if not args.skip_quantum:
        quantum_dir = args.output_dir / "quantum_2d"
        quantum_tasks = [
            {
                **common_payload,
                "observable": observable,
                "bins_2d": args.bins_2d,
                "output_dir": str(quantum_dir),
                "neutrino_prediction_labels": sorted(neutrino_prediction_labels(config)),
            }
            for observable in get_observable_names()
        ]
        print(f"[monitor-input] quantum comparisons={len(quantum_tasks)} workers={args.num_workers}", flush=True)
        quantum_results = run_tasks(quantum_tasks, args.num_workers, process_quantum_observable, "observable")
        all_results["quantum_2d"] = {
            result["observable"]: {
                comparison: {
                    process_name: {
                        plot_name: str(Path(plot_path).relative_to(args.output_dir))
                        for plot_name, plot_path in process_plots.items()
                    }
                    for process_name, process_plots in comparison_plots.items()
                }
                for comparison, comparison_plots in result["plots"].items()
            }
            for result in quantum_results
        }
        all_results["quantum_metrics"] = {
            result["observable"]: result["metrics"]
            for result in quantum_results
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"[monitor-input] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
