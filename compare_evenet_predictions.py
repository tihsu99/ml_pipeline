#!/usr/bin/env python3
"""Compare EveNet diffusion predictions without requiring A/B row alignment.

Each prediction parquet is evaluated against the truth target stored in its own
``x_invisible`` tensor.  Cross-method plots therefore compare distributions and
metrics; they never subtract checkpoint A events from checkpoint B events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


@dataclass
class FeatureSample:
    target: np.ndarray
    prediction: np.ndarray
    weight: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare truth-target and EveNet predictions for two or more methods."
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="LABEL{=,:}PATH",
        help="Method label and prediction parquet file/directory. Repeat for every method.",
    )
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "prediction_comparison.yaml",
    )
    parser.add_argument(
        "--output-dir",
        "--output-prefix",
        dest="output_dir",
        type=Path,
        required=True,
        help="Output directory. --output-prefix is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Override max_events_per_method. Use 0 to process all events.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def parse_method(spec: str) -> tuple[str, Path]:
    split_positions = [index for index in (spec.find("="), spec.find(":")) if index > 0]
    if not split_positions:
        raise ValueError(f"Invalid --method {spec!r}; expected LABEL=PATH or LABEL:PATH.")
    split_at = min(split_positions)
    label, raw_path = spec[:split_at], spec[split_at + 1:]
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label or not raw_path.strip():
        raise ValueError(f"Invalid --method {spec!r}; label and path are required.")
    return label, path


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    merged = sorted(path.rglob("*__evenet_pred.parquet"))
    if merged:
        return merged
    parts = sorted(path.rglob("*__evenet_pred.part*.parquet"))
    if parts:
        return parts
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}.")
    return files


def invisible_features(analysis_config: dict[str, Any], configured: Any) -> list[str]:
    available = list((analysis_config.get("Normalization") or {}).get("Invisible", {}).keys())
    requested = available if configured is None else [str(item) for item in configured]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Comparison features are not in Normalization.Invisible: {missing}")
    if not requested:
        raise ValueError("No Invisible features were configured.")
    return requested


def to_numpy(values: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(ak.to_numpy(values, allow_missing=False), dtype=dtype)


def p3_fields(present: set[str], prefixes: tuple[str, ...]) -> dict[str, str] | None:
    fields = {}
    for component in ("px", "py", "pz"):
        field = next(
            (f"{prefix}_{component}" for prefix in prefixes if f"{prefix}_{component}" in present),
            None,
        )
        if field is None:
            return None
        fields[component] = field
    return fields


def p3_values(events: ak.Array, fields: dict[str, str], slot: int) -> dict[str, np.ndarray]:
    output = {}
    for component, field in fields.items():
        values = to_numpy(events[field])
        if values.ndim == 2:
            if values.shape[1] <= slot:
                raise ValueError(f"{field} has shape {values.shape}; slot {slot} is unavailable.")
            values = values[:, slot]
        elif values.ndim != 1:
            raise ValueError(f"{field} has shape {values.shape}; expected one value per event or per leg.")
        output[component] = values
    return output


def four_vector_target_columns(
    present: set[str], legs: list[str]
) -> dict[str, tuple[dict[str, str], dict[str, str]]] | None:
    output = {}
    for leg in legs:
        visible = p3_fields(present, (f"lead_{leg}_visible", f"visible_{leg}", "visible"))
        missing = p3_fields(
            present,
            (
                f"target_{leg}_missing",
                f"target_missing_{leg}",
                f"lead_{leg}_missing",
                "target_missing",
            ),
        )
        if visible is None or missing is None:
            return None
        output[leg] = (visible, missing)
    return output


def angular_target_values(
    events: ak.Array,
    fields_by_leg: dict[str, tuple[dict[str, str], dict[str, str]]],
    legs: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    targets = {}
    masks = {}
    slot_index = {"a": 0, "b": 1}
    for leg in legs:
        visible = p3_values(events, fields_by_leg[leg][0], slot_index[leg])
        missing = p3_values(events, fields_by_leg[leg][1], slot_index[leg])
        visible_pt = np.hypot(visible["px"], visible["py"])
        visible_theta = np.arctan2(visible_pt, visible["pz"])
        visible_phi = np.arctan2(visible["py"], visible["px"])
        tau_px = visible["px"] + missing["px"]
        tau_py = visible["py"] + missing["py"]
        tau_pz = visible["pz"] + missing["pz"]
        tau_pt = np.hypot(tau_px, tau_py)
        tau_theta = np.arctan2(tau_pt, tau_pz)
        tau_phi = np.arctan2(tau_py, tau_px)
        targets[leg] = {
            "theta": tau_theta - visible_theta,
            "phi": (tau_phi - visible_phi + math.pi) % (2.0 * math.pi) - math.pi,
        }
        masks[leg] = (
            np.isfinite(visible_theta)
            & np.isfinite(visible_phi)
            & np.isfinite(tau_theta)
            & np.isfinite(tau_phi)
            & ((tau_pt > 0) | (np.abs(tau_pz) > 0))
        )
    return targets, masks


def load_method(
    files: list[Path],
    features: list[str],
    all_features: list[str],
    legs: list[str],
    weight_column: str | None,
    max_events: int | None,
    target_source: str,
) -> tuple[dict[tuple[str, str], FeatureSample], int, list[str], list[str]]:
    import pyarrow.parquet as pq

    feature_index = {name: index for index, name in enumerate(all_features)}
    slot_index = {"a": 0, "b": 1}
    target_chunks: dict[tuple[str, str], list[np.ndarray]] = {
        (leg, feature): [] for leg in legs for feature in features
    }
    prediction_chunks = {key: [] for key in target_chunks}
    weight_chunks = {key: [] for key in target_chunks}
    rows_read = 0
    sources_used = set()
    skipped_files = []

    for path in files:
        parquet_file = pq.ParquetFile(path)
        present = set(parquet_file.schema_arrow.names)
        required = set()
        required.update(f"evenet_invisible_{leg}_valid" for leg in legs)
        required.update(
            f"evenet_invisible_{leg}_{feature}" for leg in legs for feature in features
        )
        if weight_column:
            required.add(weight_column)
        has_tensor_target = {"x_invisible", "x_invisible_mask"}.issubset(present)
        flat_target_columns = {
            (leg, feature): f"x_invisible:{slot_index[leg]}:{feature_index[feature]}"
            for leg in legs
            for feature in features
        }
        flat_mask_columns = {
            leg: f"x_invisible_mask:{slot_index[leg]}" for leg in legs
        }
        has_flat_tensor_target = set(flat_target_columns.values()).issubset(present) and set(
            flat_mask_columns.values()
        ).issubset(present)
        target_p3_fields = four_vector_target_columns(present, legs)
        if target_source == "x_invisible" or (
            target_source == "auto" and (has_tensor_target or has_flat_tensor_target)
        ):
            if not has_tensor_target and not has_flat_tensor_target:
                raise ValueError(
                    f"{path} has no nested or flattened x_invisible target columns required "
                    "by target_source=x_invisible."
                )
            if has_tensor_target:
                file_target_source = "x_invisible"
                required.update(("x_invisible", "x_invisible_mask"))
            else:
                file_target_source = "x_invisible_flattened"
                required.update(flat_target_columns.values())
                required.update(flat_mask_columns.values())
        else:
            if target_p3_fields is None:
                if target_source == "auto":
                    skipped_files.append(str(path))
                    print(
                        f"[compare] skipping targetless parquet: {path}",
                        flush=True,
                    )
                    continue
                raise ValueError(
                    f"{path} has neither x_invisible targets nor a complete visible + target-missing "
                    "three-vector. Set PredictionComparison.target_source only after checking its schema."
                )
            unsupported = sorted(set(features) - {"theta", "phi"})
            if unsupported:
                raise ValueError(
                    f"Four-vector target reconstruction supports theta/phi only, not {unsupported}."
                )
            file_target_source = "four_vector"
            for visible_fields, missing_fields in target_p3_fields.values():
                required.update(visible_fields.values())
                required.update(missing_fields.values())
        sources_used.add(file_target_source)
        missing = sorted(required - present)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        columns = sorted(required)

        for row_group in range(parquet_file.num_row_groups):
            if max_events is not None and rows_read >= max_events:
                break
            events = ak.from_arrow(parquet_file.read_row_group(row_group, columns=columns))
            if max_events is not None:
                events = events[: max_events - rows_read]
            if len(events) == 0:
                continue
            if file_target_source == "x_invisible":
                tensor_targets = to_numpy(events["x_invisible"])
                tensor_target_mask = to_numpy(events["x_invisible_mask"], bool)
                if tensor_targets.ndim != 3:
                    raise ValueError(
                        f"x_invisible in {path} has shape {tensor_targets.shape}, expected rank 3."
                    )
                if tensor_targets.shape[2] < len(all_features):
                    raise ValueError(
                        f"x_invisible in {path} has {tensor_targets.shape[2]} features, "
                        f"but analysis config has {len(all_features)}."
                    )
                derived_targets = None
                derived_target_masks = None
            elif file_target_source == "x_invisible_flattened":
                tensor_targets = None
                tensor_target_mask = None
                derived_targets = None
                derived_target_masks = None
            else:
                derived_targets, derived_target_masks = angular_target_values(
                    events, target_p3_fields, legs
                )
            base_weight = (
                to_numpy(events[weight_column])
                if weight_column and weight_column in events.fields
                else np.ones(len(events), dtype=np.float64)
            )

            for leg in legs:
                slot = slot_index[leg]
                pred_valid = to_numpy(events[f"evenet_invisible_{leg}_valid"], bool)
                if file_target_source == "x_invisible":
                    targets_by_feature = {
                        feature: tensor_targets[:, slot, feature_index[feature]]
                        for feature in features
                    }
                    target_valid = tensor_target_mask[:, slot]
                elif file_target_source == "x_invisible_flattened":
                    targets_by_feature = {
                        feature: to_numpy(events[flat_target_columns[(leg, feature)]])
                        for feature in features
                    }
                    target_valid = to_numpy(events[flat_mask_columns[leg]], bool)
                else:
                    targets_by_feature = derived_targets[leg]
                    target_valid = derived_target_masks[leg]
                predictions_by_feature = {
                    feature: to_numpy(events[f"evenet_invisible_{leg}_{feature}"])
                    for feature in features
                }
                valid = target_valid & pred_valid & np.isfinite(base_weight)
                for feature in features:
                    valid &= np.isfinite(targets_by_feature[feature])
                    valid &= np.isfinite(predictions_by_feature[feature])
                for feature in features:
                    key = (leg, feature)
                    target_chunks[key].append(targets_by_feature[feature][valid])
                    prediction_chunks[key].append(predictions_by_feature[feature][valid])
                    weight_chunks[key].append(base_weight[valid])
            rows_read += len(events)
        if max_events is not None and rows_read >= max_events:
            break

    output = {}
    if rows_read == 0:
        raise ValueError(
            "No evaluable MC events were found after skipping targetless parquet files."
        )
    for key in target_chunks:
        output[key] = FeatureSample(
            target=np.concatenate(target_chunks[key]) if target_chunks[key] else np.array([]),
            prediction=np.concatenate(prediction_chunks[key]) if prediction_chunks[key] else np.array([]),
            weight=np.concatenate(weight_chunks[key]) if weight_chunks[key] else np.array([]),
        )
    return output, rows_read, sorted(sources_used), skipped_files


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0 else float("nan")


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(mean):
        return float("nan")
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(y, weights)
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    covariance = float(np.sum(weights * (x - mean_x) * (y - mean_y)) / total)
    variance_x = float(np.sum(weights * (x - mean_x) ** 2) / total)
    variance_y = float(np.sum(weights * (y - mean_y) ** 2) / total)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0 else float("nan")


def js_divergence(sample: FeatureSample, edges: np.ndarray) -> float:
    target_hist = np.histogram(sample.target, bins=edges, weights=sample.weight)[0].astype(float)
    pred_hist = np.histogram(sample.prediction, bins=edges, weights=sample.weight)[0].astype(float)
    if target_hist.sum() <= 0 or pred_hist.sum() <= 0:
        return float("nan")
    target_hist /= target_hist.sum()
    pred_hist /= pred_hist.sum()
    mixture = 0.5 * (target_hist + pred_hist)
    target_mask = target_hist > 0
    pred_mask = pred_hist > 0
    return float(
        0.5 * np.sum(target_hist[target_mask] * np.log(target_hist[target_mask] / mixture[target_mask]))
        + 0.5 * np.sum(pred_hist[pred_mask] * np.log(pred_hist[pred_mask] / mixture[pred_mask]))
    )


def feature_edges(samples: list[FeatureSample], bins: int, quantiles: tuple[float, float]) -> np.ndarray:
    values = [np.concatenate([sample.target, sample.prediction]) for sample in samples if sample.target.size]
    if not values:
        return np.linspace(-1.0, 1.0, bins + 1)
    merged = np.concatenate(values)
    low, high = np.quantile(merged, quantiles)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(merged)), float(np.max(merged))
    if low == high:
        padding = abs(low) * 0.05 if low else 1.0
        low, high = low - padding, high + padding
    return np.linspace(float(low), float(high), bins + 1)


def sample_metrics(sample: FeatureSample, edges: np.ndarray) -> dict[str, float]:
    if sample.target.size == 0:
        return {key: float("nan") for key in ("entries", "sum_weight", "bias", "resolution", "mae", "rmse", "pearson", "jsd")}
    residual = sample.prediction - sample.target
    return {
        "entries": float(sample.target.size),
        "sum_weight": float(np.sum(sample.weight)),
        "bias": weighted_mean(residual, sample.weight),
        "resolution": weighted_std(residual, sample.weight),
        "mae": weighted_mean(np.abs(residual), sample.weight),
        "rmse": math.sqrt(weighted_mean(residual**2, sample.weight)),
        "pearson": weighted_pearson(sample.target, sample.prediction, sample.weight),
        "jsd": js_divergence(sample, edges),
    }


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    methods: dict[str, FeatureSample], edges: np.ndarray, title: str, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(methods)))
    for color, (label, sample) in zip(colors, methods.items()):
        axis.hist(sample.target, bins=edges, weights=sample.weight, density=True, histtype="step", linestyle="--", color=color, label=f"{label} target")
        axis.hist(sample.prediction, bins=edges, weights=sample.weight, density=True, histtype="step", linewidth=1.6, color=color, label=f"{label} prediction")
    axis.set_xlabel(title)
    axis.set_ylabel("Normalized density")
    axis.set_title(f"1D target vs prediction: {title}", loc="left")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, path)


def plot_response_maps(
    methods: dict[str, FeatureSample], edges: np.ndarray, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(6.0 * len(methods), 5.2),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (label, sample) in zip(axes[0], methods.items()):
        counts = np.histogram2d(sample.target, sample.prediction, bins=(edges, edges), weights=sample.weight)[0].T
        positive = counts[counts > 0]
        norm = LogNorm(vmin=max(float(np.min(positive)), 1e-12), vmax=float(np.max(positive))) if positive.size else None
        mesh = axis.pcolormesh(edges, edges, np.ma.masked_where(counts <= 0, counts), cmap="viridis", norm=norm)
        axis.plot([edges[0], edges[-1]], [edges[0], edges[-1]], "w--", linewidth=1)
        axis.set_xlabel("Target")
        axis.set_ylabel("Prediction")
        axis.set_title(label)
        fig.colorbar(mesh, ax=axis, label="Weighted entries")
    fig.suptitle(f"2D target-to-prediction mapping: {title}")
    save_figure(fig, path)


def plot_residuals_and_profiles(
    methods: dict[str, FeatureSample], edges: np.ndarray, profile_bins: int, title: str, path: Path
) -> None:
    fig, (residual_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(11.0, 4.5), constrained_layout=True
    )
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(methods)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile_indices = np.unique(np.linspace(0, len(edges) - 1, min(profile_bins, len(edges) - 1) + 1, dtype=int))
    profile_edges = edges[profile_indices]
    profile_centers = 0.5 * (profile_edges[:-1] + profile_edges[1:])
    all_residuals = [sample.prediction - sample.target for sample in methods.values() if sample.target.size]
    residual_edges = feature_edges(
        [FeatureSample(values, values, np.ones_like(values)) for values in all_residuals],
        len(edges) - 1,
        (0.005, 0.995),
    )
    for color, (label, sample) in zip(colors, methods.items()):
        residual = sample.prediction - sample.target
        residual_axis.hist(residual, bins=residual_edges, weights=sample.weight, density=True, histtype="step", color=color, label=label)
        bias = np.full(len(profile_centers), np.nan)
        resolution = np.full(len(profile_centers), np.nan)
        for index in range(len(profile_centers)):
            selected = (sample.target >= profile_edges[index]) & (sample.target < profile_edges[index + 1])
            if index == len(profile_centers) - 1:
                selected |= sample.target == profile_edges[index + 1]
            if np.any(selected):
                bias[index] = weighted_mean(residual[selected], sample.weight[selected])
                resolution[index] = weighted_std(residual[selected], sample.weight[selected])
        valid = np.isfinite(bias)
        profile_axis.errorbar(profile_centers[valid], bias[valid], yerr=resolution[valid], fmt="o-", markersize=3, linewidth=1, color=color, label=label)
    residual_axis.axvline(0.0, color="gray", linestyle="--")
    residual_axis.set_xlabel("Prediction - target")
    residual_axis.set_ylabel("Normalized density")
    residual_axis.set_title("Residual distribution", loc="left")
    residual_axis.legend(frameon=False)
    profile_axis.axhline(0.0, color="gray", linestyle="--")
    profile_axis.set_xlabel("Target")
    profile_axis.set_ylabel("Bias with residual RMS")
    profile_axis.set_title("Bias and resolution vs target", loc="left")
    profile_axis.legend(frameon=False)
    fig.suptitle(title)
    save_figure(fig, path)


def normalized_hist2d(x: np.ndarray, y: np.ndarray, weight: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    counts = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=weight)[0]
    total = float(np.sum(counts))
    return counts / total if total > 0 else counts


def plot_joint_maps(
    methods: dict[str, dict[tuple[str, str], FeatureSample]],
    leg: str,
    x_feature: str,
    y_feature: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(methods),
        3,
        figsize=(16.5, 4.8 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (label, samples) in enumerate(methods.items()):
        x_sample = samples[(leg, x_feature)]
        y_sample = samples[(leg, y_feature)]
        size = min(len(x_sample.target), len(y_sample.target))
        target_density = normalized_hist2d(x_sample.target[:size], y_sample.target[:size], x_sample.weight[:size], x_edges, y_edges)
        pred_density = normalized_hist2d(x_sample.prediction[:size], y_sample.prediction[:size], x_sample.weight[:size], x_edges, y_edges)
        difference = pred_density - target_density
        for column, (values, panel_title) in enumerate(((target_density, "Target density"), (pred_density, "Prediction density"))):
            positive = values[values > 0]
            norm = LogNorm(vmin=max(float(np.min(positive)), 1e-12), vmax=float(np.max(positive))) if positive.size else None
            mesh = axes[row, column].pcolormesh(x_edges, y_edges, np.ma.masked_where(values.T <= 0, values.T), cmap="viridis", norm=norm)
            fig.colorbar(mesh, ax=axes[row, column], label="Probability / bin")
            axes[row, column].set_title(f"{label}: {panel_title}")
        limit = float(np.max(np.abs(difference))) if difference.size else 1.0
        limit = limit if limit > 0 else 1.0
        mesh = axes[row, 2].pcolormesh(x_edges, y_edges, difference.T, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
        fig.colorbar(mesh, ax=axes[row, 2], label="Prediction - target")
        axes[row, 2].set_title(f"{label}: density difference")
        for axis in axes[row]:
            axis.set_xlabel(x_feature)
            axis.set_ylabel(y_feature)
    fig.suptitle(f"Joint diffusion-target kinematics, leg {leg}")
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    analysis = read_yaml(args.analysis_config)
    comparison = read_yaml(args.comparison_config).get("PredictionComparison", {})
    all_features = list((analysis.get("Normalization") or {}).get("Invisible", {}).keys())
    features = invisible_features(analysis, comparison.get("features"))
    legs = [str(item) for item in comparison.get("legs", ["a", "b"])]
    if len(legs) > 2 or any(leg not in {"a", "b"} for leg in legs):
        raise ValueError("PredictionComparison.legs must be a subset of [a, b].")
    max_events_cfg = comparison.get("max_events_per_method")
    max_events = max_events_cfg if args.max_events is None else args.max_events
    max_events = None if max_events in (None, 0) else int(max_events)
    weight_column = comparison.get("weight_column") or None
    target_source = str(comparison.get("target_source", "auto"))
    if target_source not in {"auto", "x_invisible", "four_vector"}:
        raise ValueError("target_source must be auto, x_invisible, or four_vector.")
    bins_1d = int(comparison.get("bins_1d", 80))
    bins_2d = int(comparison.get("bins_2d", 70))
    profile_bins = int(comparison.get("profile_bins", 20))
    quantiles = tuple(float(value) for value in comparison.get("quantile_range", [0.005, 0.995]))
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("quantile_range must contain two increasing values between 0 and 1.")

    method_entries = [parse_method(spec) for spec in args.method]
    if len({label for label, _ in method_entries}) != len(method_entries):
        raise ValueError("Every --method label must be unique.")
    method_paths = dict(method_entries)
    if len(method_paths) < 2:
        raise ValueError("Pass at least two distinct --method labels.")
    method_samples: dict[str, dict[tuple[str, str], FeatureSample]] = {}
    rows_by_method = {}
    target_sources_by_method = {}
    skipped_files_by_method = {}
    for label, path in method_paths.items():
        files = prediction_files(path)
        print(f"[compare] loading {label}: {len(files)} parquet file(s)", flush=True)
        (
            method_samples[label],
            rows_by_method[label],
            target_sources_by_method[label],
            skipped_files_by_method[label],
        ) = load_method(
            files,
            features,
            all_features,
            legs,
            weight_column,
            max_events,
            target_source,
        )
        print(
            f"[compare] {label}: target source(s)={target_sources_by_method[label]}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    csv_rows = []
    edges_by_feature = {
        feature: feature_edges(
            [method_samples[label][(leg, feature)] for label in method_samples for leg in legs],
            max(bins_1d, bins_2d),
            quantiles,
        )
        for feature in features
    }
    for leg in legs:
        for feature in features:
            samples = {label: values[(leg, feature)] for label, values in method_samples.items()}
            common_edges = edges_by_feature[feature]
            edges_1d = np.linspace(common_edges[0], common_edges[-1], bins_1d + 1)
            edges_2d = np.linspace(common_edges[0], common_edges[-1], bins_2d + 1)
            name = f"leg_{leg}_{feature}"
            plot_distributions(samples, edges_1d, name, args.output_dir / "one_dimensional" / f"{name}.png")
            plot_response_maps(samples, edges_2d, name, args.output_dir / "response_maps" / f"{name}.png")
            plot_residuals_and_profiles(samples, edges_1d, profile_bins, name, args.output_dir / "residuals" / f"{name}.png")
            metrics[name] = {}
            for label, sample in samples.items():
                payload = sample_metrics(sample, edges_1d)
                metrics[name][label] = payload
                csv_rows.append({"leg": leg, "feature": feature, "method": label, **payload})

    joint_features = [str(item) for item in comparison.get("joint_features", [])]
    if len(joint_features) == 2 and all(feature in features for feature in joint_features):
        x_feature, y_feature = joint_features
        x_common = edges_by_feature[x_feature]
        y_common = edges_by_feature[y_feature]
        x_edges = np.linspace(x_common[0], x_common[-1], bins_2d + 1)
        y_edges = np.linspace(y_common[0], y_common[-1], bins_2d + 1)
        for leg in legs:
            plot_joint_maps(
                method_samples,
                leg,
                x_feature,
                y_feature,
                x_edges,
                y_edges,
                args.output_dir / "joint_kinematics" / f"leg_{leg}_{x_feature}_vs_{y_feature}.png",
            )

    summary = {
        "methods": {label: str(path) for label, path in method_paths.items()},
        "rows_read": rows_by_method,
        "target_source_requested": target_source,
        "target_sources_used": target_sources_by_method,
        "skipped_targetless_files": skipped_files_by_method,
        "features": features,
        "legs": legs,
        "max_events_per_method": max_events,
        "weight_column": weight_column,
        "bins_1d": bins_1d,
        "bins_2d": bins_2d,
        "profile_bins": profile_bins,
        "quantile_range": list(quantiles),
        "joint_features": joint_features,
        "metrics": metrics,
    }
    (args.output_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="") as handle:
        fieldnames = ["leg", "feature", "method", "entries", "sum_weight", "bias", "resolution", "mae", "rmse", "pearson", "jsd"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[compare] wrote {args.output_dir / 'comparison_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
