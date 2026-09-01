#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from plot_style import channel_latex_label
except Exception:
    def channel_latex_label(channel: str) -> str:
        if channel.startswith("Ztautau_"):
            return channel.removeprefix("Ztautau_").replace("_", r"\_")
        return channel.replace("_", r"\_")

from quantum.observables_builder import get_observable_names

ROOT.gROOT.SetBatch(True)

METHOD_MARKERS = ("o", "D", "^", "s", "P", "X", "v", "*")
METRIC_SPECS = [
    ("diagonal_fraction", "Diagonal fraction", (0.0, 1.0)),
    ("near_diagonal_fraction", "Near-diagonal fraction", (0.0, 1.0)),
    ("mean_abs_bin_offset", "Mean |reco bin - truth bin|", None),
    ("rms_bin_offset", "RMS(reco bin - truth bin)", None),
]
DEFAULT_CHANNEL_ORDER = [
    "Ztautau_pipi",
    "Ztautau_pirho",
    "Ztautau_rhopi",
    "Ztautau_rhorho",
    "Ztautau_ee",
    "Ztautau_emu",
    "Ztautau_mue",
    "Ztautau_mumu",
    "Ztautau_pie",
    "Ztautau_epi",
    "Ztautau_pimu",
    "Ztautau_mupi",
    "Ztautau_rhoe",
    "Ztautau_erho",
    "Ztautau_rhomu",
    "Ztautau_murho",
]
FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
RESPONSE_FILE_RE = re.compile(r"^response_(?P<region>.+)\.root$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize central RooUnfold response matrices across methods and channels. "
            "Writes per-matrix JSON/CSV tables, diagonality summary plots, and one large 2D grid "
            "for each observable."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help=(
            "Method spec Label:/path/to/response_matrices or Label:/path/to/response_<region>.root. "
            "Repeat for Baseline, EveNet, Truth, etc."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix. Writes <prefix>_matrix_metrics.*, <prefix>_summary.*, and per-observable grids.",
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        default=None,
        help="Optional subset of observables. Default: all cos_theta observables in the response files.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Optional subset and display order of channels. Default: standard Ztautau fine-channel order.",
    )
    parser.add_argument(
        "--normalize",
        choices=["total", "truth", "reco"],
        default="total",
        help=(
            "Normalization used in the big 2D matrix panels. "
            "'total' divides by the total matrix yield, "
            "'truth' normalizes each truth column, "
            "'reco' normalizes each reco row."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write JSON/CSV summaries; skip PNG/PDF plots.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics while reading response matrices.",
    )
    return parser.parse_args()


def parse_method_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    methods: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --method '{spec}'. Expected Label:/path/to/response_matrices")
        label, raw_path = spec.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --method '{spec}'. Empty method label.")
        if label in seen:
            raise ValueError(f"Duplicate method label '{label}'.")
        seen.add(label)
        methods.append((label, resolve_response_path(Path(raw_path))))
    return methods


def resolve_response_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    if any(child.is_file() and RESPONSE_FILE_RE.match(child.name) for child in path.iterdir()):
        return path
    raise FileNotFoundError(
        f"No response_*.root files found directly under: {path}\n"
        "Please pass either an exact response_<region>.root file or the response_matrices directory itself."
    )


def canonical_channel_key(region: str, signal: str) -> str:
    for raw in (signal, region):
        channel = normalize_signal_channel(raw)
        if channel is not None:
            return channel
    return signal if signal else region


def canonical_region_key(region: str) -> str:
    return normalize_signal_channel(region) or region


def response_matrix_key(region: str, signal: str) -> str:
    return f"{canonical_region_key(region)}::{canonical_channel_key(region, signal)}"


def response_matrix_label(region: str, signal: str) -> str:
    region_key = canonical_region_key(region)
    signal_key = canonical_channel_key(region, signal)
    if region_key == signal_key:
        return signal_key
    return f"{region_key} -> {signal_key}"


def normalize_signal_channel(name: str) -> str | None:
    raw = name.strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered.startswith("ztautau_"):
        lowered = lowered.removeprefix("ztautau_")
    valid = {
        "pipi", "pirho", "rhopi", "rhorho", "ee", "emu", "mue", "mumu",
        "pie", "epi", "pimu", "mupi", "rhoe", "erho", "rhomu", "murho",
    }
    if lowered in valid:
        return f"Ztautau_{lowered}"
    return None


def extract_region_from_filename(path: Path) -> str | None:
    match = RESPONSE_FILE_RE.match(path.name)
    if match is None:
        return None
    return match.group("region")


def response_observable_names() -> list[str]:
    return [name for name in get_observable_names() if "cos_theta" in name]


def maybe_load_roounfold() -> None:
    candidates: list[str] = []
    roounfold_lib = os.environ.get("ROOUNFOLD_LIB")
    if roounfold_lib:
        candidates.append(roounfold_lib)
    repo_lib = Path(__file__).resolve().parents[1] / "RooUnfold" / "build" / "libRooUnfold.so"
    if repo_lib.exists():
        candidates.append(str(repo_lib))
    candidates.extend(["libRooUnfold.so", "libRooUnfold"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            status = ROOT.gSystem.Load(candidate)
        except Exception:
            continue
        if status >= 0:
            return


def parse_response_key(key_name: str, default_region: str | None) -> tuple[str, str, str] | None:
    for observable in sorted(response_observable_names(), key=len, reverse=True):
        suffix = f"_{observable}"
        if not key_name.endswith(suffix):
            continue
        prefix = key_name[: -len(suffix)]
        if prefix.endswith("_"):
            prefix = prefix[:-1]
        if default_region and prefix.startswith(f"{default_region}_"):
            signal = prefix[len(default_region) + 1 :]
            return default_region, signal, observable
        if "_" in prefix:
            region, signal = prefix.split("_", 1)
            return region, signal, observable
        return default_region or prefix, prefix, observable
    return None


def object_name(value: Any, fallback: str = "<unknown>") -> str:
    get_name = getattr(value, "GetName", None)
    if callable(get_name):
        try:
            return str(get_name())
        except Exception:
            pass
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return fallback


def object_class_name(value: Any) -> str:
    class_name = getattr(value, "ClassName", None)
    if callable(class_name):
        try:
            return str(class_name())
        except Exception:
            pass
    return type(value).__name__


def detach_root_histogram(histogram: Any) -> Any:
    set_directory = getattr(histogram, "SetDirectory", None)
    if callable(set_directory):
        try:
            set_directory(0)
        except Exception:
            pass
    return histogram


def call_root_method(value: Any, method_name: str) -> Any | None:
    method = getattr(value, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def get_response_histogram(response_object: Any, key_name: str) -> Any:
    for method_name in (
        "HresponseNoOverflow",
        "Hresponse",
        "H2D",
        "Mresponse",
    ):
        histogram = call_root_method(response_object, method_name)
        if histogram is not None and hasattr(histogram, "GetNbinsX") and hasattr(histogram, "GetNbinsY"):
            return detach_root_histogram(histogram)
    if hasattr(response_object, "GetNbinsX") and hasattr(response_object, "GetNbinsY"):
        return detach_root_histogram(response_object)
    raise TypeError(
        f"Object '{object_name(response_object, key_name)}' (class {object_class_name(response_object)}) "
        "is not a supported response matrix. RooUnfold dictionaries may be missing; "
        "source setup.sh or set ROOUNFOLD_LIB before running."
    )


def th2_to_numpy(histogram: Any) -> np.ndarray:
    values = np.zeros((histogram.GetNbinsY(), histogram.GetNbinsX()), dtype=np.float64)
    for x_bin in range(1, histogram.GetNbinsX() + 1):
        for y_bin in range(1, histogram.GetNbinsY() + 1):
            values[y_bin - 1, x_bin - 1] = float(histogram.GetBinContent(x_bin, y_bin))
    return values


def normalize_matrix(values: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.array(values, copy=True, dtype=np.float64)
    if mode == "total":
        total = float(np.sum(matrix))
        return matrix / total if total > 0 else matrix
    if mode == "truth":
        denom = np.sum(matrix, axis=0, keepdims=True)
        return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)
    if mode == "reco":
        denom = np.sum(matrix, axis=1, keepdims=True)
        return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)
    raise ValueError(f"Unknown normalization mode: {mode}")


def compute_matrix_metrics(values: np.ndarray) -> dict[str, float]:
    total = float(np.sum(values))
    if total <= 0.0:
        return {
            "total": 0.0,
            "diagonal_fraction": float("nan"),
            "near_diagonal_fraction": float("nan"),
            "mean_abs_bin_offset": float("nan"),
            "rms_bin_offset": float("nan"),
        }
    x_idx = np.arange(values.shape[1], dtype=np.float64)
    y_idx = np.arange(values.shape[0], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_idx, y_idx)
    abs_offset = np.abs(x_grid - y_grid)
    diagonal_sum = float(np.trace(values))
    near_diagonal_sum = float(np.sum(values[abs_offset <= 1]))
    mean_abs_bin_offset = float(np.sum(values * abs_offset) / total)
    rms_bin_offset = float(np.sqrt(np.sum(values * np.square(abs_offset)) / total))
    return {
        "total": total,
        "diagonal_fraction": diagonal_sum / total,
        "near_diagonal_fraction": near_diagonal_sum / total,
        "mean_abs_bin_offset": mean_abs_bin_offset,
        "rms_bin_offset": rms_bin_offset,
    }


def collect_response_rows(methods: list[tuple[str, Path]], selected_observables: set[str] | None, debug: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_label, method_path in methods:
        root_files = [method_path] if method_path.is_file() else sorted(method_path.glob("response_*.root"))
        if not root_files:
            raise FileNotFoundError(f"No response_*.root files found for method '{method_label}' under {method_path}")
        for root_file in root_files:
            default_region = extract_region_from_filename(root_file)
            root_handle = ROOT.TFile.Open(str(root_file), "READ")
            if root_handle is None or root_handle.IsZombie():
                raise OSError(f"Failed to open ROOT file: {root_file}")
            try:
                for key in root_handle.GetListOfKeys():
                    key_name = key.GetName()
                    parsed = parse_response_key(key_name, default_region)
                    if parsed is None:
                        continue
                    region, signal, observable = parsed
                    if selected_observables is not None and observable not in selected_observables:
                        continue
                    response_object = root_handle.Get(key_name)
                    histogram = get_response_histogram(response_object, key_name)
                    values = th2_to_numpy(histogram)
                    metrics = compute_matrix_metrics(values)
                    row = {
                        "method": method_label,
                        "root_file": str(root_file),
                        "region": region,
                        "region_key": canonical_region_key(region),
                        "signal": signal,
                        "channel": canonical_channel_key(region, signal),
                        "matrix_key": response_matrix_key(region, signal),
                        "matrix_label": response_matrix_label(region, signal),
                        "observable": observable,
                        "matrix_shape": [int(values.shape[0]), int(values.shape[1])],
                        "matrix_values": values.tolist(),
                        **metrics,
                    }
                    rows.append(row)
                    if debug:
                        print(
                            "[response-summary] "
                            f"method={method_label} region={region} signal={signal} observable={observable} "
                            f"diag={metrics['diagonal_fraction']:.4f} total={metrics['total']:.4f}",
                            flush=True,
                        )
            finally:
                root_handle.Close()
    return rows


def rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "root_file",
        "region",
        "region_key",
        "signal",
        "channel",
        "matrix_key",
        "matrix_label",
        "observable",
        "total",
        "diagonal_fraction",
        "near_diagonal_fraction",
        "mean_abs_bin_offset",
        "rms_bin_offset",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def channel_sort_key(channel: str) -> tuple[int, str]:
    if channel in DEFAULT_CHANNEL_ORDER:
        return (DEFAULT_CHANNEL_ORDER.index(channel), channel)
    return (len(DEFAULT_CHANNEL_ORDER), channel)


def matrix_label_display(label: str) -> str:
    if " -> " not in label:
        return channel_latex_label(label)
    region_label, signal_label = label.split(" -> ", 1)
    return f"{channel_latex_label(region_label)} -> {channel_latex_label(signal_label)}"


def matrix_sort_key(region_key: str, channel: str, matrix_label: str) -> tuple[tuple[int, str], tuple[int, str], str]:
    return (channel_sort_key(region_key), channel_sort_key(channel), matrix_label)


def observable_sort_key(observable: str) -> tuple[int, str]:
    ordered = response_observable_names()
    if observable in ordered:
        return (ordered.index(observable), observable)
    return (len(ordered), observable)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["matrix_key"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (method, matrix_key), channel_rows in grouped.items():
        representative = channel_rows[0]
        diag_values = np.array([row["diagonal_fraction"] for row in channel_rows], dtype=np.float64)
        near_values = np.array([row["near_diagonal_fraction"] for row in channel_rows], dtype=np.float64)
        mean_offsets = np.array([row["mean_abs_bin_offset"] for row in channel_rows], dtype=np.float64)
        rms_offsets = np.array([row["rms_bin_offset"] for row in channel_rows], dtype=np.float64)
        output.append(
            {
                "method": method,
                "region": representative["region"],
                "region_key": representative["region_key"],
                "signal": representative["signal"],
                "channel": representative["channel"],
                "matrix_key": matrix_key,
                "matrix_label": representative["matrix_label"],
                "num_observables": int(len(channel_rows)),
                "mean_diagonal_fraction": float(np.nanmean(diag_values)),
                "mean_near_diagonal_fraction": float(np.nanmean(near_values)),
                "mean_abs_bin_offset": float(np.nanmean(mean_offsets)),
                "mean_rms_bin_offset": float(np.nanmean(rms_offsets)),
            }
        )
    output.sort(key=lambda row: (row["method"], matrix_sort_key(row["region_key"], row["channel"], row["matrix_label"])))
    return output


def aggregate_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "region",
        "region_key",
        "signal",
        "channel",
        "matrix_key",
        "matrix_label",
        "num_observables",
        "mean_diagonal_fraction",
        "mean_near_diagonal_fraction",
        "mean_abs_bin_offset",
        "mean_rms_bin_offset",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_color(method: str, index: int) -> str:
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("_") or "response"


def observable_output_dir(output_prefix: Path, observable: str) -> Path:
    path = output_prefix.parent / f"{output_prefix.name}_observables" / sanitize_filename(observable)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_canonical_channel_matrix(row: dict[str, Any]) -> bool:
    return row["region_key"] == row["channel"]


def blue_white_cmap() -> Any:
    return plt.get_cmap("Blues")


def plot_diagonal_summary(rows: list[dict[str, Any]], output_path: Path, title: str, value_key: str) -> None:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    matrix_keys = sorted(
        {row["matrix_key"] for row in rows},
        key=lambda key: matrix_sort_key(
            next(row["region_key"] for row in rows if row["matrix_key"] == key),
            next(row["channel"] for row in rows if row["matrix_key"] == key),
            next(row["matrix_label"] for row in rows if row["matrix_key"] == key),
        ),
    )
    matrix_labels = [next(row["matrix_label"] for row in rows if row["matrix_key"] == key) for key in matrix_keys]
    matrix = np.full((len(matrix_keys), len(methods)), np.nan, dtype=np.float64)
    for row in rows:
        y_idx = matrix_keys.index(row["matrix_key"])
        x_idx = methods.index(row["method"])
        matrix[y_idx, x_idx] = float(row[value_key])

    fig_width = max(5.4, 1.3 * len(methods) + 2.4)
    fig_height = max(5.0, 0.45 * len(matrix_keys) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0 if "fraction" in value_key else None, vmax=1.0 if "fraction" in value_key else None, cmap="viridis")
    ax.set_xticks(np.arange(len(methods), dtype=np.float64))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(matrix_keys), dtype=np.float64))
    ax.set_yticklabels([matrix_label_display(label) for label in matrix_labels])
    ax.set_title(title)
    for y_idx, _ in enumerate(matrix_keys):
        for x_idx, method in enumerate(methods):
            value = matrix[y_idx, x_idx]
            if np.isfinite(value):
                ax.text(x_idx, y_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.95)
    cbar.set_label(value_key.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_diagonal_scatter(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    matrix_keys = sorted(
        {row["matrix_key"] for row in rows},
        key=lambda key: matrix_sort_key(
            next(row["region_key"] for row in rows if row["matrix_key"] == key),
            next(row["channel"] for row in rows if row["matrix_key"] == key),
            next(row["matrix_label"] for row in rows if row["matrix_key"] == key),
        ),
    )
    matrix_labels = [next(row["matrix_label"] for row in rows if row["matrix_key"] == key) for key in matrix_keys]
    fig_width = max(7.0, 0.55 * len(matrix_keys) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), dpi=220)
    x = np.arange(len(matrix_keys), dtype=np.float64)
    for index, method in enumerate(methods):
        method_rows = {row["matrix_key"]: row for row in rows if row["method"] == method}
        y = np.array([method_rows[key]["mean_diagonal_fraction"] if key in method_rows else np.nan for key in matrix_keys], dtype=np.float64)
        ax.plot(
            x,
            y,
            marker=METHOD_MARKERS[index % len(METHOD_MARKERS)],
            linewidth=1.6,
            markersize=5.5,
            color=method_color(method, index),
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([matrix_label_display(label) for label in matrix_labels], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean diagonal fraction")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_observable_metric_summaries(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    channels_override: list[str] | None = None,
) -> dict[str, Any]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    method_index = {method: index for index, method in enumerate(methods)}
    observables = sorted({row["observable"] for row in rows}, key=observable_sort_key)
    plot_summary: dict[str, Any] = {}

    for observable in observables:
        plot_dir = observable_output_dir(output_prefix, observable)
        observable_rows = [row for row in rows if row["observable"] == observable and is_canonical_channel_matrix(row)]
        discovered_channels = list(
            dict.fromkeys(
                row["channel"]
                for row in sorted(
                    observable_rows,
                    key=lambda row: (channel_sort_key(row["channel"]), row["method"]),
                )
            )
        )
        if channels_override is None:
            channels = discovered_channels
        else:
            requested = set(channels_override)
            channels = [channel for channel in discovered_channels if channel in requested]
        if not channels:
            continue
        y_base = np.arange(len(channels), dtype=np.float64)
        channel_index = {channel: index for index, channel in enumerate(channels)}

        for metric_key, metric_label, xlim in METRIC_SPECS:
            values = np.array([row[metric_key] for row in observable_rows], dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue

            fig_height = max(4.4, 0.62 * len(channels) + 2.2)
            fig, ax = plt.subplots(figsize=(11.6, fig_height), dpi=200)

            if xlim is None:
                xmin = float(np.nanmin(values[finite]))
                xmax = float(np.nanmax(values[finite]))
                span = xmax - xmin
                pad = max(0.12 * span, 0.04)
                ax.set_xlim(xmin - pad, xmax + pad)
                x_text_value = 1.025
            else:
                ax.set_xlim(*xlim)
                x_text_value = 1.025

            for channel in channels:
                channel_rows = [row for row in observable_rows if row["channel"] == channel]
                channel_rows.sort(key=lambda row: method_index[row["method"]])
                offsets = np.linspace(-0.24, 0.24, len(channel_rows)) if len(channel_rows) > 1 else np.array([0.0])
                for offset, row in zip(offsets, channel_rows):
                    value = float(row[metric_key])
                    if not np.isfinite(value):
                        continue
                    method_i = method_index[row["method"]]
                    y = y_base[channel_index[channel]] + offset
                    color = method_color(row["method"], method_i)
                    marker = METHOD_MARKERS[method_i % len(METHOD_MARKERS)]
                    ax.plot(
                        value,
                        y,
                        marker=marker,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=6.5,
                        linestyle="None",
                    )
                    ax.text(
                        x_text_value,
                        y,
                        f"{value:.4f}",
                        color=color,
                        fontsize=8,
                        va="center",
                        ha="left",
                        transform=ax.get_yaxis_transform(),
                        clip_on=False,
                    )

            if metric_key.endswith("fraction"):
                ax.axvline(1.0, color="#D9D9D9", linewidth=0.8, linestyle=":", zorder=0)
                ax.axvline(0.0, color="#B0B0B0", linewidth=0.8, linestyle="--", zorder=0)
            else:
                ax.axvline(0.0, color="#B0B0B0", linewidth=0.8, linestyle="--", zorder=0)
            ax.text(x_text_value, 1.02, "value", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
            ax.set_yticks(y_base)
            ax.set_yticklabels([channel_latex_label(channel) for channel in channels])
            ax.invert_yaxis()
            ax.grid(axis="y", alpha=0.18, linestyle=":")
            for separator in np.arange(len(channels) - 1, dtype=np.float64) + 0.5:
                ax.axhline(separator, color="#D9D9D9", linewidth=0.8, zorder=0)
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Channel")
            ax.set_title(f"{observable}: {metric_label} by channel and method")

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=METHOD_MARKERS[index % len(METHOD_MARKERS)],
                    color=method_color(method, index),
                    markerfacecolor=method_color(method, index),
                    markeredgecolor=method_color(method, index),
                    markersize=7,
                    linestyle="None",
                    label=method,
                )
                for index, method in enumerate(methods)
            ]
            ax.legend(
                handles=handles,
                title="Method",
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=min(len(handles), 4),
            )
            fig.subplots_adjust(right=0.74, top=0.82, left=0.16, bottom=0.16)

            plot_path = plot_dir / f"{sanitize_filename(metric_key)}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            plot_summary[f"{observable}:{metric_key}"] = {
                "plot": str(plot_path),
                "directory": str(plot_dir),
                "observable": observable,
                "metric": metric_key,
                "num_points": len(observable_rows),
                "methods": methods,
                "channels": channels,
            }

    return plot_summary


def plot_observable_grids(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    normalize: str,
    channels_override: list[str] | None = None,
) -> list[str]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    canonical_rows = [row for row in rows if is_canonical_channel_matrix(row)]
    all_channels = sorted({row["channel"] for row in canonical_rows}, key=channel_sort_key)
    if channels_override is None:
        channels = all_channels
    else:
        requested = set(channels_override)
        channels = [channel for channel in all_channels if channel in requested]
    observables = sorted({row["observable"] for row in rows}, key=observable_sort_key)
    output_paths: list[str] = []

    for observable in observables:
        plot_dir = observable_output_dir(output_prefix, observable)
        observable_rows = [row for row in canonical_rows if row["observable"] == observable and row["channel"] in channels]
        if not observable_rows:
            continue

        value_max = 0.0
        panel_lookup: dict[tuple[str, str], np.ndarray] = {}
        diag_lookup: dict[tuple[str, str], float] = {}
        for row in observable_rows:
            normalized = normalize_matrix(np.asarray(row["matrix_values"], dtype=np.float64), normalize)
            panel_lookup[(row["channel"], row["method"])] = normalized
            diag_lookup[(row["channel"], row["method"])] = float(row["diagonal_fraction"])
            finite_values = normalized[np.isfinite(normalized)]
            if finite_values.size:
                value_max = max(value_max, float(np.max(finite_values)))
        if value_max <= 0:
            value_max = 1.0

        fig_width = max(2.8 * len(methods) + 1.6, 6.0)
        fig_height = max(2.8 * len(channels) + 1.6, 6.0)
        fig, axes = plt.subplots(len(channels), len(methods), figsize=(fig_width, fig_height), dpi=220, squeeze=False)
        cmap = blue_white_cmap()
        for y_idx, channel in enumerate(channels):
            for x_idx, method in enumerate(methods):
                ax = axes[y_idx][x_idx]
                matrix = panel_lookup.get((channel, method))
                if matrix is None:
                    ax.axis("off")
                    continue
                image = ax.imshow(matrix, origin="lower", aspect="equal", cmap=cmap, vmin=0.0, vmax=value_max)
                ax.set_box_aspect(1)
                if y_idx == 0:
                    ax.set_title(method)
                if x_idx == 0:
                    ax.set_ylabel(channel_latex_label(channel))
                ax.set_xticks([])
                ax.set_yticks([])
                diag_value = diag_lookup[(channel, method)]
                ax.text(
                    0.04,
                    0.96,
                    f"D={diag_value:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="black",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.70, "pad": 2.4, "edgecolor": "none"},
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.04)
                cbar = fig.colorbar(image, cax=cax)
                cbar.set_label("Normalized response", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
        fig.suptitle(f"{observable} response matrices ({normalize} normalized)", y=0.995)
        fig.tight_layout()
        png_path = plot_dir / "response_grid.png"
        pdf_path = plot_dir / "response_grid.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        output_paths.extend([str(png_path), str(pdf_path)])
    return output_paths


def json_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        output.append(clean)
    return output


def main() -> None:
    args = parse_args()
    maybe_load_roounfold()
    methods = parse_method_specs(args.method)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected_observables = None if args.observables is None else set(args.observables)
    rows = collect_response_rows(methods, selected_observables, args.debug)
    if not rows:
        raise ValueError("No response matrices matched the requested inputs.")

    if args.channels is not None:
        allowed = set(args.channels)
        rows = [
            row
            for row in rows
            if row["channel"] in allowed or row["region_key"] in allowed or row["matrix_label"] in allowed
        ]
        if not rows:
            raise ValueError("No response matrices remain after applying --channels.")

    rows.sort(
        key=lambda row: (
            observable_sort_key(row["observable"]),
            matrix_sort_key(row["region_key"], row["channel"], row["matrix_label"]),
            row["method"],
        )
    )
    aggregate_rows_by_channel = aggregate_rows(rows)

    matrix_json = args.output_prefix.parent / f"{args.output_prefix.name}_matrix_metrics.json"
    matrix_csv = args.output_prefix.parent / f"{args.output_prefix.name}_matrix_metrics.csv"
    summary_json = args.output_prefix.parent / f"{args.output_prefix.name}_summary.json"
    summary_csv = args.output_prefix.parent / f"{args.output_prefix.name}_summary.csv"

    matrix_json.write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n")
    summary_json.write_text(json.dumps(json_safe(aggregate_rows_by_channel), indent=2, sort_keys=True) + "\n")
    rows_to_csv(rows, matrix_csv)
    aggregate_to_csv(aggregate_rows_by_channel, summary_csv)

    plot_paths: list[str] = []
    plot_summary_json: Path | None = None
    if not args.no_plots:
        plot_summary = plot_observable_metric_summaries(rows, args.output_prefix, args.channels)
        plot_summary_json = args.output_prefix.parent / f"{args.output_prefix.name}_plot_summary.json"
        plot_summary_json.write_text(json.dumps(plot_summary, indent=2, sort_keys=True) + "\n")
        plot_paths.extend(item["plot"] for item in plot_summary.values())
        mean_diag_png = args.output_prefix.parent / f"{args.output_prefix.name}_mean_diagonal_fraction_heatmap.png"
        near_diag_png = args.output_prefix.parent / f"{args.output_prefix.name}_mean_near_diagonal_fraction_heatmap.png"
        scatter_png = args.output_prefix.parent / f"{args.output_prefix.name}_mean_diagonal_fraction_by_channel.png"
        plot_diagonal_summary(
            aggregate_rows_by_channel,
            mean_diag_png,
            "Mean diagonal fraction across response observables",
            "mean_diagonal_fraction",
        )
        plot_diagonal_summary(
            aggregate_rows_by_channel,
            near_diag_png,
            "Mean near-diagonal fraction across response observables",
            "mean_near_diagonal_fraction",
        )
        plot_diagonal_scatter(
            aggregate_rows_by_channel,
            scatter_png,
            "Response matrix diagonal fraction by channel",
        )
        plot_paths.extend([str(mean_diag_png), str(near_diag_png), str(scatter_png)])
        plot_paths.extend(plot_observable_grids(rows, args.output_prefix, args.normalize, args.channels))

    payload = {
        "methods": [{"label": label, "path": str(path)} for label, path in methods],
        "num_matrices": len(rows),
        "matrix_metrics_json": str(matrix_json),
        "matrix_metrics_csv": str(matrix_csv),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "plot_summary_json": str(plot_summary_json) if plot_summary_json is not None else None,
        "plots": plot_paths,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
