#!/usr/bin/env python3
"""Matched-test-event ML4Jets inset for the existing invisible phi target.

The talk's Delta-phi(l,nu)-like variable is the signed, wrapped direction
offset phi(tau) - phi(visible), NOT phi(missing) - phi(lepton). Targets follow
compare_evenet_predictions.py: prefer unnormalized x_invisible (nested or
flattened), otherwise use its visible + target-missing vector helper. Predictions
are the exported evenet_invisible_{a,b}_phi offsets, before QI post-calibration.

Both panels use the intersection of valid truth e/mu legs and identical event
weights. MAE = sum(w * abs(wrap(prediction - truth))) / sum(w), over all selected
legs, including tails outside the displayed range. The density is normalized
by the same full selected weight and bin area; zooming never renormalizes it.

Optional ML4JetsDeltaPhi keys in --analysis-config: bins (70), limit_rad (null
for a symmetric pooled quantile zoom), quantile (0.995), cmap (Blues), and
figsize ([4.6, 2.25], inches). No configuration files need to be edited.
Run --self-test for a small dependency-light check of wrapping and alignment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import awkward as ak
import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from compare_evenet_predictions import (
    angular_target_values,
    four_vector_target_columns,
    invisible_features,
    prediction_files,
    read_yaml,
    to_numpy,
)

ROOT = Path(__file__).resolve().parent
LEGS = ("a", "b")
LABELS = ("Baseline EveNet (5%)", "Population-guided DGPO (5%)")


def wrap_phi(values: np.ndarray) -> np.ndarray:
    """Same [-pi, pi) convention as the comparison and QI-export utilities."""
    return (values + np.pi) % (2 * np.pi) - np.pi


def repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def pipeline_inputs(path: Path) -> tuple[dict, dict[str, list[Path]]]:
    options = read_yaml(path)["predict"]["options"]
    # Pipeline paths are repository-relative, like generate_pipeline_shortcut.py.
    sources = options["converted_parquet"]
    if isinstance(sources, str):
        sources = [sources]
    provenance = {
        "inputs": sorted(str(repository_path(item)) for item in sources),
        "split_fraction": options.get("converted_split_fraction"),
    }
    if any("train" in Path(item).parts or "train-diffusion" in Path(item).parts
           for item in provenance["inputs"]):
        raise ValueError("Use validation/test predictions, not training inputs.")
    grouped = {}
    for file in prediction_files(repository_path(options["output_dir"])):
        if "__evenet_pred" not in file.name:
            raise ValueError(f"Cannot recover the converted-input name from {file}.")
        stem = file.name.split("__evenet_pred", 1)[0]
        grouped.setdefault(stem, []).append(file)
    return provenance, grouped


def truth_columns(present: set[str], phi_index: int) -> tuple[str, set[str], dict | None]:
    if {"x_invisible", "x_invisible_mask"} <= present:
        return "nested", {"x_invisible", "x_invisible_mask"}, None
    flat = {f"x_invisible:{slot}:{phi_index}" for slot in range(2)}
    flat |= {f"x_invisible_mask:{slot}" for slot in range(2)}
    if flat <= present:
        return "flat", flat, None
    fields = four_vector_target_columns(present, list(LEGS))
    if fields is None:
        raise ValueError("Missing truth: need x_invisible targets or visible + target-missing vectors.")
    columns = {name for pair in fields.values() for vector in pair for name in vector.values()}
    return "four_vector", columns, fields


def sample_keys(events: ak.Array, analysis: dict) -> np.ndarray:
    samples = analysis["Samples"]
    if "sample_key" in events.fields:
        keys = to_numpy(events.sample_key, None)
    elif "source_sample_index" in events.fields:
        indices = to_numpy(events.source_sample_index, np.int64)
        if np.any((indices < 0) | (indices >= len(samples))):
            raise ValueError("source_sample_index is outside the analysis Samples order.")
        keys = np.asarray(list(samples))[indices]
    else:
        raise ValueError("Need sample_key or source_sample_index to select signal MC.")
    unknown = set(keys) - samples.keys()
    if unknown:
        raise ValueError(f"Unknown sample keys {sorted(unknown)}.")
    return keys


def load_events(files: list[Path], phi_index: int, weight_column: str | None,
                analysis: dict, signal_keys: list[str]) -> ak.Array:
    chunks = []
    for file in files:
        present = set(pq.ParquetFile(file).schema_arrow.names)
        membership = sorted(present & {"sample_key", "source_sample_index"})
        if not membership:
            raise ValueError(f"{file}: need sample_key or source_sample_index.")
        if not np.any(np.isin(sample_keys(ak.from_parquet(file, columns=membership), analysis), signal_keys)):
            print(f"Skipping non-signal parquet: {file}", flush=True)
            continue
        _, targets, _ = truth_columns(present, phi_index)
        required = targets | {"event_category"}
        required |= {f"evenet_invisible_{leg}_{field}" for leg in LEGS for field in ("phi", "valid")}
        if weight_column:
            required.add(weight_column)
        if not required <= present:
            raise ValueError(f"{file}: missing {sorted(required - present)}")
        columns = required | (present & {"event_index", "sample_key", "source_sample_index"})
        chunks.append(ak.from_parquet(file, columns=sorted(columns)))
    return ak.concatenate(chunks) if chunks else ak.Array([])


def aligned_indices(left: ak.Array, right: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """An event_index is local to an input parquet and, if present, its sample."""
    keys = ["event_index"]
    for name in ("sample_key", "source_sample_index"):
        if name in left.fields and name in right.fields:
            keys.insert(0, name)
            break
    arrays = []
    for events in (left, right):
        key = np.rec.fromarrays([to_numpy(events[name], None) for name in keys], names=keys)
        if len(np.unique(key)) != len(key):
            raise ValueError("Duplicate event identities within a converted-input parquet.")
        arrays.append(key)
    _, i, j = np.intersect1d(*arrays, return_indices=True)
    return i, j


def verify_preserved_rows(left_files: list[Path], right_files: list[Path]) -> None:
    """For older files without IDs, verify every preserved input field exactly.

    predict_evenet.py augments the converted parquet without reordering rows.
    This fallback costs a full input-column read and rejects unverified ordering.
    """
    arrays = []
    for files in (left_files, right_files):
        chunks = []
        for file in files:
            names = pq.ParquetFile(file).schema_arrow.names
            columns = sorted(name for name in names
                             if not name.startswith("evenet_") and name != "event_weight")
            chunks.append(ak.from_parquet(file, columns=columns))
        arrays.append(ak.concatenate(chunks))
    left_form, left_n, left_buffers = ak.to_buffers(ak.to_packed(arrays[0]))
    right_form, right_n, right_buffers = ak.to_buffers(ak.to_packed(arrays[1]))
    if left_n != right_n or left_form != right_form or left_buffers.keys() != right_buffers.keys():
        raise ValueError("No event_index and preserved input schemas/row counts differ; cannot align safely.")
    for key, values in left_buffers.items():
        if not np.array_equal(values, right_buffers[key], equal_nan=values.dtype.kind in "fc"):
            raise ValueError("No event_index and preserved input rows differ; cannot align safely.")
    print("  No event_index: verified identical preserved input rows.", flush=True)


def extract_phi(events: ak.Array, phi_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, _, fields = truth_columns(set(events.fields), phi_index)
    if source == "nested":
        truth = to_numpy(events.x_invisible)[:, :2, phi_index]
        valid = to_numpy(events.x_invisible_mask, bool)[:, :2].copy()
    elif source == "flat":
        truth = np.column_stack([to_numpy(events[f"x_invisible:{slot}:{phi_index}"]) for slot in range(2)])
        valid = np.column_stack([to_numpy(events[f"x_invisible_mask:{slot}"], bool) for slot in range(2)])
    else:
        targets, masks = angular_target_values(events, fields, list(LEGS))
        truth = np.column_stack([targets[leg]["phi"] for leg in LEGS])
        valid = np.column_stack([masks[leg] for leg in LEGS])
    prediction = np.column_stack([to_numpy(events[f"evenet_invisible_{leg}_phi"]) for leg in LEGS])
    valid &= np.column_stack([to_numpy(events[f"evenet_invisible_{leg}_valid"], bool) for leg in LEGS])
    valid &= np.isfinite(truth) & np.isfinite(prediction)
    return wrap_phi(truth), wrap_phi(prediction), valid


def collect_samples(analysis: dict, pipelines: list[Path], max_events: int | None):
    # Reuse the production decay convention from the parent/sibling LEP checkout.
    roots = [Path(os.environ.get("LEP_TREE_ANA_ROOT", ROOT.parent)), ROOT.parent / "lep_tree_ana"]
    lep_root = next((path for path in roots if (path / "utils/tau_decay.py").is_file()), None)
    if lep_root is None:
        raise FileNotFoundError("Set LEP_TREE_ANA_ROOT to the checkout containing utils/tau_decay.py.")
    sys.path.insert(0, str(lep_root))
    from utils.tau_decay import DECAY_MODE_IDS, decode_event_categories

    phi_index = invisible_features(analysis, None).index("phi")
    comparison = read_yaml(ROOT / "config/prediction_comparison.yaml")["PredictionComparison"]
    weight_column = comparison.get("weight_column")
    (left_provenance, left_files), (right_provenance, right_files) = map(pipeline_inputs, pipelines)
    if left_provenance != right_provenance:
        raise ValueError("Pipelines must use the same converted validation/test inputs and split fraction.")
    print(f"Input provenance: {left_provenance}", flush=True)
    if left_files.keys() != right_files.keys():
        raise ValueError("Prediction input-file sets differ; complete both predictions before comparing.")
    chunks = []
    selected_events = 0
    signal_keys = [key for key, config in analysis["Samples"].items()
                   if config.get("is_signal") and not config.get("is_data")]
    for stem in sorted(left_files):
        left = load_events(left_files[stem], phi_index, weight_column, analysis, signal_keys)
        right = load_events(right_files[stem], phi_index, weight_column, analysis, signal_keys)
        if not len(left) and not len(right):
            continue
        if not len(left) or not len(right):
            raise ValueError(f"{stem}: signal sample membership differs between predictions.")
        if "event_index" in left.fields and "event_index" in right.fields:
            i, j = aligned_indices(left, right)
        else:
            verify_preserved_rows(left_files[stem], right_files[stem])
            i = j = np.arange(len(left))
        print(f"{stem}: {len(i)} matched / {len(left)} baseline / {len(right)} DGPO rows", flush=True)
        left, right = left[i], right[j]
        if not len(left):
            continue
        categories = to_numpy(left.event_category, np.int64)
        if not np.array_equal(categories, to_numpy(right.event_category, np.int64)):
            raise ValueError(f"{stem}: matched truth event categories disagree.")
        truth, baseline, valid_left = extract_phi(left, phi_index)
        other_truth, dgpo, valid_right = extract_phi(right, phi_index)
        modes = np.column_stack(decode_event_categories(categories))
        leptonic = np.isin(modes, [DECAY_MODE_IDS["e"], DECAY_MODE_IDS["mu"]])
        keys = sample_keys(left, analysis)
        if not np.array_equal(keys, sample_keys(right, analysis)):
            raise ValueError(f"{stem}: matched sample keys disagree.")
        valid = leptonic & valid_left & valid_right & np.isin(keys, signal_keys)[:, None]
        if np.any(np.abs(wrap_phi(truth[valid] - other_truth[valid])) > 1e-6):
            raise ValueError(f"{stem}: matched phi targets disagree.")
        weights = to_numpy(left[weight_column]) if weight_column else np.ones(len(left))
        other_weights = to_numpy(right[weight_column]) if weight_column else np.ones(len(right))
        if not np.allclose(weights, other_weights, rtol=1e-6, atol=0, equal_nan=True):
            raise ValueError(f"{stem}: matched event weights disagree.")
        if np.any(weights[np.any(valid, axis=1)] < 0):
            raise ValueError("Negative weights cannot define a probability density.")
        valid &= (np.isfinite(weights) & (weights > 0))[:, None]
        rows = np.flatnonzero(np.any(valid, axis=1))
        if max_events:
            rows = rows[:max_events - selected_events]
        mask = valid[rows]
        chunks.append((truth[rows][mask], baseline[rows][mask], dgpo[rows][mask],
                       np.broadcast_to(weights[:, None], truth.shape)[rows][mask]))
        selected_events += len(rows)
        if max_events and selected_events >= max_events:
            break
    if not chunks or not any(len(chunk[0]) for chunk in chunks):
        raise ValueError("No common valid e/mu tau legs were found.")
    samples = tuple(np.concatenate(items) for items in zip(*chunks))
    print(f"Selected {selected_events} matched events, {len(samples[0])} e/mu legs.", flush=True)
    return samples


def plot_samples(truth, baseline, dgpo, weight, output_dir: Path, style: dict):
    bins = int(style.get("bins", 70))
    quantile = float(style.get("quantile", 0.995))
    if bins < 2 or not 0 < quantile <= 1:
        raise ValueError("ML4JetsDeltaPhi needs bins >= 2 and 0 < quantile <= 1.")
    limit = style.get("limit_rad")
    if limit is None:
        # Symmetric, method-neutral zoom; all three arrays participate equally.
        limit = min(np.pi, max(0.01, float(np.quantile(np.abs(np.concatenate(
            [truth, baseline, dgpo])), quantile))))
    limit = float(limit)
    if not 0 < limit <= np.pi:
        raise ValueError("ML4JetsDeltaPhi.limit_rad must be in (0, pi].")
    edges = np.linspace(-limit, limit, bins + 1)
    densities = [np.histogram2d(truth, pred, bins=(edges, edges), weights=weight)[0]
                 / (weight.sum() * (edges[1] - edges[0]) ** 2) for pred in (baseline, dgpo)]
    norm = Normalize(vmin=0, vmax=max(float(d.max()) for d in densities))
    if norm.vmax <= 0:
        raise ValueError("No selected legs in the displayed range; increase limit_rad.")
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.usetex": False,
    }):
        fig = plt.figure(figsize=style.get("figsize", [4.6, 2.25]), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.06)
        axes = [fig.add_subplot(grid[0, 0])]
        axes.append(fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0]))
        for ax, label, prediction, density in zip(axes, LABELS, (baseline, dgpo), densities):
            mesh = ax.pcolormesh(edges, edges, density.T, cmap=style.get("cmap", "Blues"),
                                 norm=norm, shading="flat", rasterized=True)
            ax.plot([-limit, limit], [-limit, limit], color="0.4", lw=0.55, ls="--")
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                   xlabel=r"Truth $\Delta\phi(\ell,\nu)$ [rad]", title=label)
            ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
            ax.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
            mae = float(np.average(np.abs(wrap_phi(prediction - truth)), weights=weight))
            ax.text(0.04, 0.96, f"MAE {mae:.3f} rad", transform=ax.transAxes, va="top",
                    fontsize=6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
            fraction = np.sum(weight[(np.abs(truth) <= limit) & (np.abs(prediction) <= limit)]) / weight.sum()
            print(f"{label}: wrapped MAE={mae:.6g} rad; displayed weight={fraction:.3%}")
        axes[0].set_ylabel(r"Reconstructed $\Delta\phi(\ell,\nu)$ [rad]")
        axes[1].tick_params(labelleft=False)
        cbar = fig.colorbar(mesh, cax=fig.add_subplot(grid[0, 2]))
        cbar.set_label(r"Probability density [rad$^{-2}$]", fontsize=6)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
        cbar.locator = MaxNLocator(4)
        cbar.update_ticks()
        print(f"Shared limits: +/-{limit:.6g} rad; bins: {bins}; full-sample normalization.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            path = output_dir / f"ml4jets_delta_phi_lv.{suffix}"
            fig.savefig(path, dpi=600, facecolor="white")
            print(f"Wrote {path}")
        plt.close(fig)


def self_test():
    assert np.allclose(wrap_phi(np.array([2 * np.pi - 0.02, -2 * np.pi + 0.02])), [-0.02, 0.02])
    left = ak.Array({"event_index": [3, 1, 2], "source_sample_index": [0, 0, 0]})
    right = ak.Array({"event_index": [2, 3, 4], "source_sample_index": [0, 0, 0]})
    i, j = aligned_indices(left, right)
    assert ak.to_list(left[i].event_index) == ak.to_list(right[j].event_index) == [2, 3]
    print("Self-test passed: branch-cut wrapping and reordered event intersection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-config", type=Path, default=ROOT / "config/analysis.yaml")
    parser.add_argument("--baseline-pipeline", type=Path, default=ROOT / "config/pipelines/baseline_pct0p05.yaml")
    parser.add_argument("--dgpo-pipeline", type=Path, default=ROOT / "config/pipelines/DGPO_pct0p05.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ml4jets_final")
    parser.add_argument("--max-events", type=int, help="Cap matched events with valid e/mu legs; 0 or omitted uses all.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_events is not None and args.max_events < 0:
        parser.error("--max-events must be nonnegative")
    analysis = read_yaml(args.analysis_config.expanduser())
    samples = collect_samples(analysis, [args.baseline_pipeline.expanduser(), args.dgpo_pipeline.expanduser()], args.max_events)
    plot_samples(*samples, args.output_dir.expanduser(), analysis.get("ML4JetsDeltaPhi", {}))


if __name__ == "__main__":
    main()
