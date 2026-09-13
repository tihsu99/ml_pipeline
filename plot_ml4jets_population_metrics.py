#!/usr/bin/env python3
"""ML4Jets marginal W1, raw joint SWD and copula SWD, configured in YAML.

Joint columns are the existing leg/feature outputs on common-valid event rows.
Raw SWD uses target-only standardization. Copula SWD uses independent average
ranks (rank-0.5)/N. Both joint metrics use uniform event weights and identical
fixed random projections for all methods, with deterministic equal-count
subsampling. Optional bootstrap refits target statistics and ranks per replica.
No training, inference, additional candidates or EMA smoothing is performed.

Reuses compare_evenet_predictions.load_method and its own-file truth/masks.
W1 compares weighted empirical populations, not paired residuals. Primary W1
is linear; circular phi W1 is saved as a supplementary diagnostic. Bootstrap intervals resample paired truth/prediction
rows within each method and leg; they do not measure training-seed variability.

Optional posterior_npz (one per method): samples [draw,event,leg,feature], truth
[event,leg,feature], valid [event,leg], weight [event], features and legs string
arrays. These are repeated conditional draws, never different events repurposed
as posterior draws. TARP reports marginal coverage and area between the mean
coverage curve and diagonal (not a p-value or a joint calibration guarantee).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np

from compare_evenet_predictions import FeatureSample, invisible_features, load_method, prediction_files, read_yaml
from plot_ml4jets_final_delta_phi import pipeline_inputs
from prediction_population_metrics import (
    coverage_area, probability_weights, tarp_coverage, wasserstein1,
    target_normalization, empirical_copula, random_projections, sliced_wasserstein,
)
from plot_ml4jets_final_qe_summary import COLORS as QE_COLORS, dataset_size_label

STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
         "figure.facecolor": "white", "savefig.facecolor": "white"}


def relative_path(raw, config_path):
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def w1_summary(sample, period, replicates, rng):
    truth, prediction, weight = sample.target, sample.prediction, sample.weight
    probability_weights(truth, weight)
    value = wasserstein1(truth, prediction, weight, weight, period)
    boot = []
    for _ in range(replicates):
        index = rng.integers(0, len(truth), len(truth))
        if weight[index].sum() <= 0:
            continue
        boot.append(wasserstein1(truth[index], prediction[index], weight[index], weight[index], period))
    low, high = np.quantile(boot, [0.16, 0.84]) if boot else (None, None)
    return {"entries": len(truth), "sum_weight": float(weight.sum()),
            "effective_entries": float(weight.sum() ** 2 / np.sum(weight**2)),
            "w1": value, "bootstrap_low": low, "bootstrap_high": high,
            "bootstrap_valid": len(boot)}


def load_tarp(methods, config_path, features, legs, periods, options):
    """Read explicitly supplied conditional ensembles; absent files stay unavailable."""
    rows, curves = [], []
    if not any(item.get("posterior_npz") for item in methods.values()):
        return rows, curves
    repeats = positive_int(options.get("reference_repeats", 20), "tarp.reference_repeats")
    bins = positive_int(options.get("alpha_bins", 20), "tarp.alpha_bins", 2)
    seed = positive_int(options.get("seed", 42), "tarp.seed", 0)
    alpha = np.linspace(0, 1, bins + 1)
    for label, item in methods.items():
        if not item.get("posterior_npz"):
            continue
        path = relative_path(item["posterior_npz"], config_path)
        with np.load(path, allow_pickle=False) as data:
            samples, truth = data["samples"], data["truth"]
            valid, weights = data["valid"], data["weight"]
            names, leg_names = data["features"].tolist(), data["legs"].tolist()
        if (truth.ndim != 3 or samples.ndim != 4 or samples.shape[1:] != truth.shape
                or samples.shape[0] < 2 or valid.shape != truth.shape[:2]
                or weights.shape != truth.shape[:1] or valid.dtype != bool
                or len(names) != truth.shape[2] or len(leg_names) != truth.shape[1]
                or len(set(names)) != len(names) or len(set(leg_names)) != len(leg_names)):
            raise ValueError(f"{path}: invalid conditional ensemble schema; see script docstring")
        for leg in legs:
            for feature in features:
                slot, column = leg_names.index(leg), names.index(feature)
                mask = valid[:, slot]
                target = truth[mask, slot, column]
                draws = samples[:, mask, slot, column].T
                weight = weights[mask]
                probability_weights(target, weight)
                period = periods[feature]
                bounds = options.get("reference_ranges", {}).get(feature)
                if bounds is None:
                    raise ValueError(f"TARP needs fixed tarp.reference_ranges.{feature}, chosen independently of evaluated truth")
                low, high = map(float, bounds)
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError(f"Invalid reference range for {feature}")
                if period is not None and not np.isclose(high - low, period):
                    raise ValueError(f"Circular references for {feature} must cover one full period")
                rng = np.random.default_rng(seed)
                coverage = np.mean([tarp_coverage(draws, target, rng.uniform(low, high, len(target)),
                                                alpha, weight, period) for _ in range(repeats)], axis=0)
                area = coverage_area(alpha, coverage)
                rows.append({"method": label, "leg": leg, "feature": feature,
                             "entries": len(target), "draws_per_event": samples.shape[0],
                             "reference_repeats": repeats, "coverage_area": area,
                             "max_grid_deviation": float(np.max(np.abs(coverage-alpha))),
                             "source": str(path)})
                curves.extend({"method": label, "leg": leg, "feature": feature,
                               "alpha": float(a), "coverage": float(c)} for a, c in zip(alpha, coverage))
    return rows, curves


def make_tarp_plot(tarp_rows, curves, config, output, formats, dpi):
    features, legs, methods = config["features"], config["legs"], config["methods"]
    display = config.get("feature_labels", {})
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(methods)}
    outputs = []

    def save(fig, stem):
        for suffix in formats:
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append(str(path))
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(legs), len(features), figsize=(3.8 * len(features), 3.1 * len(legs)),
                                 squeeze=False, layout="constrained")
        for i, leg in enumerate(legs):
            for j, feature in enumerate(features):
                ax = axes[i, j]
                ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
                for row in tarp_rows:
                    if (row["leg"], row["feature"]) != (leg, feature):
                        continue
                    points = [p for p in curves if (p["method"], p["leg"], p["feature"]) == (row["method"], leg, feature)]
                    ax.plot([p["alpha"] for p in points], [p["coverage"] for p in points],
                            color=colors[row["method"]], label=f'{row["method"]}: area={row["coverage_area"]:.3f}')
                ax.set(title=f'{display.get(feature, feature)} · leg {leg}', xlabel="Credibility", ylabel="Expected coverage",
                       xlim=(0, 1), ylim=(0, 1))
                ax.legend(fontsize=8)
        fig.suptitle("Marginal TARP coverage · smaller area is better", fontsize=12)
        save(fig, "ml4jets_tarp_coverage")
    return outputs

def equal_count_sample(sample, count, seed):
    index = np.sort(np.random.default_rng(seed).choice(len(sample.target), count, replace=False))
    return FeatureSample(sample.target[index], sample.prediction[index], sample.weight[index])


def evaluate_joint(samples, config):
    options = config.get("joint", {})
    variables = [f"{leg}:{feature}" for leg in config["legs"] for feature in config["features"]]
    requested = options.get("variables") or variables
    if len(set(requested)) != len(requested) or set(requested) - set(variables):
        raise ValueError("joint.variables must be unique existing leg:feature names")
    columns = [variables.index(name) for name in requested]
    target_method = options.get("target_method") or next(
        (name for name, item in config["methods"].items() if item["flag"] == "Baseline"), next(iter(samples)))
    if target_method not in samples:
        raise ValueError(f"Joint target method {target_method!r} is not available")
    cap = positive_int(options.get("max_events", 20000), "joint.max_events", 0)
    count = min(len(sample.target) for sample in samples.values())
    count = min(count, cap) if cap else count
    if count < 2:
        raise ValueError("Joint comparison needs at least two events valid in every requested leg")
    seed = positive_int(config.get("seed", 42), "seed", 0)
    selected = {name: equal_count_sample(sample, count, seed) for name, sample in samples.items()}
    target = selected[target_method].target[:, columns]
    min_std = float(options.get("min_target_std", 1e-12))
    mean, std, keep = target_normalization(target, min_std)
    included = [name for name, use in zip(requested, keep) if use]
    excluded = [{"variable": name, "reason": "negligible target standard deviation", "target_std": float(scale)}
                for name, scale, use in zip(requested, std, keep) if not use]
    excluded += [{"variable": name, "reason": "excluded by joint.variables"} for name in variables if name not in requested]
    predictions = {name: sample.prediction[:, columns][:, keep] for name, sample in selected.items()}
    own_truth = {name: sample.target[:, columns][:, keep] for name, sample in selected.items()}
    target = target[:, keep]
    mean, std = mean[keep], std[keep]
    normalized_target = (target - mean) / std
    normalized = {name: (values - mean) / std for name, values in predictions.items()}
    copula_target = empirical_copula(target)
    copulas = {name: empirical_copula(values) for name, values in predictions.items()}
    projections = positive_int(options.get("projections", 1000), "joint.projections")
    seeds = options.get("projection_seeds", [42])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("joint.projection_seeds must be nonempty and unique")
    seeds = [positive_int(s, "projection seed", 0) for s in seeds]
    batch = positive_int(options.get("projection_batch_size", 16), "projection_batch_size")
    seed_scores = {name: {"raw_swd": [], "copula_swd": []} for name in samples}
    reference_drift = {}
    for seed_index, projection_seed in enumerate(seeds):
        directions = random_projections(len(included), projections, projection_seed)
        for name in samples:
            print(f"Joint {name}: {projections} projections, seed {projection_seed}", flush=True)
            seed_scores[name]["raw_swd"].append(sliced_wasserstein(normalized_target, normalized[name], directions, batch))
            seed_scores[name]["copula_swd"].append(sliced_wasserstein(copula_target, copulas[name], directions, batch))
            if seed_index == 0:
                reference_drift[name] = sliced_wasserstein(normalized_target, (own_truth[name]-mean)/std, directions, batch)
    directions = random_projections(len(included), projections, seeds[0])
    replicas = positive_int(options.get("bootstrap_count", 0), "joint.bootstrap_count", 0)
    boot = {name: {"raw_swd": [], "copula_swd": []} for name in samples}
    rng = np.random.default_rng(seed)
    paired = {name: name == target_method for name in samples}
    for replica in range(replicas):
        if replica % 10 == 0:
            print(f"Joint bootstrap {replica+1}/{replicas}", flush=True)
        target_index = rng.integers(0, count, count)
        bt = target[target_index]
        bm, bs = bt.mean(axis=0), bt.std(axis=0)
        if np.any(bs <= min_std):
            continue
        bt_copula = empirical_copula(bt)
        for name, prediction in predictions.items():
            index = target_index if paired[name] else rng.integers(0, count, count)
            bp = prediction[index]
            boot[name]["raw_swd"].append(sliced_wasserstein((bt-bm)/bs, (bp-bm)/bs, directions, batch))
            boot[name]["copula_swd"].append(sliced_wasserstein(bt_copula, empirical_copula(bp), directions, batch))
    rows = []
    for name in samples:
        item = config["methods"][name]
        for metric, scores in seed_scores[name].items():
            values = boot[name][metric]
            quantiles = np.quantile(values, [0.16, 0.5, 0.84]) if values else [None]*3
            rows.append({"method": name, "flag": item["flag"], "dataset_size": item["dataset_size"],
                         "metric": metric, "value": float(scores[0]), "projection_seed_mean": float(np.mean(scores)),
                         "projection_seed_std": float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                         "bootstrap_low": quantiles[0], "bootstrap_median": quantiles[1], "bootstrap_high": quantiles[2],
                         "bootstrap_valid": len(values), "entries": count,
                         "own_target_to_reference_swd": reference_drift[name]})
    for row in rows:
        base = next((r for r in rows if r["dataset_size"] == row["dataset_size"]
                     and r["flag"] == "Baseline" and r["metric"] == row["metric"]), None)
        row["relative_change"] = row["value"] / base["value"] - 1 if base and base["value"] > 0 else None
        base_name = base["method"] if base else None
        if base_name and boot[base_name][row["metric"]]:
            numerator, denominator = np.asarray(boot[row["method"]][row["metric"]]), np.asarray(boot[base_name][row["metric"]])
            relative = numerator[denominator > 0] / denominator[denominator > 0] - 1
        else:
            relative = []
        interval = np.quantile(relative, [0.16, 0.5, 0.84]) if len(relative) else [None]*3
        row.update(zip(("relative_bootstrap_low", "relative_bootstrap_median", "relative_bootstrap_high"), interval))
    metadata = {"variables_requested": requested, "variables_included": included, "variables_excluded": excluded,
                "target_method": target_method, "target_mean": mean.tolist(), "target_std": std.tolist(),
                "common_event_count": count, "available_joint_counts": {name: len(sample.target) for name, sample in samples.items()},
                "event_weighting": "uniform per event for raw SWD and exact empirical-rank copula SWD",
                "sampling": "deterministic without-replacement subsampling; same count across all methods and target",
                "projection_count": projections, "projection_seeds": seeds,
                "bootstrap_count": replicas, "bootstrap_paired_to_reference": paired,
                "bootstrap_definition": "shared resampled target; preserve target/prediction pairing only for the reference method, otherwise independent rows; refit target-only normalization and ranks",
                "copula_ties": "average ranks; ties may prevent uniform marginals",
                "coordinate_geometry": "existing exported coordinates, no embedding or angular recentering; raw SWD uses target-only standardization",
                "dependence_dimensions": len(included),
                "plot_statistic": "bootstrap median and 16-84 percentile interval when available; point estimate otherwise"}
    print("Joint variables: " + ", ".join(included))
    print("Excluded variables: " + (str(excluded) if excluded else "none"))
    print(f"Joint common event count: {count}; target reference: {target_method}; uniform event weights")
    return rows, metadata, (target, predictions, included)


def make_joint_plots(rows, w1_rows, diagnostic, config, output, formats, dpi):
    sizes = sorted({row["dataset_size"] for row in rows})
    labels = {size: dataset_size_label(size, config.get("full_dataset_size", max(sizes))) for size in sizes}
    colors = {size: QE_COLORS[index % (len(QE_COLORS)-1)] for index, size in enumerate(sizes)}
    if config.get("full_dataset_size", max(sizes)) in colors:
        colors[config.get("full_dataset_size", max(sizes))] = QE_COLORS[-1]
    paths = []

    def save(fig, stem):
        for suffix in formats:
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(max(8.5, len(sizes)*2.1), 3.8), layout="constrained")
        for ax, metric, title, ylabel in zip(axes, ["raw_swd", "copula_swd"],
                ["(a) Joint distribution", "(b) Dependence structure"],
                ["Sliced Wasserstein distance", "Copula sliced Wasserstein distance"]):
            for i, size in enumerate(sizes):
                group = [row for row in rows if row["dataset_size"] == size and row["metric"] == metric]
                for row in group:
                    offset = (0.18 if row["flag"] == "DGPO" else -0.18) if len(group)>1 else 0
                    x = i + offset
                    height = row["bootstrap_median"] if row["bootstrap_median"] is not None else row["value"]
                    ax.bar(x, height, width=0.32, color=colors[size], edgecolor="0.25",
                                  linewidth=0.6, hatch="///" if row["flag"] == "DGPO" else "")
                    label_y = height
                    if row["bootstrap_low"] is not None:
                        ax.vlines(x, row["bootstrap_low"], row["bootstrap_high"], color="0.2", linewidth=0.8)
                        label_y = max(label_y, row["bootstrap_high"])
                    ax.annotate(f'{height:.3g}', (x, label_y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            ax.set(title=title, ylabel=ylabel, xlabel="Training dataset size")
            ax.set_xticks(range(len(sizes)), list(labels.values()))
            ax.set_ylim(bottom=0)
            ax.set_ylim(0, ax.get_ylim()[1]*1.18)
            ax.grid(axis="y", color="0.92")
            ax.set_axisbelow(True)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
        axes[0].legend(handles=[Patch(facecolor="white", edgecolor="0.3", hatch=hatch, label=label)
                                for label, hatch in [("Baseline", ""), ("DGPO", "///")]], fontsize=8)
        subtitle = "Population-level distribution fidelity"
        if any(row["bootstrap_median"] is not None for row in rows):
            subtitle += "\nBars: bootstrap median; intervals: 16–84%"
        fig.suptitle(subtitle, fontsize=10)
        save(fig, "ml4jets_population_swd")
        variables = [f"{leg}:{feature}" for leg in config["legs"] for feature in config["features"]]
        fig, ax = plt.subplots(figsize=(max(6, len(variables)*1.5), 3.5), layout="constrained")
        for i, size in enumerate(sizes):
            group = [row for row in w1_rows if row["dataset_size"] == size and row["flag"] == "DGPO"]
            xs, ys = [], []
            for row in group:
                if row["relative_change"] is not None:
                    xs.append(variables.index(f'{row["leg"]}:{row["feature"]}') + (i-(len(sizes)-1)/2)*0.65/len(sizes))
                    ys.append(100*row["relative_change"])
            if xs:
                ax.scatter(xs, ys, color=colors[size], label=labels[size])
        ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
        names = [config.get("feature_labels", {}).get(feature, feature) + f" · leg {leg}"
                 for leg in config["legs"] for feature in config["features"]]
        ax.set_xticks(range(len(variables)), names)
        ax.set_ylabel(r"$(W_{1,\mathrm{DGPO}}-W_{1,\mathrm{Baseline}})/W_{1,\mathrm{Baseline}}$ [%]")
        ax.set_title("Marginal population agreement · negative = improvement")
        if ax.collections:
            ax.legend(title="Training dataset size", fontsize=8)
        else:
            ax.text(0.5,0.5,"No paired Baseline/DGPO datasets",transform=ax.transAxes,ha="center")
        ax.grid(axis="y", color="0.92")
        save(fig, "ml4jets_population_w1")
        pair = config.get("joint", {}).get("diagnostic_pair")
        if pair:
            target, predictions, included = diagnostic
            if len(pair) != 2 or any(name not in included for name in pair):
                print(f"Skipping 2D diagnostic: pair {pair} is unavailable after variable selection")
            else:
                slot = [included.index(name) for name in pair]
                requested_size = config.get("joint", {}).get("diagnostic_dataset_size")
                paired_sizes = [size for size in sizes if {row["flag"] for row in rows if row["dataset_size"]==size} == {"Baseline", "DGPO"}]
                chosen_size = requested_size if requested_size is not None else (paired_sizes[0] if paired_sizes else None)
                methods = [name for name, item in config["methods"].items() if item["dataset_size"]==chosen_size]
                if chosen_size not in paired_sizes:
                    print("Skipping 2D diagnostic: no complete Baseline/DGPO pair at requested size")
                else:
                    clouds = [target[:,slot]] + [predictions[name][:,slot] for name in methods]
                    limits = np.concatenate(clouds)
                    bins = positive_int(config.get("joint", {}).get("diagnostic_bins", 40), "diagnostic_bins", 2)
                    edges = [np.linspace(limits[:,i].min(), limits[:,i].max(), bins+1) for i in range(2)]
                    densities = [np.histogram2d(cloud[:,0],cloud[:,1],bins=edges,density=True)[0] for cloud in clouds]
                    fig, axes = plt.subplots(1,len(clouds),figsize=(3*len(clouds),3),layout="constrained")
                    maximum = max(density.max() for density in densities)
                    for ax, density, label in zip(axes, densities, ["Target"]+[config["methods"][name]["flag"] for name in methods]):
                        mesh = ax.pcolormesh(*edges, density.T, vmin=0, vmax=maximum, cmap="Blues", rasterized=True)
                        pair_labels = [config.get("feature_labels", {}).get(name.split(":")[1], name.split(":")[1])
                                       + f" · leg {name.split(':')[0]}" for name in pair]
                        ax.set(title=label, xlabel=pair_labels[0], ylabel=pair_labels[1])
                    fig.colorbar(mesh,ax=axes,label="Normalized density")
                    fig.suptitle(f"Training dataset size: {labels[chosen_size]}")
                    save(fig,"ml4jets_population_joint_2d")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/ml4jets_population_metrics.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
    parser.add_argument("--projections", type=int)
    parser.add_argument("--projection-seed", type=int)
    parser.add_argument("--bootstrap-count", type=int)
    parser.add_argument("--joint-max-events", type=int)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_yaml(config_path)
    joint_options = config.setdefault("joint", {})
    for value, key in [(args.projections,"projections"),(args.bootstrap_count,"bootstrap_count"),(args.joint_max_events,"max_events")]:
        if value is not None:
            joint_options[key] = value
    if args.projection_seed is not None:
        joint_options["projection_seeds"] = [args.projection_seed]
    analysis = read_yaml(relative_path(config["analysis_config"], config_path))
    comparison = read_yaml(relative_path(config["comparison_config"], config_path))["PredictionComparison"] if config.get("comparison_config") else {}
    features = invisible_features(analysis, config.get("features", comparison.get("features")))
    config["features"] = features
    legs = config.setdefault("legs", comparison.get("legs", ["a", "b"]))
    if not legs or len(set(legs)) != len(legs) or set(legs) - {"a", "b"}:
        raise ValueError("legs must select a and/or b without duplicates")
    if len(set(features)) != len(features):
        raise ValueError("features must not contain duplicates")
    size_config = read_yaml(relative_path(config["dataset_sizes_config"], config_path)) if config.get("dataset_sizes_config") else {}
    keys = set()
    if not config.get("methods"):
        raise ValueError("Configure at least one prediction method")
    for name, item in config["methods"].items():
        entry = size_config.get(item.get("size_key", name), {})
        item.setdefault("dataset_size", entry.get("dataset_size"))
        item.setdefault("flag", entry.get("flag"))
        size = item["dataset_size"]
        if isinstance(size, bool) or not isinstance(size, (int,float)) or not np.isfinite(size) or size<=0:
            raise ValueError(f"{name}: requires positive dataset_size, directly or via dataset_sizes_config/size_key")
        if item["flag"] not in {"Baseline","DGPO"}:
            raise ValueError(f"{name}: flag must be Baseline or DGPO")
        key = (size,item["flag"])
        if key in keys:
            raise ValueError(f"Duplicate dataset_size/flag: {key}")
        keys.add(key)
    config["methods"] = dict(sorted(config["methods"].items(),key=lambda pair:(pair[1]["dataset_size"],pair[1]["flag"])))
    config.setdefault("full_dataset_size", max(item["dataset_size"] for item in config["methods"].values()))
    if not np.isfinite(config["full_dataset_size"]) or config["full_dataset_size"]<=0:
        raise ValueError("full_dataset_size must be positive and finite")
    replicates = positive_int(config.get("bootstrap_replicates", 0), "bootstrap_replicates", 0)
    seed = positive_int(config.get("seed", 42), "seed", 0)
    maximum = positive_int(config.get("max_events_per_method", comparison.get("max_events_per_method", 0)), "max_events_per_method", 0) or None
    dpi = positive_int(config.get("dpi", 600), "dpi")
    formats = args.formats or config.get("formats", ["png", "pdf"])
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("formats must select png, pdf and/or svg")
    periods = {feature: config.get("circular_periods", {}).get(feature) for feature in features}
    rows, provenance, truth_samples, joint_samples, skipped_methods = [], {}, {}, {}, {}
    common_provenance = None
    for label, item in config["methods"].items():
        if ("pipeline" in item) == ("path" in item):
            raise ValueError(f"{label}: specify exactly one of pipeline or path")
        if "pipeline" in item:
            pipeline = relative_path(item["pipeline"], config_path)
            if not pipeline.is_file():
                raise FileNotFoundError(f"Pipeline config does not exist: {pipeline}")
        try:
            if "pipeline" in item:
                inputs, groups = pipeline_inputs(pipeline)
                if common_provenance is not None and inputs != common_provenance:
                    raise ValueError("Pipeline test-input provenance or split fractions differ")
                common_provenance = inputs
                files = sorted(file for group in groups.values() for file in group)
            else:
                files = prediction_files(relative_path(item["path"], config_path))
                inputs = "Direct path: test-input provenance is supplied by the caller"
        except FileNotFoundError as exc:
            if not config.get("skip_missing_inputs", False):
                raise
            skipped_methods[label] = str(exc)
            print(f"Skipping unavailable prediction {label}: {exc}", flush=True)
            continue
        print(f"Loading {label}: {len(files)} parquet files", flush=True)
        samples, count, sources, skipped, joint = load_method(
            files, features, invisible_features(analysis, None), legs,
            config.get("weight_column", comparison.get("weight_column")), maximum,
            config.get("target_source", comparison.get("target_source", "auto")), return_joint=True)
        provenance[label] = {"inputs": inputs, "files": [str(p) for p in files], "rows_read": count,
                             "target_sources": sources, "skipped_targetless_files": skipped,
                             "joint_valid_events": len(joint.target), "joint_excluded_rows": count-len(joint.target)}
        truth_samples[label], joint_samples[label] = samples, joint
    if not truth_samples:
        raise ValueError("No available prediction inputs; no figures written")
    config["methods"] = {name:item for name,item in config["methods"].items() if name in truth_samples}
    for label, samples in truth_samples.items():
        item = config["methods"][label]
        for leg in legs:
            for feature in features:
                key = (leg,feature)
                peers = [name for name,other in config["methods"].items() if other["dataset_size"]==item["dataset_size"]]
                common_count = min(len(truth_samples[name][key].target) for name in peers)
                sample = equal_count_sample(samples[key], common_count, seed)
                row = {"method":label,"flag":item["flag"],"dataset_size":item["dataset_size"],"leg":leg,"feature":feature,
                       "geometry":"linear", "available_entries":len(samples[key].target),
                       **w1_summary(sample, None, replicates, np.random.default_rng(seed))}
                row["circular_w1"] = wasserstein1(sample.target,sample.prediction,sample.weight,sample.weight,periods[feature]) if periods[feature] else None
                rows.append(row)
    for row in rows:
        base = next((other for other in rows if other["dataset_size"]==row["dataset_size"] and other["flag"]=="Baseline"
                     and (other["leg"],other["feature"])==(row["leg"],row["feature"])),None)
        reference = base["w1"] if base else None
        row["ratio_to_baseline"] = row["w1"]/reference if reference and reference>0 else None
        row["relative_change"] = row["ratio_to_baseline"]-1 if row["ratio_to_baseline"] is not None else None
        if base:
            left,right = truth_samples[base["method"]][(row["leg"],row["feature"])],truth_samples[row["method"]][(row["leg"],row["feature"])]
            row["truth_population_w1_to_baseline"] = wasserstein1(left.target,right.target,left.weight,right.weight)
        else:
            row["truth_population_w1_to_baseline"] = None
    joint_rows, joint_metadata, diagnostic = evaluate_joint(joint_samples, config)
    tarp_rows, curves = load_tarp(config["methods"], config_path, features, legs, periods, config.get("tarp", {}))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"population_w1.csv", rows)
    write_csv(output/"population_swd.csv", joint_rows)
    if tarp_rows:
        write_csv(output/"tarp_metrics.csv",tarp_rows)
        write_csv(output/"tarp_coverage.csv",curves)
    outputs = make_joint_plots(joint_rows,rows,diagnostic,config,output,formats,dpi)
    if tarp_rows:
        outputs += make_tarp_plot(tarp_rows,curves,config,output,formats,dpi)
    summary = {"config":config,"provenance":provenance,"skipped_methods":skipped_methods,
               "w1":rows,"joint_metrics":joint_rows,"joint":joint_metadata,"tarp":tarp_rows,
               "w1_definition":"exact linear empirical W1 with configured event weights; circular phi W1 retained as an additional diagnostic",
               "w1_sampling":"same per-leg event count within each training size, deterministic paired truth/prediction subsampling",
               "relative_change_definition":"(DGPO-Baseline)/Baseline; negative is improvement; null for absent/zero baseline",
               "tarp_status":{name:"computed" if item.get("posterior_npz") else "unavailable: one candidate per event"
                              for name,item in config["methods"].items()}, "outputs":outputs}
    (output/"population_metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for size in sorted({row["dataset_size"] for row in joint_rows}):
        print(f"\nTraining size: {dataset_size_label(size,config['full_dataset_size'])}")
        for metric in ["raw_swd","copula_swd"]:
            group = [row for row in joint_rows if row["dataset_size"]==size and row["metric"]==metric]
            print(metric+":")
            for row in group:
                print(f'  {row["flag"]} = {row["value"]:.6g}')
                if row["projection_seed_std"] is not None:
                    print(f'    projection seeds: {row["projection_seed_mean"]:.6g} +/- {row["projection_seed_std"]:.3g}')
                if row["bootstrap_median"] is not None:
                    print(f'    bootstrap median: {row["bootstrap_median"]:.6g}; 16-84%: [{row["bootstrap_low"]:.6g}, {row["bootstrap_high"]:.6g}]')
            dgpo = next((row for row in group if row["flag"]=="DGPO"),None)
            print(f'  relative change = {100*dgpo["relative_change"]:.2f}%' if dgpo and dgpo["relative_change"] is not None else "  relative change = unavailable")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
