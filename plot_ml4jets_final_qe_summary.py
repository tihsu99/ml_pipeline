#!/usr/bin/env python3
"""Final ML4Jets Asimov QI summaries from already-combined fit results.

Read only the five requested training dataset size/method combinations from the existing
uncertainty_scaling YAML. Other dataset sizes are excluded before their files are
opened. Relative result paths are relative to that YAML, as in the existing
plotter. No fit, channel combination, or result-file modification is performed.

Precision is (err_up + err_down) / 2. Concurrence signed sensitivity uses the
existing value / uncertainty-toward-zero implementation. Ratios are DGPO /
baseline precision, not significance ratios. Training dataset size controls color; only hatch
distinguishes DGPO. B_Ak, B_An, B_Ar are displayed as B_k, B_n, B_r.

The five figures are exported in PNG/PDF/SVG by default. The console lists all
outputs, the seven precision ratios at each reduced dataset size, and a rerun command.
Run --self-test for a small check of selection and asymmetric metric handling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import yaml

from plot_uncertainty_scaling import extract_measurements, sensitivity_value


# Isolate the talk's scientific selection and styling from the plotting code.
DATASET_SIZES = (50_000, 250_000, 5_000_000)
DATASET_SIZE_LABELS = {50_000: "1%", 250_000: "5%", 5_000_000: "100% (5M)"}
COLORS = {50_000: "#B98968", 250_000: "#8196AE", 5_000_000: "#62656B"}
SERIES = ((50_000, "Baseline"), (50_000, "DGPO"),
          (250_000, "Baseline"), (250_000, "DGPO"), (5_000_000, "Baseline"))
PARAMETERS = ("Concurrence", "B_Ak", "B_An", "B_Ar", "Ckk", "Cnn", "Crr")
PARAMETER_LABELS = ("Concurrence", "B_k", "B_n", "B_r", "C_kk", "C_nn", "C_rr")
FIGURES = (
    ("ml4jets_concurrence_sensitivity", "Concurrence", ("Concurrence",), "sensitivity"),
    ("ml4jets_concurrence_uncertainty", "Concurrence", ("Concurrence",), "uncertainty"),
    ("ml4jets_polarization_uncertainty", "Polarization", ("B_Ak", "B_An", "B_Ar"), "uncertainty"),
    ("ml4jets_correlation_uncertainty", "Spin correlation", ("Ckk", "Cnn", "Crr"), "uncertainty"),
)
MATH_LABELS = {"B_Ak": r"$B_k$", "B_An": r"$B_n$", "B_Ar": r"$B_r$",
               "Ckk": r"$C_{kk}$", "Cnn": r"$C_{nn}$", "Crr": r"$C_{rr}$"}
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.axisbelow": True, "axes.grid": False, "legend.frameon": False,
    "hatch.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
}


def series_label(series):
    size, method = series
    return f"{DATASET_SIZE_LABELS[size]} {method}"


def selected_inputs(config: dict) -> dict:
    """Filter before checking paths, so excluded dataset-size files are never needed."""
    selected = {}
    for name, item in config.items():
        if not isinstance(item, dict):
            continue
        size = item.get("dataset_size")
        method = str(item.get("flag", "")).strip().lower()
        key = (size, {"baseline": "Baseline", "dgpo": "DGPO"}.get(method))
        if key not in SERIES:
            continue
        if key in selected:
            raise ValueError(f"Duplicate configuration for {series_label(key)}: {name}.")
        selected[key] = (name, item)
    missing = [series_label(key) for key in SERIES if key not in selected]
    if missing:
        raise ValueError("Missing required configurations: " + ", ".join(missing))
    return selected


def load_results(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("The config must be a YAML mapping.")
    selected = selected_inputs(config)
    results, errors = {}, []
    for key in SERIES:
        name, item = selected[key]
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            errors.append(f"{series_label(key)} ({name}): missing result path")
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        try:
            # Reuse combined-text/JSON/CSV parsing, aliases, uncertainty and sensitivity.
            measurements = extract_measurements(path)
            missing = [metric for metric in PARAMETERS if metric not in measurements]
            if missing:
                raise ValueError("missing required combined parameters: " + ", ".join(missing))
            for metric in PARAMETERS:
                row = measurements[metric]
                if row["err_up"] <= 0 or row["err_down"] <= 0:
                    raise ValueError(f"{metric} needs strictly positive asymmetric uncertainties")
                if not all(math.isfinite(row[field]) for field in ("uncertainty", "sensitivity")):
                    raise ValueError(f"{metric} has non-finite derived metrics")
            results[key] = measurements
            print(f"Source {series_label(key)}: {path.resolve()}")
        except (ValueError, OSError) as exc:
            errors.append(f"{series_label(key)} ({name}): {exc}")
    if errors:
        raise ValueError("No figures written; input validation failed:\n  " + "\n  ".join(errors))
    return results


def legends_and_note(fig, combined=False):
    colors = [Patch(facecolor=COLORS[size], edgecolor="0.25", linewidth=0.5,
                    label=DATASET_SIZE_LABELS[size]) for size in DATASET_SIZES]
    methods = [Patch(facecolor="white", edgecolor="0.25", linewidth=0.5,
                     hatch="///" if method == "DGPO" else "", label=method)
               for method in ("Baseline", "DGPO")]
    # Separate keys for the two independent visual encodings.
    if combined:
        fig.legend(handles=colors, loc="upper center", bbox_to_anchor=(0.34, 0.99), ncol=3)
        fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(0.71, 0.99), ncol=2)
    else:
        fig.legend(handles=colors, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=3)
        fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(0.53, 0.91), ncol=2)
    fig.text(0.5, 0.025, "Expected Asimov performance", ha="center", color="0.4", fontsize=7)


def draw_panel(ax, results: dict, title: str, metrics: tuple, field: str):
    heights = []
    for index, (size, method) in enumerate(SERIES):
        values = [results[(size, method)][metric][field] for metric in metrics]
        if metrics == ("Concurrence",):
            positions = [DATASET_SIZES.index(size) + ({"Baseline": -0.18, "DGPO": 0.18}[method]
                                              if size != DATASET_SIZES[-1] else 0)]
            width = 0.32
        else:
            positions = np.arange(len(metrics)) + (index - 2) * 0.15
            width = 0.135
        bars = ax.bar(positions, values, width=width, color=COLORS[size],
                      edgecolor="0.2", linewidth=0.55, hatch="///" if method == "DGPO" else "")
        if metrics == ("Concurrence",):
            ax.bar_label(bars, labels=[f"{value:.3g}" for value in values], padding=3, fontsize=8)
        heights.extend(values)
    if metrics == ("Concurrence",):
        ax.set_xticks(range(3), [DATASET_SIZE_LABELS[size] for size in DATASET_SIZES])
        ax.set_xlabel("Training dataset size")
    else:
        ax.set_xticks(range(len(metrics)), [MATH_LABELS[metric] for metric in metrics])
    ax.set_title(title, loc="left", pad=9)
    ax.set_ylabel(r"Expected sensitivity to zero [$\sigma$]" if field == "sensitivity"
                  else "Combined uncertainty")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    lower, upper = min(heights), max(heights)
    span = max(upper, 0) - min(lower, 0)
    padding = (span or 1) * 0.18
    ax.set_ylim(min(lower, 0) - (padding if lower < 0 else 0), max(upper, 0) + padding)
    if field == "sensitivity":
        ax.axhline(0, color="0.3", linewidth=0.7)
    elif upper / lower > 30:
        # A shared nonlinear physical axis retains zero and avoids invisible small bars.
        ax.set_yscale("symlog", linthresh=lower)
        ax.set_ylim(0, upper * 1.8)
        ax.set_ylabel("Combined uncertainty\n(shared symlog scale)")
        print(f"{title}: uncertainty range spans {upper / lower:.3g}x; using a common "
              f"physical symlog axis, linear below {lower:.3g}. No per-parameter rescaling.")


def make_figures(results: dict, output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, stem):
        for suffix in formats:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, transparent=False, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    with plt.rc_context(STYLE):
        for stem, title, metrics, field in FIGURES:
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            fig.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.73)
            draw_panel(ax, results, title, metrics, field)
            legends_and_note(fig)
            save(fig, stem)
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.79, wspace=0.42)
        for letter, ax, (_, title, metrics, field) in zip("abc", axes, (FIGURES[0], FIGURES[2], FIGURES[3])):
            draw_panel(ax, results, f"({letter}) {title}", metrics, field)
        legends_and_note(fig, combined=True)
        save(fig, "ml4jets_qe_summary_3panel")
    return outputs


def print_ratios(results):
    print("\nCombined uncertainty ratios: DGPO / baseline")
    print(f"{'Parameter':<15} {'1%':>10} {'5%':>10}")
    for metric, label in zip(PARAMETERS, PARAMETER_LABELS):
        ratios = [results[(size, "DGPO")][metric]["uncertainty"] /
                  results[(size, "Baseline")][metric]["uncertainty"] for size in DATASET_SIZES[:2]]
        print(f"{label:<15} {ratios[0]:>10.4f} {ratios[1]:>10.4f}")


def self_test():
    config = {str(index): {"dataset_size": size, "flag": method, "path": "unused"}
              for index, (size, method) in enumerate(SERIES)}
    config["excluded"] = {"dataset_size": 500_000, "flag": "Baseline", "path": "absent"}
    assert tuple(selected_inputs(config)) == SERIES
    assert sensitivity_value(2, 4, 0.5) == 4
    assert sensitivity_value(-2, 4, 0.5) == -0.5
    assert sensitivity_value(0, 4, 0.5) == 0
    del config["1"]
    try:
        selected_inputs(config)
    except ValueError as exc:
        assert "1% DGPO" in str(exc)
    else:
        raise AssertionError("Missing 1% DGPO was not rejected")
    print("Self-test passed: exact five-series selection, required DGPO, signed asymmetric sensitivity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/uncertainty_scaling.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        results = load_results(args.config)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    outputs = make_figures(results, args.output_dir, list(dict.fromkeys(args.formats)), args.dpi)
    print("\nGenerated files:")
    for path in outputs:
        print(path)
    print_ratios(results)
    print("\nRun command:\n" + shlex.join([
        "python3", str(Path(__file__).resolve()), "--config", str(args.config),
        "--output-dir", str(args.output_dir), "--formats", *args.formats, "--dpi", str(args.dpi),
    ]))


if __name__ == "__main__":
    main()
