#!/usr/bin/env python3
"""Final ML4Jets Asimov QI summaries from already-combined fit results.

Read all Baseline/DGPO entries from the uncertainty_scaling YAML, ordered by
dataset_size and method. Add or remove YAML entries to select the plotted series.
plot.full_dataset_size sets the denominator for percentage labels; when omitted,
the largest configured dataset size is used. Relative result paths are relative
to that YAML, as in the existing plotter. No fit, channel combination, or result-file modification is performed.

Precision is (err_up + err_down) / 2. Concurrence signed sensitivity uses the
existing value / uncertainty-toward-zero implementation. Ratios are DGPO /
baseline precision, not significance ratios. Training dataset size controls color; only hatch
distinguishes DGPO. B_Ak, B_An, B_Ar are displayed as B_k, B_n, B_r.

By default, export one row of individual B/C uncertainties with numeric bar labels
in PNG/PDF/SVG, selected and ordered by YAML metrics. Use --layout summary for the
original five figures. The console lists outputs, precision ratios at each paired
dataset size, and a rerun command. C_ij and Cij are aliases; Cij and Cji remain distinct.
Run --self-test for a small check of selection and asymmetric metric handling.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import shlex

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import yaml

from plot_uncertainty_scaling import canonical_metric, extract_measurements, sensitivity_value


# Keep the scientific panels and styling separate from YAML-selected inputs.
COLORS = ("#B98968", "#8196AE", "#8F9F88", "#B69AB0", "#C2AE73", "#62656B")
PARAMETERS = ("Concurrence", "B_Ak", "B_An", "B_Ar", "Ckk", "Cnn", "Crr")
PARAMETER_LABELS = ("Concurrence", "B_k", "B_n", "B_r", "C_kk", "C_nn", "C_rr")
FIGURES = (
    ("ml4jets_concurrence_sensitivity", "Concurrence", ("Concurrence",), "sensitivity"),
    ("ml4jets_concurrence_uncertainty", "Concurrence", ("Concurrence",), "uncertainty"),
    ("ml4jets_polarization_uncertainty", "Polarization", ("B_Ak", "B_An", "B_Ar"), "uncertainty"),
    ("ml4jets_correlation_uncertainty", "Spin correlation", ("Ckk", "Cnn", "Crr"), "uncertainty"),
)
MATH_LABELS = {**{f"B_A{i}": rf"$B_{i}$" for i in "knr"},
               **{f"B_B{i}": rf"$B^{{B}}_{i}$" for i in "knr"},
               **{f"C{i}{j}": rf"$C_{{{i}{j}}}$" for i in "knr" for j in "knr"}}
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


def dataset_size_label(size, full_dataset_size):
    count = f"{size / 1_000_000:g}M" if size >= 1_000_000 else f"{size:g}"
    percentage = f"{100 * size / full_dataset_size:g}%"
    return f"{percentage} ({count})" if size == full_dataset_size else percentage


def series_label(series):
    size, method = series
    return f"{size:g} events {method}"


def selected_inputs(config: dict) -> dict:
    """Use every configured result entry; never silently discard a dataset size."""
    selected = {}
    for name, item in config.items():
        if name in {"metrics", "plot"}:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Input {name!r} must be a mapping")
        size = item.get("dataset_size")
        if (isinstance(size, bool) or not isinstance(size, (int, float))
                or not math.isfinite(size) or size <= 0):
            raise ValueError(f"Input {name!r} requires a positive finite dataset_size")
        method = {"baseline": "Baseline", "dgpo": "DGPO"}.get(str(item.get("flag", "")).strip().lower())
        if method is None:
            raise ValueError(f"Input {name!r} requires flag: Baseline or DGPO")
        key = (size, method)
        if key in selected:
            raise ValueError(f"Duplicate configuration for {series_label(key)}: {name}.")
        selected[key] = (name, item)
    if not selected:
        raise ValueError("Config contains no result inputs")
    return dict(sorted(selected.items()))


def bc_row_metrics(config: dict) -> tuple:
    """Select individual spin coefficients, preserving YAML order."""
    metrics = config.get("metrics")
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise ValueError("bc-row requires a YAML metrics list")
    selected = tuple(canonical_metric(item) for item in metrics
                     if re.fullmatch(r"B_[AB][knr]|C_?[knr]{2}", item.strip()))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("bc-row requires non-empty, unique B/C metrics")
    return selected


def load_results(config_path: Path, parameters=PARAMETERS) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("The config must be a YAML mapping.")
    selected = selected_inputs(config)
    results, errors = {}, []
    for key, (name, item) in selected.items():
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
            missing = [metric for metric in parameters if metric not in measurements]
            if missing:
                raise ValueError("missing required combined parameters: " + ", ".join(missing))
            for metric in parameters:
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


def legends_and_note(fig, results, labels, colors):
    color_handles = [Patch(facecolor=colors[size], edgecolor="0.25", linewidth=0.5,
                           label=label) for size, label in labels.items()]
    methods = [Patch(facecolor="white", edgecolor="0.25", linewidth=0.5,
                     hatch="///" if method == "DGPO" else "", label=method)
               for method in ("Baseline", "DGPO") if any(key[1] == method for key in results)]
    fig.legend(handles=color_handles, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               ncol=min(len(labels), 4))
    rows = math.ceil(len(labels) / 4)
    fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(0.53, 0.99 - rows * 0.065), ncol=2)
    fig.text(0.5, 0.025, "Expected Asimov performance", ha="center", color="0.4", fontsize=7)


def draw_panel(ax, results: dict, title: str, metrics: tuple, field: str, labels, colors):
    heights = []
    sizes = list(labels)
    series = sorted(results)
    for index, (size, method) in enumerate(series):
        values = [results[(size, method)][metric][field] for metric in metrics]
        if metrics == ("Concurrence",):
            paired = all((size, name) in results for name in ("Baseline", "DGPO"))
            offset = {"Baseline": -0.18, "DGPO": 0.18}[method] if paired else 0
            positions = [sizes.index(size) + offset]
            width = 0.32
        else:
            spacing = 0.75 / len(series)
            positions = np.arange(len(metrics)) + (index - (len(series) - 1) / 2) * spacing
            width = spacing * 0.9
        bars = ax.bar(positions, values, width=width, color=colors[size],
                      edgecolor="0.2", linewidth=0.55, hatch="///" if method == "DGPO" else "")
        if metrics == ("Concurrence",):
            ax.bar_label(bars, labels=[f"{value:.3g}" for value in values], padding=3, fontsize=8)
        elif field == "uncertainty":
            ax.bar_label(bars, labels=[np.format_float_positional(
                value, precision=3, fractional=False, trim="-") for value in values],
                padding=3, fontsize=6, rotation=90)
        heights.extend(values)
    if metrics == ("Concurrence",):
        ax.set_xticks(range(len(sizes)), list(labels.values()))
        ax.set_xlabel("Training dataset size")
    else:
        ax.set_xticks(range(len(metrics)), [MATH_LABELS[metric] for metric in metrics])
    ax.set_title(title, loc="left", pad=9)
    ax.set_ylabel(r"Expected sensitivity to zero [$\sigma$]" if field == "sensitivity"
                  else "Combined uncertainty")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    lower, upper = min(heights), max(heights)
    span = max(upper, 0) - min(lower, 0)
    padding = (span or 1) * (0.30 if field == "uncertainty" and metrics != ("Concurrence",) else 0.18)
    ax.set_ylim(min(lower, 0) - (padding if lower < 0 else 0), max(upper, 0) + padding)
    if field == "sensitivity":
        ax.axhline(0, color="0.3", linewidth=0.7)


def make_figures(results: dict, output_dir: Path, formats: list[str], dpi: int,
                 full_dataset_size=None, row_metrics=None) -> list[Path]:
    sizes = sorted({size for size, _ in results})
    full_dataset_size = full_dataset_size or max(sizes)
    labels = {size: dataset_size_label(size, full_dataset_size) for size in sizes}
    palette = [COLORS[index % (len(COLORS) - 1)] for index in range(len(sizes))]
    colors = dict(zip(sizes, palette))
    if full_dataset_size in colors:
        colors[full_dataset_size] = COLORS[-1]
    legend_extra = max(0, math.ceil(len(sizes) / 4) - 1) * 0.065
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, stem):
        for suffix in formats:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi, transparent=False, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    with plt.rc_context(STYLE):
        if row_metrics is not None:
            fig, ax = plt.subplots(figsize=(max(7, len(row_metrics) * 1.15), 3.8))
            fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.73 - legend_extra)
            draw_panel(ax, results, "Polarization and spin correlation", row_metrics,
                       "uncertainty", labels, colors)
            legends_and_note(fig, results, labels, colors)
            save(fig, "ml4jets_bc_uncertainty_row")
            return outputs
        for stem, title, metrics, field in FIGURES:
            fig, ax = plt.subplots(figsize=(max(4.8, len(sizes) * 1.1), 3.6))
            fig.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.73 - legend_extra)
            draw_panel(ax, results, title, metrics, field, labels, colors)
            legends_and_note(fig, results, labels, colors)
            save(fig, stem)
        fig, axes = plt.subplots(1, 3, figsize=(max(11.8, len(sizes) * 2.5), 3.5))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.73 - legend_extra, wspace=0.42)
        for letter, ax, (_, title, metrics, field) in zip("abc", axes, (FIGURES[0], FIGURES[2], FIGURES[3])):
            draw_panel(ax, results, f"({letter}) {title}", metrics, field, labels, colors)
        legends_and_note(fig, results, labels, colors)
        save(fig, "ml4jets_qe_summary_3panel")
    return outputs


def print_ratios(results, full_dataset_size=None, parameters=PARAMETERS):
    sizes = sorted({size for size, _ in results})
    full_dataset_size = full_dataset_size or max(sizes)
    paired = [size for size in sizes if all((size, method) in results for method in ("Baseline", "DGPO"))]
    print("\nCombined uncertainty ratios: DGPO / baseline")
    if not paired:
        print("No dataset sizes have both Baseline and DGPO results.")
        return
    print(f"{'Parameter':<15}" + "".join(f"{dataset_size_label(size, full_dataset_size):>14}" for size in paired))
    for metric in parameters:
        ratios = [results[(size, "DGPO")][metric]["uncertainty"] /
                  results[(size, "Baseline")][metric]["uncertainty"] for size in paired]
        label = PARAMETER_LABELS[PARAMETERS.index(metric)] if metric in PARAMETERS else metric
        print(f"{label:<15}" + "".join(f"{ratio:>14.4f}" for ratio in ratios))


def self_test():
    from contextlib import redirect_stdout
    from io import StringIO
    import tempfile

    series = ((50_000, "Baseline"), (50_000, "DGPO"), (250_000, "Baseline"),
              (250_000, "DGPO"), (500_000, "Baseline"), (500_000, "DGPO"),
              (5_000_000, "Baseline"), (5_000_000, "DGPO"))
    config = {str(index): {"dataset_size": size, "flag": method, "path": "results.txt"}
              for index, (size, method) in enumerate(reversed(series))}
    assert tuple(selected_inputs(config)) == series
    assert dataset_size_label(500_000, 5_000_000) == "10%"
    assert dataset_size_label(100_000, 5_000_000) == "2%"
    config["duplicate"] = config["0"]
    try:
        selected_inputs(config)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate input was accepted")
    del config["duplicate"]
    assert sensitivity_value(2, 4, 0.5) == 4
    assert sensitivity_value(-2, 4, 0.5) == -0.5
    assert sensitivity_value(0, 4, 0.5) == 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result_path = root / "results.txt"
        result_path.write_text("Combined fit regions: test\n" + "\n".join(
            f"{metric} 0.5 +0.2 -0.1" for metric in PARAMETERS))
        results = load_results(config_path)
        assert tuple(results) == series
        assert math.isclose(results[(500_000, "DGPO")]["Concurrence"]["uncertainty"], 0.15)
        labels = {size: dataset_size_label(size, 5_000_000) for size, _ in series}
        colors = dict.fromkeys(labels, "gray")
        fig, ax = plt.subplots()
        draw_panel(ax, results, "test", ("Concurrence",), "uncertainty", labels, colors)
        assert len(ax.patches) == len(series)
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1%", "5%", "10%", "100% (5M)"]
        assert ax.get_yscale() == "linear"
        plt.close(fig)
        output = StringIO()
        with redirect_stdout(output):
            print_ratios(results, 5_000_000)
        assert "10%" in output.getvalue()
        # Exercise all nine independent C entries through the production parser.
        raw_metrics = ["B_Ak", "B_An", "B_Ar"] + [f"C_{i}{j}" for i in "knr" for j in "knr"]
        config["metrics"] = raw_metrics + ["Concurrence", "Ckk + Cnn"]
        row_metrics = bc_row_metrics(config)
        assert len(row_metrics) == 12 and "Ckn" in row_metrics and "Cnk" in row_metrics
        assert canonical_metric("C_nn + C_rk") == "Cnn + Crk"
        try:
            bc_row_metrics({"metrics": ["C_kn", "Ckn"]})
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate aliases were accepted")
        config_path.write_text(yaml.safe_dump(config))
        result_path.write_text("Combined fit regions: test\n" + "\n".join(
            f"{metric} 0.5 +{0.02 * (index + 1)} -{0.01 * (index + 1)}"
            for index, metric in enumerate(raw_metrics)))
        row_results = load_results(config_path, row_metrics)
        assert row_results[series[0]]["Ckn"] != row_results[series[0]]["Cnk"]
        fig, ax = plt.subplots()
        draw_panel(ax, row_results, "test", row_metrics, "uncertainty", labels, colors)
        assert len(ax.patches) == 12 * len(series)
        assert len(ax.texts) == len(ax.patches)
        assert [text.get_text() for text in ax.texts[:3]] == ["0.015", "0.03", "0.045"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [MATH_LABELS[m] for m in row_metrics]
        assert np.allclose([bar.get_height() for bar in ax.patches[:12]],
                           [0.015 * (index + 1) for index in range(12)])
        plt.close(fig)
        result_path.write_text(result_path.read_text().replace("C_rn", "unused"))
        try:
            load_results(config_path, row_metrics)
        except ValueError as exc:
            assert "Crn" in str(exc) and "No figures written" in str(exc)
        else:
            raise AssertionError("A missing off-diagonal measurement was accepted")
        result_path.write_text("Region: test\nConcurrence 0.5 +0.2 -0.1\n")
        try:
            load_results(config_path)
        except ValueError as exc:
            assert "No figures written" in str(exc)
        else:
            raise AssertionError("Per-channel results were accepted as combined")
    print("Self-test passed: YAML selection, parsing, bars, ratios, all nine C entries, missing-metric validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/uncertainty_scaling.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--layout", choices=("summary", "bc-row"), default="bc-row",
                        help="bc-row (default) plots individual B/C entries from YAML metrics in one row")
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
        config = yaml.safe_load(args.config.read_text())
        if not isinstance(config, dict) or not isinstance(config.get("plot", {}), dict):
            raise ValueError("The config and plot options must be YAML mappings")
        full_dataset_size = config.get("plot", {}).get("full_dataset_size")
        if full_dataset_size is not None and (
                isinstance(full_dataset_size, bool) or not isinstance(full_dataset_size, (int, float))
                or not math.isfinite(full_dataset_size) or full_dataset_size <= 0):
            raise ValueError("plot.full_dataset_size must be a positive finite number")
        row_metrics = bc_row_metrics(config) if args.layout == "bc-row" else None
        parameters = row_metrics if row_metrics is not None else PARAMETERS
        results = load_results(args.config, parameters)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    outputs = make_figures(results, args.output_dir, list(dict.fromkeys(args.formats)), args.dpi,
                           full_dataset_size, row_metrics)
    print("\nGenerated files:")
    for path in outputs:
        print(path)
    print_ratios(results, full_dataset_size, parameters)
    print("\nRun command:\n" + shlex.join([
        "python3", str(Path(__file__).resolve()), "--config", str(args.config),
        "--output-dir", str(args.output_dir), "--formats", *args.formats, "--dpi", str(args.dpi),
        "--layout", args.layout,
    ]))


if __name__ == "__main__":
    main()
