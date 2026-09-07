#!/usr/bin/env python3
"""Plot already-combined measurement uncertainties versus dataset size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


RESERVED_CONFIG_KEYS = {"metrics", "plot"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ASYMMETRIC_LINE = re.compile(
    rf"^\s*(?P<metric>.+?)\s+(?P<value>{NUMBER})\s+"
    rf"\+(?P<err_up>{NUMBER})(?:\s+|/)\-(?P<err_down>{NUMBER})\s*$"
)
SYMMETRIC_LINE = re.compile(
    rf"^\s*(?P<metric>.+?)\s*(?::|=)\s*(?P<value>{NUMBER})\s*"
    rf"(?:±|\+/-)\s*(?P<error>{NUMBER})\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot already-computed combined uncertainty versus dataset size."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_metric(name: object) -> str:
    metric = " ".join(str(name).strip().rstrip(":=").split())
    for old, new in (("C_nn", "Cnn"), ("C_rr", "Crr"), ("C_kk", "Ckk")):
        metric = metric.replace(old, new)
    return re.sub(r"\s*([+-])\s*", r" \1 ", metric).strip()


def as_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def load_table(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    mapping_payload = False
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                mapping_payload = True
                if isinstance(payload.get("rows"), list):
                    payload = payload["rows"]
                    mapping_payload = False
                else:
                    payload = [dict(value, parameter=key) for key, value in payload.items()
                               if isinstance(value, dict)]
            if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
                raise ValueError("JSON must contain a list of measurement rows")
            rows = payload
        elif suffix == ".csv":
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        elif suffix == ".txt":
            return load_combined_text(path)
        else:
            raise ValueError("supported result formats are .json, .csv, and combined results.txt")
    except (OSError, csv.Error, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Could not read result file {path}: {exc}") from exc

    marked = [row for row in rows if as_bool(row.get("is_combined"))
              or str(row.get("channel", "")).lower() == "combined"]
    if marked:
        return marked
    if "combined" in path.stem.lower() or mapping_payload:
        return rows
    raise ValueError(
        f"No combined rows found in {path}; pass an *_combined.json/csv output, "
        "not per-channel measurements"
    )


def load_combined_text(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read result file {path}: {exc}") from exc
    combined = False
    rows: list[dict[str, object]] = []
    for line in lines:
        if line.strip().startswith("Combined fit regions:"):
            combined = True
            continue
        if not combined:
            continue
        match = ASYMMETRIC_LINE.match(line)
        if match:
            rows.append(match.groupdict())
            continue
        match = SYMMETRIC_LINE.match(line)
        if match:
            row = match.groupdict()
            row["err_up"] = row["err_down"] = row.pop("error")
            rows.append(row)
    if not rows:
        raise ValueError(f"No measurements found after 'Combined fit regions:' in {path}")
    return rows


def extract_measurements(path: Path) -> dict[str, dict[str, float]]:
    measurements: dict[str, dict[str, float]] = {}
    for row in load_table(path):
        raw_metric = row.get("parameter", row.get("metric", row.get("name")))
        if raw_metric is None:
            raise ValueError(f"A measurement row in {path} has no parameter/metric/name field")
        metric = canonical_metric(raw_metric)
        try:
            value = float(row["value"])
            err_up = abs(float(row["err_up"]))
            err_down = abs(float(row["err_down"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric {metric!r} in {path} needs numeric value, err_up, and err_down"
            ) from exc
        if not all(math.isfinite(number) for number in (value, err_up, err_down)):
            raise ValueError(f"Metric {metric!r} in {path} contains non-finite values")
        if metric in measurements:
            raise ValueError(f"Duplicate combined metric {metric!r} in {path}")
        measurements[metric] = {
            "value": value,
            "err_up": err_up,
            "err_down": err_down,
            "uncertainty": 0.5 * (err_up + err_down),
        }
    if not measurements:
        raise ValueError(f"No combined measurements could be extracted from {path}")
    return measurements


def load_config(path: Path) -> tuple[list[dict[str, object]], list[str] | None, dict[str, bool]]:
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    metrics = config.get("metrics")
    if metrics is not None and (not isinstance(metrics, list) or not all(isinstance(x, str) for x in metrics)):
        raise ValueError("metrics must be a list of names")
    plot = config.get("plot", {})
    if not isinstance(plot, dict) or set(plot) - {"log_x", "log_y"}:
        raise ValueError("plot accepts only log_x and log_y")
    plot_options = {"log_x": plot.get("log_x", True), "log_y": plot.get("log_y", False)}
    if not all(isinstance(value, bool) for value in plot_options.values()):
        raise ValueError("plot.log_x and plot.log_y must be booleans")

    inputs = []
    for name, raw in config.items():
        if name in RESERVED_CONFIG_KEYS:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Input {name!r} must be a mapping")
        unknown = set(raw) - {"dataset_size", "flag", "path", "dash_line"}
        if unknown:
            raise ValueError(f"Unknown fields for input {name!r}: {sorted(unknown)}")
        if "flag" not in raw or "path" not in raw:
            raise ValueError(f"Input {name!r} requires flag and path")
        dash_line = raw.get("dash_line", False)
        if not isinstance(dash_line, bool):
            raise ValueError(f"Input {name!r} dash_line must be boolean")
        size = raw.get("dataset_size")
        if not dash_line and (isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0):
            raise ValueError(f"Input {name!r} requires a positive dataset_size")
        if not isinstance(raw["path"], str) or not raw["path"].strip():
            raise ValueError(f"Input {name!r} path must be a non-empty string")
        result_path = Path(raw["path"]).expanduser()
        if not result_path.is_absolute():
            result_path = path.parent / result_path
        if not result_path.is_file():
            raise FileNotFoundError(f"Result path for input {name!r} does not exist: {result_path}")
        inputs.append({"input": str(name), "flag": str(raw["flag"]), "dataset_size": size,
                       "path": result_path, "dash_line": dash_line})
    if not inputs:
        raise ValueError("Config contains no result inputs")
    return inputs, None if metrics is None else [canonical_metric(x) for x in metrics], plot_options


def safe_filename(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("._") or "metric"


def main() -> None:
    args = parse_args()
    inputs, requested_metrics, plot_options = load_config(args.config)
    rows = []
    discovered = []
    for item in inputs:
        measurements = extract_measurements(item["path"])
        for metric, result in measurements.items():
            if metric not in discovered:
                discovered.append(metric)
            rows.append({
                "input": item["input"], "flag": item["flag"],
                "dataset_size": item["dataset_size"], "metric": metric,
                **result, "dash_line": item["dash_line"],
            })
    metrics = requested_metrics or discovered
    missing = [metric for metric in metrics if metric not in discovered]
    if missing:
        print(f"[uncertainty-scaling] skipped unavailable metrics: {', '.join(missing)}")
    metrics = [metric for metric in metrics if metric in discovered]
    if not metrics:
        raise ValueError("None of the requested metrics are available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in rows if row["metric"] in metrics]
    csv_path = args.output_dir / "uncertainty_scaling_summary.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["input", "flag", "dataset_size", "metric", "value", "err_up",
                  "err_down", "uncertainty", "dash_line"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    flags = list(dict.fromkeys(row["flag"] for row in selected_rows))
    colors = {flag: plt.get_cmap("tab10")(index % 10) for index, flag in enumerate(flags)}
    for metric in metrics:
        metric_rows = [row for row in selected_rows if row["metric"] == metric]
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for flag in flags:
            points = sorted(
                (row for row in metric_rows if row["flag"] == flag and not row["dash_line"]),
                key=lambda row: row["dataset_size"],
            )
            if points:
                ax.plot([row["dataset_size"] for row in points],
                        [row["uncertainty"] for row in points], marker="o", linewidth=1.8,
                        color=colors[flag], label=flag)
            references = [row for row in metric_rows if row["flag"] == flag and row["dash_line"]]
            for index, row in enumerate(references):
                label = f"{flag} reference" if index == 0 else None
                ax.axhline(row["uncertainty"], linestyle="--", linewidth=1.5,
                           color=colors[flag], label=label)
        if plot_options["log_x"]:
            ax.set_xscale("log")
        if plot_options["log_y"]:
            if any(row["uncertainty"] <= 0 for row in metric_rows):
                raise ValueError(f"Metric {metric!r} has non-positive uncertainty on log-y scale")
            ax.set_yscale("log")
        ax.set(xlabel="Dataset size", ylabel="Combined uncertainty",
               title=f"Uncertainty scaling: {metric}")
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.7)
        ax.legend(frameon=False)
        stem = args.output_dir / f"uncertainty_scaling_{safe_filename(metric)}"
        fig.savefig(stem.with_suffix(".png"), dpi=180)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"[uncertainty-scaling] wrote {len(metrics)} PNG/PDF plot pairs and {csv_path}")


if __name__ == "__main__":
    main()
