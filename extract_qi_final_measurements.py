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

try:
    from plot_style import channel_latex_label, method_color
except Exception:
    def channel_latex_label(channel: str) -> str:
        if channel == "combined":
            return r"$\bf{Combined}$"
        if channel.startswith("Ztautau_"):
            return channel.removeprefix("Ztautau_").replace("_", r"\_")
        return channel.replace("_", r"\_")

    _FALLBACK_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    def method_color(method: str, index: int) -> str:
        return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


RESULT_LINE_ASYMMETRIC_RE = re.compile(
    r"^\s*(?P<name>[^:]+):\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"\+(?P<err_up>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"/-(?P<err_down>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
RESULT_LINE_ASYMMETRIC_PLAIN_RE = re.compile(
    r"^\s*(?P<name>\S(?:.*\S)?)\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"\+(?P<err_up>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"-(?P<err_down>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
RESULT_LINE_SYMMETRIC_RE = re.compile(
    r"^\s*(?P<name>[^:=]+)\s*(?::|=)\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?:±|\+/-)\s*"
    r"(?P<err>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
SECTION_RE = re.compile(
    r"(?:(?P<source>Unfolded|Truth|Final|Nominal)\s+)?"
    r"(?P<group>B and C matrices|BC matrices|Quantum results|Final metrics|Metrics):$",
    re.IGNORECASE,
)
UNFOLDING_HEADER_RE = re.compile(
    r"Unfolding results for signal (?P<signal>.+?) in region (?P<region>.+):$"
)

METHOD_MARKERS = ("o", "D", "^", "s", "P", "X", "v", "*")
DEFAULT_IGNORED_REGIONS = {"hadhad"}
COMBINED_CHANNEL_KEY = "combined"
COMBINED_REGION = "combined"
COMBINED_SIGNAL = "combined"

CHANNEL_ORDER = [
    "Ztautau_pipi",
    "Ztautau_pirho",
    "Ztautau_rhopi",
    "Ztautau_rhorho",
    "Ztautau_ee",
    "Ztautau_emu",
    "Ztautau_mue",
    "Ztautau_mumu",
    COMBINED_CHANNEL_KEY,
]

CANONICAL_SIGNAL_CHANNELS = {
    "pipi",
    "pirho",
    "rhopi",
    "rhorho",
    "ee",
    "emu",
    "mue",
    "mumu",
    "pie",
    "epi",
    "pimu",
    "mupi",
    "rhoe",
    "erho",
    "rhomu",
    "murho",
}

QUANTUM_PARAMETER_ORDER = [
    "Concurrence",
    "Ckk + Cnn",
    "Ckk - Cnn",
    "Ckk + Crr",
    "Ckk - Crr",
    "Cnn + Crr",
    "Cnn - Crr",
]
BC_PARAMETER_ORDER = [
    "B_An",
    "B_Bn",
    "B_Ar",
    "B_Br",
    "B_Ak",
    "B_Bk",
    "Cnn",
    "C_nr",
    "C_nk",
    "C_rn",
    "Crr",
    "C_rk",
    "C_kn",
    "C_kr",
    "Ckk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse central QIProcessor results.txt files, draw per-channel method comparisons, "
            "and append statistically combined all-channel results for each method."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="LABEL{=,:}PATH",
        help=(
            "Method spec LABEL=PATH or LABEL:PATH for a results.txt file or directory. "
            "If a directory is supplied, the script searches common results.txt locations. "
            "Repeat for Baseline, EveNet-Pretrain, Scratch, etc."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        "--output-dir",
        dest="output_prefix",
        type=Path,
        required=True,
        help=(
            "Output prefix. Writes <prefix>_per_channel.*, <prefix>_combined.*, and summary plots. "
            "--output-dir is accepted as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--keep-truth",
        action="store_true",
        help="Also keep Truth sections. By default only Unfolded/Final/Nominal measurements are written.",
    )
    parser.add_argument(
        "--keep-hadhad",
        action="store_true",
        help="Keep broad hadhad rows. By default they are ignored in favor of fine channels.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write JSON/CSV tables; skip plots.",
    )
    parser.add_argument(
        "--no-tension-scale",
        action="store_true",
        help=(
            "Do not inflate combined uncertainties when channels disagree more than expected. "
            "By default a PDG/Birge scale factor is applied when chi2/ndof > 1."
        ),
    )
    parser.add_argument(
        "--combination-model",
        choices=["split-normal", "symmetric"],
        default="split-normal",
        help=(
            "Uncertainty model for cross-channel combination. split-normal uses err_down on the left "
            "side and err_up on the right side of each measurement. symmetric uses the average error."
        ),
    )
    parser.add_argument(
        "--include-groups",
        nargs="+",
        choices=["BC", "quantum"],
        default=["quantum", "BC"],
        help="Which result groups to plot/write after parsing. Default: quantum BC.",
    )
    parser.add_argument(
        "--only-key-quantum",
        action="store_true",
        help="For quantum plots/tables, keep only Concurrence and Ckk + Cnn.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-method and per-parameter parsing/combination diagnostics.",
    )
    return parser.parse_args()


def parse_method_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    methods: list[tuple[str, Path]] = []
    for spec in specs:
        split_positions = [index for index in (spec.find("="), spec.find(":")) if index > 0]
        if not split_positions:
            raise ValueError(
                f"Invalid --method '{spec}'. Expected LABEL=PATH or LABEL:PATH."
            )
        split_at = min(split_positions)
        label, raw_path = spec[:split_at], spec[split_at + 1:]
        label = label.strip()
        if not label or not raw_path.strip():
            raise ValueError(f"Invalid --method '{spec}'. Label and path are required.")
        methods.append((label, resolve_results_path(Path(raw_path))))
    return methods


def resolve_results_path(path: Path) -> Path:
    """
    Accept either a direct results.txt path or a method directory.

    This matters for your use case because baseline may live at:
      baseline/QI_analysis/results.txt
    while EveNet exports may live inside a qi-export directory.
    """
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    candidates = [
        path / "QI_analysis" / "results.txt",
        path / "results.txt",
        path / "central" / "QI_analysis" / "results.txt",
        path / "central" / "results.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(path.rglob("results.txt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No results.txt found under directory: {path}")

    pretty = "\n".join(f"  - {match}" for match in matches[:20])
    extra = "" if len(matches) <= 20 else f"\n  ... and {len(matches) - 20} more"
    raise ValueError(
        f"Multiple results.txt files found under {path}. Please pass the exact one:\n{pretty}{extra}"
    )


def parse_results_text(
    text: str,
    method: str,
    keep_truth: bool = False,
    ignored_regions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ignored_regions = ignored_regions or set()
    region: str | None = None
    signal: str | None = None
    section_source: str | None = None
    section_group: str | None = None
    combined_fit = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Region: "):
            region = line.removeprefix("Region: ").strip()
            signal = region
            section_source = "Unfolded"
            section_group = None
            combined_fit = False
            continue

        if line.startswith("Combined fit regions:"):
            region = COMBINED_REGION
            signal = COMBINED_SIGNAL
            section_source = "Unfolded"
            section_group = None
            combined_fit = True
            continue

        header_match = UNFOLDING_HEADER_RE.match(line)
        if header_match:
            signal = header_match.group("signal").strip()
            region = header_match.group("region").strip()
            section_source = None
            section_group = None
            combined_fit = False
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            section_source = canonical_source_name(section_match.group("source") or "Unfolded")
            section_group = canonical_group_name(section_match.group("group"))
            continue

        parsed_value = parse_measurement_line(raw_line)
        if parsed_value is None:
            continue
        if region is None or signal is None or section_source is None:
            continue
        parameter_name = canonical_parameter_name(str(parsed_value["name"]))
        if section_group is None or combined_fit:
            if parameter_name in BC_PARAMETER_ORDER:
                row_group = "BC"
            elif parameter_name in QUANTUM_PARAMETER_ORDER:
                row_group = "quantum"
            else:
                continue
        else:
            row_group = section_group
        if section_source == "Truth" and not keep_truth:
            continue
        if region in ignored_regions:
            continue

        rows.append(
            {
                "method": method,
                "region": region,
                "signal": signal,
                "channel": canonical_channel_key(region, signal),
                "source": section_source,
                "group": row_group,
                "parameter": parameter_name,
                "value": float(parsed_value["value"]),
                "err_up": float(parsed_value["err_up"]),
                "err_down": float(parsed_value["err_down"]),
                "is_combined": combined_fit,
            }
        )

    return rows


def canonical_source_name(source: str) -> str:
    normalized = source.strip().lower()
    if normalized in {"final", "nominal", "unfolded"}:
        return "Unfolded"
    if normalized == "truth":
        return "Truth"
    return "Unfolded"


def canonical_group_name(group: str) -> str:
    normalized = group.strip().lower()
    if "b and c" in normalized or normalized.startswith("bc"):
        return "BC"
    return "quantum"


def canonical_parameter_name(parameter: str) -> str:
    cleaned = " ".join(parameter.strip().replace("_", "_").split())
    compact = cleaned.replace("C_kk", "Ckk").replace("C_nn", "Cnn").replace("C_rr", "Crr")
    compact = compact.replace("Ckk+Cnn", "Ckk + Cnn")
    compact = compact.replace("Ckk-Cnn", "Ckk - Cnn")
    compact = compact.replace("Ckk+Crr", "Ckk + Crr")
    compact = compact.replace("Ckk-Crr", "Ckk - Crr")
    compact = compact.replace("Cnn+Crr", "Cnn + Crr")
    compact = compact.replace("Cnn-Crr", "Cnn - Crr")
    return compact


def parse_measurement_line(raw_line: str) -> dict[str, float | str] | None:
    match = RESULT_LINE_ASYMMETRIC_RE.match(raw_line)
    if match:
        return {
            "name": match.group("name").strip(),
            "value": float(match.group("value")),
            "err_up": abs(float(match.group("err_up"))),
            "err_down": abs(float(match.group("err_down"))),
        }

    match = RESULT_LINE_ASYMMETRIC_PLAIN_RE.match(raw_line)
    if match:
        return {
            "name": match.group("name").strip(),
            "value": float(match.group("value")),
            "err_up": abs(float(match.group("err_up"))),
            "err_down": abs(float(match.group("err_down"))),
        }

    match = RESULT_LINE_SYMMETRIC_RE.match(raw_line)
    if match:
        err = abs(float(match.group("err")))
        return {
            "name": match.group("name").strip(),
            "value": float(match.group("value")),
            "err_up": err,
            "err_down": err,
        }
    return None


def canonical_channel_key(region: str, signal: str) -> str:
    """
    Put baseline bare channels and EveNet Ztautau_* channels onto common rows.

    Examples:
      region=pipi, signal=pipi              -> Ztautau_pipi
      region=pipi, signal=Ztautau_pipi      -> Ztautau_pipi
      region=emu,  signal=Ztautau_mue       -> Ztautau_mue
      region=Ztautau_pie, signal=pie        -> Ztautau_pie
    """
    for raw in (signal, region):
        channel = normalize_channel_name(raw)
        if channel is not None:
            return channel
    return normalize_freeform_channel_name(signal or region)


def normalize_channel_name(name: str) -> str | None:
    raw = name.strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered.startswith("ztautau_"):
        lowered = lowered.removeprefix("ztautau_")
    if lowered in CANONICAL_SIGNAL_CHANNELS:
        return f"Ztautau_{lowered}"
    return None


def normalize_freeform_channel_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        return raw
    if raw.lower().startswith("ztautau_"):
        suffix = raw[len("Ztautau_"):]
        if suffix:
            return f"Ztautau_{suffix}"
    return raw


def filter_rows(rows: list[dict[str, Any]], include_groups: list[str], only_key_quantum: bool) -> list[dict[str, Any]]:
    output = [row for row in rows if row["group"] in set(include_groups)]
    if only_key_quantum:
        output = [
            row
            for row in output
            if row["group"] != "quantum" or row["parameter"] in {"Concurrence", "Ckk + Cnn"}
        ]
    return output


def combine_all_channels(
    rows: list[dict[str, Any]],
    combination_model: str = "split-normal",
    apply_tension_scale: bool = True,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Combine channels inside each method/source/group/parameter only.

    This never combines Baseline with EveNet. Each method gets its own combined row.

    Uncertainty treatment:
      - split-normal: use err_down when theta < measured value and err_up when theta > measured value.
      - symmetric: use 0.5 * (err_up + err_down).
      - The combined central value is the minimum of the summed independent-channel NLL.
      - The raw 1-sigma interval is found from delta NLL = 0.5.
      - By default, if channels are more scattered than expected, apply a PDG/Birge scale factor:
            scale = sqrt(chi2_min / ndof), if chi2_min / ndof > 1.
        The raw and scaled uncertainties are both written to the output.

    Limitation:
      If you have a full covariance matrix across channels, use that instead. This script assumes
      independent channel measurements because results.txt contains only per-channel up/down errors.
    """
    combined_rows: list[dict[str, Any]] = []
    keys = list(
        dict.fromkeys(
            (row["method"], row["source"], row["group"], row["parameter"])
            for row in rows
            if not row.get("is_combined", False)
        )
    )

    for method, source, group, parameter in keys:
        channel_rows = [
            row
            for row in rows
            if not row.get("is_combined", False)
            and row["method"] == method
            and row["source"] == source
            and row["group"] == group
            and row["parameter"] == parameter
        ]
        valid_rows = [
            row
            for row in channel_rows
            if is_valid_measurement(row["value"], row["err_down"], row["err_up"])
        ]
        if not valid_rows:
            continue

        values = np.array([row["value"] for row in valid_rows], dtype=np.float64)
        err_down = np.array([row["err_down"] for row in valid_rows], dtype=np.float64)
        err_up = np.array([row["err_up"] for row in valid_rows], dtype=np.float64)

        if combination_model == "symmetric":
            result = combine_symmetric(values, err_down, err_up)
        else:
            result = combine_split_normal(values, err_down, err_up)

        chi2_min = float(2.0 * result["nll_min"])
        ndof = max(int(len(values) - 1), 0)
        tension_scale = 1.0
        if apply_tension_scale and ndof > 0:
            reduced_chi2 = chi2_min / ndof
            if np.isfinite(reduced_chi2) and reduced_chi2 > 1.0:
                tension_scale = float(math.sqrt(reduced_chi2))

        err_down_raw = float(result["err_down"])
        err_up_raw = float(result["err_up"])
        err_down_scaled = err_down_raw * tension_scale
        err_up_scaled = err_up_raw * tension_scale

        combined_rows.append(
            {
                "method": method,
                "region": COMBINED_REGION,
                "signal": COMBINED_SIGNAL,
                "channel": COMBINED_CHANNEL_KEY,
                "source": source,
                "group": group,
                "parameter": parameter,
                "value": float(result["value"]),
                "err_up": err_up_scaled,
                "err_down": err_down_scaled,
                "err_up_raw": err_up_raw,
                "err_down_raw": err_down_raw,
                "tension_scale": tension_scale,
                "chi2_min": chi2_min,
                "ndof": ndof,
                "num_channels": int(len(values)),
                "channels": ",".join(row["channel"] for row in valid_rows),
                "combination_model": combination_model,
                "is_combined": True,
            }
        )

        if debug:
            print(
                "[qi-combine] "
                f"method={method} group={group} parameter={parameter} "
                f"n={len(values)} value={result['value']:.6g} "
                f"raw=-{err_down_raw:.6g}/+{err_up_raw:.6g} "
                f"scale={tension_scale:.4g} chi2={chi2_min:.4g} ndof={ndof}",
                flush=True,
            )

    return combined_rows


def is_valid_measurement(value: Any, err_down: Any, err_up: Any) -> bool:
    try:
        v = float(value)
        lo = float(err_down)
        hi = float(err_up)
    except Exception:
        return False
    return np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0


def combine_symmetric(values: np.ndarray, err_down: np.ndarray, err_up: np.ndarray) -> dict[str, float]:
    sigmas = 0.5 * (err_down + err_up)
    weights = 1.0 / np.square(sigmas)
    value = float(np.sum(weights * values) / np.sum(weights))
    sigma = float(math.sqrt(1.0 / np.sum(weights)))
    nll_min = float(0.5 * np.sum(np.square((value - values) / sigmas)))
    return {"value": value, "err_down": sigma, "err_up": sigma, "nll_min": nll_min}


def combine_split_normal(values: np.ndarray, err_down: np.ndarray, err_up: np.ndarray) -> dict[str, float]:
    def nll(theta: float) -> float:
        sigma = np.where(theta < values, err_down, err_up)
        return float(0.5 * np.sum(np.square((theta - values) / sigma)))

    lower, upper = robust_search_bounds(values, err_down, err_up)
    theta_hat = minimize_1d(nll, lower, upper)
    nll_min = nll(theta_hat)

    target = nll_min + 0.5
    left = find_delta_nll_crossing(nll, target, theta_hat, lower, side="left")
    right = find_delta_nll_crossing(nll, target, theta_hat, upper, side="right")

    return {
        "value": float(theta_hat),
        "err_down": float(theta_hat - left),
        "err_up": float(right - theta_hat),
        "nll_min": float(nll_min),
    }


def robust_search_bounds(values: np.ndarray, err_down: np.ndarray, err_up: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin(values - 8.0 * err_down))
    hi = float(np.nanmax(values + 8.0 * err_up))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmean(values))
        width = float(np.nanmax(np.maximum(err_down, err_up)))
        width = width if np.isfinite(width) and width > 0 else 1.0
        lo = center - 8.0 * width
        hi = center + 8.0 * width
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def minimize_1d(func: Any, lower: float, upper: float, iterations: int = 180) -> float:
    """Golden-section minimization with no scipy dependency."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lower), float(upper)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)
    for _ in range(iterations):
        if fc > fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
    return 0.5 * (a + b)


def find_delta_nll_crossing(func: Any, target: float, theta_hat: float, bound: float, side: str) -> float:
    """Find theta where NLL(theta) = target on the requested side of theta_hat."""
    assert side in {"left", "right"}
    if side == "left":
        low, high = bound, theta_hat
        if func(low) < target:
            step = theta_hat - bound
            for _ in range(12):
                low = theta_hat - 2.0 * step
                step *= 2.0
                if func(low) >= target:
                    break
    else:
        low, high = theta_hat, bound
        if func(high) < target:
            step = bound - theta_hat
            for _ in range(12):
                high = theta_hat + 2.0 * step
                step *= 2.0
                if func(high) >= target:
                    break

    for _ in range(180):
        mid = 0.5 * (low + high)
        if side == "left":
            if func(mid) >= target:
                low = mid
            else:
                high = mid
        else:
            if func(mid) >= target:
                high = mid
            else:
                low = mid
    return 0.5 * (low + high)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return

    preferred = [
        "method",
        "region",
        "signal",
        "channel",
        "source",
        "group",
        "parameter",
        "value",
        "err_up",
        "err_down",
        "is_combined",
        "num_channels",
        "channels",
        "err_up_raw",
        "err_down_raw",
        "tension_scale",
        "chi2_min",
        "ndof",
        "combination_model",
    ]
    keys = list(dict.fromkeys(preferred + [key for row in rows for key in row.keys()]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_order = {method: i for i, method in enumerate(dict.fromkeys(row["method"] for row in rows))}
    group_order = {"quantum": 0, "BC": 1}
    return sorted(
        rows,
        key=lambda row: (
            group_order.get(row["group"], 99),
            parameter_sort_key(row["group"], row["parameter"]),
            channel_sort_key(row["channel"]),
            method_order.get(row["method"], 99),
            row["source"],
        ),
    )


def parameter_sort_key(group: str, parameter: str) -> tuple[int, str]:
    order = QUANTUM_PARAMETER_ORDER if group == "quantum" else BC_PARAMETER_ORDER
    try:
        return (order.index(parameter), parameter)
    except ValueError:
        return (len(order), parameter)


def channel_sort_key(channel: str) -> tuple[int, str]:
    try:
        return (CHANNEL_ORDER.index(channel), channel)
    except ValueError:
        return (len(CHANNEL_ORDER) - 1, channel)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("_") or "measurement"


def method_marker(method: str, method_index: int) -> str:
    return METHOD_MARKERS[method_index % len(METHOD_MARKERS)]


def source_display_name(source: str) -> str:
    if source == "Unfolded":
        return ""
    if source == "Truth":
        return "Truth"
    return source


def series_label(method: str, source: str) -> str:
    source_label = source_display_name(source)
    if not source_label:
        return method
    return f"{method} ({source_label})"


def channel_label(channel: str) -> str:
    if channel == COMBINED_CHANNEL_KEY:
        return r"$\bf{Combined}$"
    return channel_latex_label(channel)


def parameter_label(parameter: str) -> str:
    if parameter.startswith("C_") and len(parameter) == 4:
        return rf"$C_{{{parameter[-2]}{parameter[-1]}}}$"
    if parameter.startswith("B_") and len(parameter) == 4:
        return rf"$B_{{{parameter[-2]},{parameter[-1]}}}$"
    if parameter == "Concurrence":
        return "Concurrence"
    match = re.match(r"C_?(?P<a>[a-z]{2}) (?P<op>[+-]) C_?(?P<b>[a-z]{2})$", parameter)
    if match:
        return rf"$C_{{{match.group('a')}}} {match.group('op')} C_{{{match.group('b')}}}$"
    return parameter.replace("_", " ")


def format_value_unc(row: dict[str, Any]) -> str:
    value = float(row["value"])
    err_up = float(row["err_up"])
    err_down = float(row["err_down"])
    if math.isclose(err_up, err_down, rel_tol=0.05, abs_tol=5.0e-4):
        return f"{value:.3f} ± {0.5 * (err_up + err_down):.3f}"
    return f"{value:.3f} +{err_up:.3f}/-{err_down:.3f}"


def sensitivity_value(row: dict[str, Any]) -> float:
    value = float(row["value"])
    err_up = float(row["err_up"])
    err_down = float(row["err_down"])
    if value > 0.0:
        uncertainty = err_down
    elif value < 0.0:
        uncertainty = err_up
    else:
        uncertainty = 0.5 * (err_up + err_down)
    if not np.isfinite(value) or not np.isfinite(uncertainty) or uncertainty <= 0.0:
        return float("nan")
    return float(value / uncertainty)


def format_value_sigma(row: dict[str, Any]) -> str:
    value = float(row["value"])
    sensitivity = sensitivity_value(row)
    if not np.isfinite(sensitivity):
        return f"{value:.3f}"
    return rf"{value:.3f} (${sensitivity:.2f}\sigma$)"


def focused_x_window(
    values: np.ndarray,
    err_down: np.ndarray,
    err_up: np.ndarray,
    coverage: float = 0.90,
) -> tuple[float, float]:
    finite = np.isfinite(values) & np.isfinite(err_down) & np.isfinite(err_up)
    if not np.any(finite):
        return (-1.0, 1.0)

    lower_bounds = values[finite] - err_down[finite]
    upper_bounds = values[finite] + err_up[finite]
    tail = 0.5 * (1.0 - coverage)
    q_low = 100.0 * tail
    q_high = 100.0 * (1.0 - tail)
    xmin = float(np.nanpercentile(lower_bounds, q_low))
    xmax = float(np.nanpercentile(upper_bounds, q_high))

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        xmin = float(np.nanmin(lower_bounds))
        xmax = float(np.nanmax(upper_bounds))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        center = float(np.nanmean(values[finite]))
        width = float(np.nanmean(np.maximum(err_down[finite], err_up[finite])))
        width = width if np.isfinite(width) and width > 0.0 else 1.0
        xmin = center - width
        xmax = center + width
    return xmin, xmax


def plot_measurement_summaries(rows: list[dict[str, Any]], output_prefix: Path) -> dict[str, Any]:
    plot_dir = output_prefix.parent / f"{output_prefix.name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    method_index = {method: index for index, method in enumerate(methods)}
    series = list(dict.fromkeys((row["method"], row["source"]) for row in rows))
    series_index = {series_key: index for index, series_key in enumerate(series)}
    groups = list(dict.fromkeys(row["group"] for row in sorted_rows(rows)))
    plot_summary: dict[str, Any] = {}

    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        parameters = list(dict.fromkeys(row["parameter"] for row in sorted_rows(group_rows)))
        for parameter in parameters:
            parameter_rows = [row for row in group_rows if row["parameter"] == parameter]
            channel_keys = list(dict.fromkeys(row["channel"] for row in sorted_rows(parameter_rows)))
            if COMBINED_CHANNEL_KEY in channel_keys:
                channel_keys = [key for key in channel_keys if key != COMBINED_CHANNEL_KEY] + [COMBINED_CHANNEL_KEY]

            row_spacing = 1.35
            y_base = np.arange(len(channel_keys), dtype=np.float64) * row_spacing
            channel_index = {key: index for index, key in enumerate(channel_keys)}

            values = np.array([row["value"] for row in parameter_rows], dtype=np.float64)
            err_up = np.array([row["err_up"] for row in parameter_rows], dtype=np.float64)
            err_down = np.array([row["err_down"] for row in parameter_rows], dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(err_up) & np.isfinite(err_down)
            if not np.any(finite):
                continue
            xmin, xmax = focused_x_window(values, err_down, err_up, coverage=0.90)
            span = xmax - xmin
            pad = max(0.12 * span, 0.02)

            fig_height = max(5.2, 0.92 * len(channel_keys) + 2.8)
            fig, ax = plt.subplots(figsize=(11.6, fig_height), dpi=200)
            ax.set_xlim(xmin - pad, xmax + pad)

            x_text_value = 1.025
            for key in channel_keys:
                channel_rows = [row for row in parameter_rows if row["channel"] == key]
                channel_rows.sort(key=lambda row: series_index[(row["method"], row["source"])])
                offsets = (
                    np.linspace(-0.42, 0.42, len(channel_rows))
                    if len(channel_rows) > 1
                    else np.array([0.0])
                )
                for offset, row in zip(offsets, channel_rows):
                    method_i = method_index[row["method"]]
                    y = y_base[channel_index[key]] + offset
                    color = method_color(row["method"], method_i)
                    marker = method_marker(row["method"], method_i)
                    is_truth = row["source"] == "Truth"
                    markersize = 8.5 if row.get("is_combined", False) else 6.5
                    linewidth = 1.6 if row.get("is_combined", False) else 1.2
                    ax.errorbar(
                        row["value"],
                        y,
                        xerr=np.array([[row["err_down"]], [row["err_up"]]], dtype=np.float64),
                        fmt=marker,
                        color=color,
                        markerfacecolor="white" if is_truth else color,
                        markeredgecolor=color,
                        capsize=2.8,
                        markersize=markersize,
                        lw=linewidth,
                    )
                    label = format_value_unc(row)
                    if row.get("is_combined", False):
                        scale = float(row.get("tension_scale", 1.0))
                        nchan = int(row.get("num_channels", 0))
                        if scale > 1.0001:
                            label += f"  [n={nchan}, scale={scale:.2f}]"
                        else:
                            label += f"  [n={nchan}]"
                    lower = float(row["value"] - row["err_down"])
                    upper = float(row["value"] + row["err_up"])
                    if lower < xmin or upper > xmax:
                        label += "  w. overflow"
                    ax.text(
                        x_text_value,
                        y,
                        label,
                        color=color,
                        fontsize=7.6,
                        va="center",
                        ha="left",
                        transform=ax.get_yaxis_transform(),
                        clip_on=False,
                    )

            ax.axvline(0.0, color="#B0B0B0", linewidth=0.8, linestyle="--", zorder=0)
            ax.text(x_text_value, 1.02, "value ± unc.", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
            ax.set_yticks(y_base)
            ax.set_yticklabels([channel_label(channel) for channel in channel_keys])
            ax.invert_yaxis()
            ax.grid(axis="y", alpha=0.18, linestyle=":")
            for separator_index in range(len(channel_keys) - 1):
                separator = y_base[separator_index] + 0.5 * row_spacing
                if channel_keys[separator_index + 1] == COMBINED_CHANNEL_KEY:
                    ax.axhline(separator, color="#808080", linewidth=1.4, zorder=0)
                else:
                    ax.axhline(separator, color="#D9D9D9", linewidth=0.8, zorder=0)
            ax.set_xlabel(parameter_label(parameter))
            ax.set_ylabel("Channel / combined")
            ax.set_title(f"{parameter_label(parameter)}: per-channel method comparison + combined")

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=method_marker(method, method_index[method]),
                    color=method_color(method, method_index[method]),
                    markerfacecolor="white" if source == "Truth" else method_color(method, method_index[method]),
                    markeredgecolor=method_color(method, method_index[method]),
                    markersize=7,
                    linestyle="None",
                    label=series_label(method, source),
                )
                for method, source in series
            ]
            ax.legend(
                handles=handles,
                title="Method / source",
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=min(len(handles), 4),
            )
            fig.subplots_adjust(right=0.74, top=0.80, left=0.16, bottom=0.16)

            plot_path = plot_dir / f"{sanitize_filename(group)}_{sanitize_filename(parameter)}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            plot_summary[f"{group}:{parameter}"] = {
                "plot": str(plot_path),
                "plot_type": "value_comparison",
                "num_points": len(parameter_rows),
                "methods": methods,
                "series": [series_label(method, source) for method, source in series],
                "channels": channel_keys,
            }
            print(f"[qi-final] wrote_plot={plot_path}", flush=True)

            sensitivity_rows = [row for row in parameter_rows if np.isfinite(sensitivity_value(row))]
            if sensitivity_rows:
                sensitivity_values = np.array([sensitivity_value(row) for row in sensitivity_rows], dtype=np.float64)
                sens_min = 0.0
                sens_max = float(np.nanmax(sensitivity_values))
                sens_span = sens_max - sens_min
                sens_pad = max(0.18 * max(sens_span, 1.0), 0.6)

                sens_fig_height = max(5.2, 0.92 * len(channel_keys) + 2.8)
                sens_fig, sens_ax = plt.subplots(figsize=(11.6, sens_fig_height), dpi=200)
                sens_ax.set_xlim(0.0, sens_max + sens_pad)

                max_rows_per_channel = max(
                    len([row for row in parameter_rows if row["channel"] == key])
                    for key in channel_keys
                )
                bar_height = min(0.22, 0.82 / max(max_rows_per_channel, 1))

                for key in channel_keys:
                    channel_rows = [row for row in parameter_rows if row["channel"] == key]
                    channel_rows.sort(key=lambda row: series_index[(row["method"], row["source"])])
                    offsets = (
                        np.linspace(-0.42, 0.42, len(channel_rows))
                        if len(channel_rows) > 1
                        else np.array([0.0])
                    )
                    for offset, row in zip(offsets, channel_rows):
                        sensitivity = sensitivity_value(row)
                        if not np.isfinite(sensitivity):
                            continue
                        method_i = method_index[row["method"]]
                        y = y_base[channel_index[key]] + offset
                        color = method_color(row["method"], method_i)
                        is_truth = row["source"] == "Truth"
                        sens_ax.barh(
                            y,
                            sensitivity,
                            height=bar_height,
                            color="white" if is_truth else color,
                            edgecolor=color,
                            linewidth=1.3,
                            zorder=2,
                        )
                        x_text = (
                            sensitivity + 0.03 * (sens_max - sens_min + 2.0)
                            if sensitivity >= 0.0
                            else sensitivity - 0.03 * (sens_max - sens_min + 2.0)
                        )
                        sens_ax.text(
                            x_text,
                            y,
                            format_value_sigma(row),
                            color=color,
                            fontsize=7.6,
                            va="center",
                            ha="left" if sensitivity >= 0.0 else "right",
                        )

                sens_ax.axvline(0.0, color="#808080", linewidth=1.0, linestyle="--", zorder=1)
                sens_ax.set_yticks(y_base)
                sens_ax.set_yticklabels([channel_label(channel) for channel in channel_keys])
                sens_ax.invert_yaxis()
                sens_ax.grid(axis="y", alpha=0.18, linestyle=":")
                for separator_index in range(len(channel_keys) - 1):
                    separator = y_base[separator_index] + 0.5 * row_spacing
                    if channel_keys[separator_index + 1] == COMBINED_CHANNEL_KEY:
                        sens_ax.axhline(separator, color="#808080", linewidth=1.4, zorder=0)
                    else:
                        sens_ax.axhline(separator, color="#D9D9D9", linewidth=0.8, zorder=0)
                sens_ax.set_xlabel("Sensitivity")
                sens_ax.set_ylabel("Channel / combined")
                sens_ax.set_title(f"{parameter_label(parameter)}: value / selected uncertainty")

                sens_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_color(method, method_index[method]),
                        marker="s",
                        markerfacecolor="white" if source == "Truth" else method_color(method, method_index[method]),
                        markeredgecolor=method_color(method, method_index[method]),
                        markersize=7,
                        linestyle="None",
                        label=series_label(method, source),
                    )
                    for method, source in series
                ]
                sens_ax.legend(
                    handles=sens_handles,
                    title="Method / source",
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=min(len(sens_handles), 4),
                )
                sens_fig.subplots_adjust(right=0.96, top=0.80, left=0.16, bottom=0.16)

                sensitivity_plot_path = plot_dir / f"{sanitize_filename(group)}_{sanitize_filename(parameter)}_sensitivity.png"
                sens_fig.savefig(sensitivity_plot_path)
                plt.close(sens_fig)
                plot_summary[f"{group}:{parameter}:sensitivity"] = {
                    "plot": str(sensitivity_plot_path),
                    "plot_type": "sensitivity",
                    "num_points": len(sensitivity_rows),
                    "methods": methods,
                    "series": [series_label(method, source) for method, source in series],
                    "channels": channel_keys,
                }
                print(f"[qi-final] wrote_plot={sensitivity_plot_path}", flush=True)

    return plot_summary


def main() -> None:
    args = parse_args()
    ignored_regions = set() if args.keep_hadhad else DEFAULT_IGNORED_REGIONS

    parsed_rows: list[dict[str, Any]] = []
    for method, results_path in parse_method_specs(args.method):
        rows = parse_results_text(
            results_path.read_text(),
            method=method,
            keep_truth=args.keep_truth,
            ignored_regions=ignored_regions,
        )
        rows = filter_rows(rows, args.include_groups, args.only_key_quantum)
        parsed_rows.extend(rows)
        print(f"[qi-final] method={method} rows={len(rows)} path={results_path}", flush=True)

    if not parsed_rows:
        raise ValueError(
            "No final measurements found. Expected QIProcessor Unfolded/Final B/C or Quantum "
            "sections, or a ForwardFoldingProcessor 'Combined fit regions:' result."
        )

    per_channel_rows = [row for row in parsed_rows if not row.get("is_combined", False)]
    combined_rows = [row for row in parsed_rows if row.get("is_combined", False)]
    combined_rows.extend(combine_all_channels(
        parsed_rows,
        combination_model=args.combination_model,
        apply_tension_scale=not args.no_tension_scale,
        debug=args.debug,
    ))
    all_rows = sorted_rows(per_channel_rows + combined_rows)
    combined_rows = sorted_rows(combined_rows)
    per_channel_rows = sorted_rows(per_channel_rows)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_channel_json = args.output_prefix.parent / f"{args.output_prefix.name}_per_channel.json"
    per_channel_csv = args.output_prefix.parent / f"{args.output_prefix.name}_per_channel.csv"
    combined_json = args.output_prefix.parent / f"{args.output_prefix.name}_combined.json"
    combined_csv = args.output_prefix.parent / f"{args.output_prefix.name}_combined.csv"
    all_json = args.output_prefix.parent / f"{args.output_prefix.name}_per_channel_plus_combined.json"
    all_csv = args.output_prefix.parent / f"{args.output_prefix.name}_per_channel_plus_combined.csv"

    per_channel_json.write_text(json.dumps(per_channel_rows, indent=2, sort_keys=True) + "\n")
    combined_json.write_text(json.dumps(combined_rows, indent=2, sort_keys=True) + "\n")
    all_json.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n")
    write_csv(per_channel_rows, per_channel_csv)
    write_csv(combined_rows, combined_csv)
    write_csv(all_rows, all_csv)

    if not args.no_plots:
        plot_summary = plot_measurement_summaries(all_rows, args.output_prefix)
        plot_json_path = args.output_prefix.parent / f"{args.output_prefix.name}_plots.json"
        plot_json_path.write_text(json.dumps(plot_summary, indent=2, sort_keys=True) + "\n")
        print(f"[qi-final] wrote_plot_summary={plot_json_path}", flush=True)

    print(f"[qi-final] per_channel_rows={len(per_channel_rows)}")
    print(f"[qi-final] combined_rows={len(combined_rows)}")
    print(f"[qi-final] wrote_json={all_json}")
    print(f"[qi-final] wrote_csv={all_csv}")
    print(f"[qi-final] wrote_combined_csv={combined_csv}")


if __name__ == "__main__":
    main()
