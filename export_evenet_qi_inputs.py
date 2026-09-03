#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import glob
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector
import yaml

ML_PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_PIPELINE_DIR.parent
LEP_TREE_ANA_ROOT = Path(os.environ.get("LEP_TREE_ANA_ROOT", REPO_ROOT)).expanduser().resolve()
if not (LEP_TREE_ANA_ROOT / "quantum" / "observables_builder.py").is_file():
    sibling = REPO_ROOT / "lep_tree_ana"
    if (sibling / "quantum" / "observables_builder.py").is_file():
        LEP_TREE_ANA_ROOT = sibling
if str(LEP_TREE_ANA_ROOT) not in sys.path:
    sys.path.insert(0, str(LEP_TREE_ANA_ROOT))
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

from evaluation_config import post_calibration_enabled, qi_region_to_signals
from common import (
    build_classification_lookup,
    event_preselection_mask,
    post_calibrate_tau_tau,
    read_yaml,
    rebuild_vector,
    to_numpy,
)
from quantum.observables_builder import build_observables, get_observable_names
from utils.ztautau_spin import (
    CM_ENERGY_GEV,
    build_tau_pair_from_direction_offsets,
)

vector.register_awkward()

METHOD_CHOICES = ("target", "baseline", "evenet", "truth")
DEFAULT_METHODS = ("target", "evenet", "baseline", "truth")
SAMPLE_ORDER = ("data94", "Zqq", "Zll", "Ztautau")
BASE_EVENT_FIELDS = (
    "event_category",
    "truth_QI_region",
    "analyzing_power_a",
    "analyzing_power_b",
    "analyzing_power",
    "initial_total_num_events",
    "nprong",
)
FINAL_SCHEMA_FIELDS = (
    ("event_category", np.int32, 0),
    ("truth_QI_region", bool, False),
    ("analyzing_power_a", np.float32, np.nan),
    ("analyzing_power_b", np.float32, np.nan),
    ("analyzing_power", np.float32, np.nan),
    ("initial_total_num_events", np.float64, 0.0),
    ("nprong", np.int32, 0),
    ("baseline_cut", bool, False),
    ("predict_weight", np.float32, 0.0),
    ("weight", np.float32, 0.0),
    ("weight_nominal", np.float32, 0.0),
    ("calibration_deltaR_a", np.float32, np.nan),
    ("calibration_deltaR_b", np.float32, np.nan),
    ("calibration_deltaR_sum", np.float32, np.nan),
    ("is_leading_OS", bool, False),
    ("charged_E", np.float32, np.nan),
    ("P_rad", np.float32, np.nan),
    ("lead_a_pdgId", np.int32, 0),
    ("lead_b_pdgId", np.int32, 0),
    ("lead_a_hpcTotalShowerEnergy", np.float32, np.nan),
    ("lead_b_hpcTotalShowerEnergy", np.float32, np.nan),
    ("lead_a_E_over_p", np.float32, np.nan),
    ("lead_b_E_over_p", np.float32, np.nan),
    ("lead_a_raw_muon_tag", np.int32, 0),
    ("lead_b_raw_muon_tag", np.int32, 0),
    ("lead_a_is_electron", bool, False),
    ("lead_b_is_electron", bool, False),
    ("lead_a_is_muon", bool, False),
    ("lead_b_is_muon", bool, False),
    ("Event_totalChargedEnergy", np.float32, np.nan),
    ("Event_totalEMEnergy", np.float32, np.nan),
    ("Event_totalHadronicEnergy", np.float32, np.nan),
    ("lead_a_raw_muon_hits", np.int32, 0),
    ("lead_b_raw_muon_hits", np.int32, 0),
    ("lead_a_hpc_E", np.float32, np.nan),
    ("lead_b_hpc_E", np.float32, np.nan),
    ("lead_a_elid", np.int32, 0),
    ("lead_b_elid", np.int32, 0),
    ("lead_a_raw_wires", np.int32, 0),
    ("lead_b_raw_wires", np.int32, 0),
)


@dataclass(frozen=True)
class RawWeightInfo:
    is_data: bool
    weight_scale: float
    total_initial_num_events: float | None
    weight_source: str


def export_observable_names() -> tuple[str, ...]:
    return tuple(get_observable_names())


def read_file_initial_total_num_events(path: str) -> float | None:
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=1, columns=["initial_total_num_events"]):
        values = ak.to_numpy(ak.from_arrow(record_batch)["initial_total_num_events"], allow_missing=False)
        if len(values):
            return float(values[0])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EveNet prediction parquets as nominal QI unfolding inputs."
    )
    parser.add_argument("--analysis-config", type=Path, default=Path("ml_pipeline/config/analysis.yaml"))
    parser.add_argument("--prediction-parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=METHOD_CHOICES)
    parser.add_argument("--regions", nargs="+", default=None, help="Defaults to Ztautau labels listed in NeutrinoPrediction.")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--mc-split-fraction",
        type=float,
        default=None,
        help=(
            "Optional MC prediction split fraction. If prediction parquets were made from a split "
            "without applying the 1/fraction weight correction, MC prediction rows are scaled by "
            "1 / mc_split_fraction during export. Data and RAW-complement rows are not scaled."
        ),
    )
    parser.add_argument(
        "--pseudo-data",
        action="store_true",
        help="Deprecated and no longer supported. Use real data inputs instead.",
    )
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def resolve_parquets(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        text = str(path.expanduser())
        matches = sorted(glob.glob(text))
        candidates = [Path(match) for match in matches] if matches else [Path(text)]
        for candidate in candidates:
            if candidate.is_dir():
                merged = sorted(candidate.glob("*__evenet_pred.parquet"))
                output.extend(merged or sorted(candidate.glob("*__evenet_pred.part*.parquet")))
            else:
                output.append(candidate)
    return [path.resolve() for path in output]


def sample_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = config.get("Samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Analysis config is missing non-empty Samples.")
    return dict(samples)


def required_sample_value(sample_key: str, sample_cfg: dict[str, Any], field: str) -> Any:
    if field not in sample_cfg or sample_cfg[field] is None:
        raise ValueError(f"Sample '{sample_key}' is missing required field '{field}'.")
    return sample_cfg[field]


def sample_name(sample_key: str, sample_cfg: dict[str, Any]) -> str:
    return str(required_sample_value(sample_key, sample_cfg, "name"))


def sample_is_data(sample_key: str, sample_cfg: dict[str, Any]) -> bool:
    return bool(required_sample_value(sample_key, sample_cfg, "is_data"))


def sample_is_signal(sample_key: str, sample_cfg: dict[str, Any]) -> bool:
    return bool(required_sample_value(sample_key, sample_cfg, "is_signal"))


def sample_raw_files(sample_key: str, sample_cfg: dict[str, Any]) -> list[Path]:
    raw_files = sample_cfg.get("raw_files") or sample_cfg.get("raw_input_files")
    if not raw_files:
        raise ValueError(f"Sample '{sample_name(sample_key, sample_cfg)}' is missing raw_files/raw_input_files.")

    output: list[Path] = []
    for raw_file in raw_files:
        matches = sorted(glob.glob(str(raw_file)))
        output.extend(Path(match).expanduser().resolve() for match in (matches or [str(raw_file)]))
    return output


def analysis_luminosity(config: dict[str, Any]) -> float | None:
    total = 0.0
    found = False
    for sample_key, sample_cfg in sample_configs(config).items():
        if sample_is_data(sample_key, sample_cfg) and sample_cfg.get("lumi") is not None:
            total += float(sample_cfg["lumi"])
            found = True
    return total if found else None


def qi_luminosity(config: dict[str, Any]) -> float:
    luminosity = analysis_luminosity(config)
    if luminosity is None:
        raise ValueError("QI config needs GlobalConfigs.luminosity or at least one data sample lumi.")
    return luminosity


def raw_weight_info(
    config: dict[str, Any],
    sample_key: str,
    sample_cfg: dict[str, Any],
    raw_files: list[Path],
) -> RawWeightInfo:
    if sample_is_data(sample_key, sample_cfg):
        return RawWeightInfo(
            is_data=True,
            weight_scale=1.0,
            total_initial_num_events=None,
            weight_source="data_unit",
        )

    norm_factor = float(required_sample_value(sample_key, sample_cfg, "norm_factor"))
    luminosity = qi_luminosity(config)
    total_initial_num_events = 0.0
    for raw_path in raw_files:
        file_total = read_file_initial_total_num_events(str(raw_path))
        if file_total is None:
            raise ValueError(
                f"MC sample '{sample_key}' raw file '{raw_path}' is missing initial_total_num_events."
            )
        total_initial_num_events += float(file_total)
    if total_initial_num_events <= 0.0:
        raise ValueError(
            f"MC sample '{sample_key}' has non-positive summed initial_total_num_events={total_initial_num_events}."
        )

    return RawWeightInfo(
        is_data=False,
        weight_scale=luminosity * norm_factor / total_initial_num_events,
        total_initial_num_events=total_initial_num_events,
        weight_source="nominal_lumi_times_norm_over_summed_raw_initial_events",
    )


def build_raw_weight_info(config: dict[str, Any], samples: dict[str, dict[str, Any]]) -> dict[str, RawWeightInfo]:
    return {
        sample_key: raw_weight_info(config, sample_key, sample_cfg, sample_raw_files(sample_key, sample_cfg))
        for sample_key, sample_cfg in samples.items()
    }


def class_regions(config: dict[str, Any]) -> list[str]:
    lookup = build_classification_lookup(config)
    return [label for label in lookup.class_labels if label.startswith("Ztautau_")]


def neutrino_prediction_regions(config: dict[str, Any]) -> list[str]:
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    regions: list[str] = []
    for value in prediction_cfg.values():
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            values = [value]

        for item in values:
            if isinstance(item, list):
                regions.extend(str(label) for label in item)
            elif item is not None:
                regions.append(str(item))

    seen: set[str] = set()
    return [region for region in regions if region.startswith("Ztautau_") and not (region in seen or seen.add(region))]


def signal_categories(config: dict[str, Any], regions: list[str]) -> dict[str, list[int]]:
    subcategories = config.get("Subcategories")
    if not isinstance(subcategories, dict) or "Ztautau" not in subcategories:
        raise ValueError("Analysis config is missing Subcategories.Ztautau.")
    categories = subcategories["Ztautau"]
    return {
        region: [int(value) for value in categories[region]]
        for region in regions
        if region.startswith("Ztautau_") and region in categories
    }


def rebuild_vectors(events: ak.Array) -> ak.Array:
    output = events
    for field in output.fields:
        if field.endswith("_p4"):
            output[field] = rebuild_vector(output[field])
    return output


def p4_from_fields(events: ak.Array, prefix: str) -> ak.Array:
    if f"{prefix}_p4" in events.fields:
        return rebuild_vector(events[f"{prefix}_p4"])
    return vector.zip(
        {
            "px": events[f"{prefix}_px"],
            "py": events[f"{prefix}_py"],
            "pz": events[f"{prefix}_pz"],
            "E": events[f"{prefix}_E"],
        }
    )


def finite(values: Any) -> np.ndarray:
    return np.isfinite(to_numpy(values, np.float64))


def finite_p4(p4: ak.Array) -> np.ndarray:
    return finite(p4.px) & finite(p4.py) & finite(p4.pz) & finite(p4.E)



def preselection_mask(events: ak.Array) -> np.ndarray:
    mask = np.ones(len(events), dtype=bool)
    fields = set(events.fields)
    if "baseline_cut" in fields:
        mask &= to_numpy(events["baseline_cut"], bool)
    mask &= event_preselection_mask(events)
    return mask

def mc_prediction_weight_scale(split_fraction: float | None) -> float:
    if split_fraction is None:
        return 1.0
    if not math.isfinite(split_fraction) or split_fraction <= 0.0 or split_fraction > 1.0:
        raise ValueError(f"--mc-split-fraction must be in (0, 1], got {split_fraction}.")
    return 1.0 / split_fraction


def prediction_weight(
    sample_key: str,
    sample_cfg: dict[str, Any],
    events: ak.Array,
    mc_weight_scale: float,
) -> np.ndarray:
    if sample_is_data(sample_key, sample_cfg):
        return np.ones(len(events), dtype=np.float32)
    if "event_weight" not in events.fields:
        raise ValueError(
            f"Prediction parquet for MC sample '{sample_key}' is missing event_weight. "
            "Do not fall back to an inferred normalization."
        )
    return to_numpy(events["event_weight"], np.float32) * np.float32(mc_weight_scale)


def raw_weight(info: RawWeightInfo, num_events: int) -> np.ndarray:
    return np.full(num_events, info.weight_scale, dtype=np.float32)


def label_from_index(indices: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    output = np.full(len(indices), "", dtype=object)
    valid = (indices >= 0) & (indices < len(labels))
    if not np.all(valid):
        bad_values = sorted(set(int(index) for index in indices[~valid]))
        raise ValueError(f"EveNet class index outside configured labels: {bad_values}")
    for index, label in enumerate(labels):
        output[valid & (indices == index)] = label
    return output


def target_labels(events: ak.Array, sample_key: str, config: dict[str, Any]) -> np.ndarray:
    if "classification_target_name" in events.fields:
        return np.asarray(ak.to_list(events["classification_target_name"]), dtype=object)
    if sample_key != "Ztautau":
        return np.full(len(events), sample_key, dtype=object)
    categories = to_numpy(events["event_category"], np.int64)
    lookup = build_classification_lookup(config)
    if "Ztautau" not in lookup.sample_event_category_to_label:
        raise ValueError("Classification lookup is missing Ztautau category labels.")
    mapping = lookup.sample_event_category_to_label["Ztautau"]
    missing_categories = sorted({int(category) for category in categories if int(category) not in mapping})
    if missing_categories:
        raise ValueError(f"Ztautau event categories are not covered by Subcategories: {missing_categories}")
    return np.asarray([mapping[int(category)] for category in categories], dtype=object)


def evenet_labels(events: ak.Array, config: dict[str, Any]) -> np.ndarray:
    if "evenet_pred_class_name" in events.fields:
        return np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    if "evenet_class_index" in events.fields:
        labels = build_classification_lookup(config).class_labels
        return label_from_index(to_numpy(events["evenet_class_index"], np.int64), labels)
    raise ValueError("EveNet region export requires evenet_pred_class_name or evenet_class_index.")


def target_tau_pair(events: ak.Array) -> tuple[ak.Array, ak.Array, np.ndarray]:
    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    tau_a = p4_from_fields(events, "truth_tau_a")
    tau_b = p4_from_fields(events, "truth_tau_b")
    valid = finite_p4(vis_a) & finite_p4(vis_b) & finite_p4(tau_a) & finite_p4(tau_b)
    return tau_a, tau_b, valid


def baseline_valid_mask(events: ak.Array) -> np.ndarray | None:
    if "baseline_flags_valid" in events.fields:
        return to_numpy(events["baseline_flags_valid"], bool)
    if "flags_valid" in events.fields:
        return to_numpy(events["flags_valid"], bool)
    return None


def finite_valid_mask(values_by_name: dict[str, Any]) -> np.ndarray:
    valid = np.ones(len(next(iter(values_by_name.values()))), dtype=bool)
    for values in values_by_name.values():
        valid &= finite(values)
    return valid


def wrapped_delta_phi(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    return (phi_a - phi_b + math.pi) % (2.0 * math.pi) - math.pi


def delta_r(before: ak.Array, after: ak.Array) -> np.ndarray:
    return before.deltaR(after)


def invalid_calibration_fields(num_events: int) -> dict[str, np.ndarray]:
    return {
        "calibration_deltaR_a": np.full(num_events, np.nan, dtype=np.float32),
        "calibration_deltaR_b": np.full(num_events, np.nan, dtype=np.float32),
        "calibration_deltaR_sum": np.full(num_events, np.nan, dtype=np.float32),
    }


def evenet_tau_pair(
    events: ak.Array,
    post_calibration: bool,
) -> tuple[ak.Array, ak.Array, np.ndarray, dict[str, np.ndarray]]:
    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    tau_a_before, tau_b_before = build_tau_pair_from_direction_offsets(
        vis_a,
        vis_b,
        events["evenet_invisible_a_theta"],
        events["evenet_invisible_a_phi"],
        events["evenet_invisible_b_theta"],
        events["evenet_invisible_b_phi"],
    )
    if post_calibration:
        tau_a, tau_b = post_calibrate_tau_tau(tau_a_before, tau_b_before)
    else:
        tau_a, tau_b = tau_a_before, tau_b_before

    valid = to_numpy(events["evenet_invisible_a_valid"], bool) & to_numpy(events["evenet_invisible_b_valid"], bool)
    valid &= finite_p4(vis_a) & finite_p4(vis_b) & finite_p4(tau_a) & finite_p4(tau_b)
    if post_calibration:
        calibration_delta_r_a = np.asarray(delta_r(tau_a_before, tau_a), dtype=np.float32)
        calibration_delta_r_b = np.asarray(delta_r(tau_b_before, tau_b), dtype=np.float32)
        calibration_fields = {
            "calibration_deltaR_a": np.asarray(np.where(valid, calibration_delta_r_a, np.nan), dtype=np.float32),
            "calibration_deltaR_b": np.asarray(np.where(valid, calibration_delta_r_b, np.nan), dtype=np.float32),
            "calibration_deltaR_sum": np.asarray(
                np.where(valid, calibration_delta_r_a + calibration_delta_r_b, np.nan),
                dtype=np.float32,
            ),
        }
    else:
        calibration_fields = invalid_calibration_fields(len(events))
    return tau_a, tau_b, valid, calibration_fields


def method_observables(
    events: ak.Array,
    method: str,
    post_calibration: bool,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    names = export_observable_names()
    if method == "truth":
        if all(f"truth_{name}" in events.fields or name == "mtautau" for name in names):
            output = {
                name: observable_field_values(events, name, (f"truth_{name}",))
                for name in names
            }
            valid = np.ones(len(events), dtype=bool)
            for values in output.values():
                valid &= finite(values)
            return output, valid, invalid_calibration_fields(len(events))
        else:
            raise ValueError(f"Method {method} not implemented.")

    elif method == "target":
        required_prefixes = (
            "lead_a_visible",
            "lead_b_visible",
            "truth_tau_a",
            "truth_tau_b",
        )
        missing = [
            prefix
            for prefix in required_prefixes
            if f"{prefix}_p4" not in events.fields
            and not all(f"{prefix}_{component}" in events.fields for component in ("px", "py", "pz", "E"))
        ]
        if missing:
            missing_truth = [f"truth_{name}" for name in names if f"truth_{name}" not in events.fields]
            raise KeyError(
                "Target method needs truth tau four-vector fields. "
                f"Missing truth observables: {missing_truth}. Missing four-vector prefixes: {missing}."
            )

    elif method == "baseline":
        output = {
            name: observable_field_values(events, name, (f"baseline_{name}", name))
            for name in names
        }
        valid = baseline_valid_mask(events)
        if valid is None:
            valid = np.ones(len(events), dtype=bool)
        valid &= finite_valid_mask(output)
        return output, valid, invalid_calibration_fields(len(events))


    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    if method == "target":
        tau_a, tau_b, valid = target_tau_pair(events)
        calibration_fields = invalid_calibration_fields(len(events))
    elif method == "evenet":
        tau_a, tau_b, valid, calibration_fields = evenet_tau_pair(
            events,
            post_calibration=post_calibration,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    built_observables = build_observables(tau_a, tau_b, vis_a, vis_b)
    output = {name: built_observables[name] for name in names}
    valid &= finite_valid_mask(output)
    return {name: ak.where(valid, values, np.nan) for name, values in output.items()}, valid, calibration_fields


def observable_field_values(events: ak.Array, observable_name: str, candidates: tuple[str, ...]) -> Any:
    for candidate in candidates:
        if candidate in events.fields:
            return events[candidate]
    if observable_name == "mtautau":
        return fixed_mtautau_values(len(events))
    raise KeyError(
        f"Missing observable '{observable_name}'. Expected one of {list(candidates)}. "
        f"Available matching fields: {matching_observable_fields(events, observable_name)}"
    )


def fixed_mtautau_values(num_events: int) -> np.ndarray:
    return np.full(num_events, CM_ENERGY_GEV, dtype=np.float32)


def matching_observable_fields(events: ak.Array, observable_name: str) -> list[str]:
    return sorted(field for field in events.fields if field == observable_name or field.endswith(f"_{observable_name}"))


def base_fields(
    events: ak.Array,
    sample_key: str,
    config: dict[str, Any],
    weights: np.ndarray,
    total_initial_num_events: float | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in BASE_EVENT_FIELDS:
        if name in events.fields:
            fields[name] = events[name]
    for name in events.fields:
        if name.startswith("truth_") and name not in fields:
            fields[name] = events[name]
        if name.endswith("_cut") and name not in fields:
            fields[name] = events[name]

    if weights.shape != (len(events),):
        raise ValueError(f"Weight shape {weights.shape} does not match {len(events)} events for sample '{sample_key}'.")
    fields["predict_weight"] = weights
    fields["weight"] = weights
    fields["weight_nominal"] = weights
    if "initial_total_num_events" not in fields and "initial_num_events" in events.fields:
        fields["initial_total_num_events"] = events["initial_num_events"]
    if total_initial_num_events is not None:
        fields["initial_total_num_events"] = np.full(
            len(events),
            total_initial_num_events,
            dtype=np.float64,
        )
    if "classification_target_name" in events.fields:
        fields["classification_target_name"] = events["classification_target_name"]
    else:
        fields["classification_target_name"] = ak.Array(target_labels(events, sample_key, config).tolist())
    return fields


def region_masks(events: ak.Array, method: str, sample_key: str, regions: list[str], config: dict[str, Any]) -> dict[str, np.ndarray]:
    truth_label = target_labels(events, sample_key, config)
    pred_label = evenet_labels(events, config) if method == "evenet" else None
    output: dict[str, np.ndarray] = {}
    region_to_signals = qi_region_to_signals(config, regions)
    unknown_regions = [region for region in regions if region not in region_to_signals]
    if unknown_regions:
        raise ValueError(
            "QI regions are missing from QIAnalysis.region_to_signals: "
            + ", ".join(unknown_regions)
        )
    signal_labels = [
        signal
        for region in regions
        for signal in region_to_signals[region]
    ]
    categories_cfg = signal_categories(config, signal_labels)
    categories = to_numpy(events["event_category"], np.int64) if "event_category" in events.fields else None

    for region in regions:
        cut = f"{region}_cut"
        region_signals = region_to_signals[region]
        if region.startswith("Ztautau_") and method == "evenet":
            if pred_label is None:
                raise ValueError("Internal error: missing EveNet labels for EveNet region masks.")
            output[region] = np.isin(pred_label, region_signals)
        elif cut in events.fields:
            output[region] = to_numpy(events[cut], bool)
        elif region.startswith("Ztautau_") and categories is not None:
            region_categories = [
                category
                for signal in region_signals
                for category in categories_cfg.get(signal, [])
            ]
            output[region] = np.isin(categories, region_categories)
        elif region.startswith("Ztautau_"):
            output[region] = np.isin(truth_label, region_signals)
        else:
            raise ValueError(f"Unsupported QI region '{region}'.")
    return output


def export_method_events(
    events: ak.Array,
    method: str,
    sample_key: str,
    sample_cfg: dict[str, Any],
    config: dict[str, Any],
    regions: list[str],
    mc_weight_scale: float,
    post_calibration: bool,
) -> ak.Array:
    weights = prediction_weight(sample_key, sample_cfg, events, mc_weight_scale)
    output = base_fields(events, sample_key, config, weights)
    observables, valid, calibration_fields = method_observables(
        events,
        method,
        post_calibration=post_calibration,
    )
    output.update(observables)
    output.update(calibration_fields)
    output["flags_valid"] = valid
    output["mmc_likelihood"] = auxiliary_field(events, method, "mmc_likelihood")

    masks = region_masks(events, method, sample_key, regions, config)
    for region, mask in masks.items():
        output[f"{region}_cut"] = mask
    return ak.Array(output)


def auxiliary_field(events: ak.Array, method: str, name: str) -> ak.Array:
    if name == "mmc_likelihood" and method in {"target", "evenet", "truth"}:
        return np.zeros(len(events), dtype=np.float32)
    candidates = [name]
    if method == "baseline":
        candidates.append(f"baseline_{name}")
    for candidate in candidates:
        if candidate in events.fields:
            return events[candidate]
    raise KeyError(f"Missing required auxiliary field '{name}' for method '{method}'.")


def invalid_float_values(num_events: int) -> np.ndarray:
    return np.full(num_events, np.nan, dtype=np.float32)


def invalid_bool_values(num_events: int) -> np.ndarray:
    return np.zeros(num_events, dtype=bool)


def invalid_likelihood_values(num_events: int) -> np.ndarray:
    return np.zeros(num_events, dtype=np.float32)


def raw_observable_values(raw_events: ak.Array, sample_key: str, name: str, require_field: bool) -> Any:
    if name in raw_events.fields:
        return raw_events[name]
    baseline_name = f"baseline_{name}"
    if baseline_name in raw_events.fields:
        return raw_events[baseline_name]
    if name == "mtautau":
        return fixed_mtautau_values(len(raw_events))
    if require_field:
        raise KeyError(f"RAW sample '{sample_key}' is missing observable '{name}' or '{baseline_name}'.")
    return invalid_float_values(len(raw_events))


def raw_auxiliary_values(raw_events: ak.Array, sample_key: str, name: str, require_field: bool) -> Any:
    if name in raw_events.fields:
        return raw_events[name]
    if require_field:
        raise KeyError(f"RAW sample '{sample_key}' is missing {name}.")
    if name == "flags_valid":
        return invalid_bool_values(len(raw_events))
    if name == "mmc_likelihood":
        return invalid_likelihood_values(len(raw_events))
    raise ValueError(f"Unsupported RAW auxiliary field '{name}'.")


def raw_region_cut_values(num_events: int, region: str) -> np.ndarray:
    if not region.startswith("Ztautau_"):
        raise ValueError(f"Unsupported QI region '{region}'.")
    return invalid_bool_values(num_events)


def export_raw_complement(
    raw_events: ak.Array,
    sample_key: str,
    sample_cfg: dict[str, Any],
    config: dict[str, Any],
    weight_info: RawWeightInfo,
    regions: list[str],
) -> ak.Array:
    keep = ~preselection_mask(raw_events)
    raw_events = raw_events[keep]
    require_qi_fields = sample_is_signal(sample_key, sample_cfg)
    weights = raw_weight(weight_info, len(raw_events))
    output = base_fields(
        raw_events,
        sample_key,
        config,
        weights,
        total_initial_num_events=weight_info.total_initial_num_events,
    )
    for name in export_observable_names():
        output[name] = raw_observable_values(raw_events, sample_key, name, require_qi_fields)
    output["flags_valid"] = raw_auxiliary_values(raw_events, sample_key, "flags_valid", require_qi_fields)
    output["mmc_likelihood"] = raw_auxiliary_values(raw_events, sample_key, "mmc_likelihood", require_qi_fields)
    for region in regions:
        output[f"{region}_cut"] = raw_region_cut_values(len(raw_events), region)
    return ak.Array(output)


def write_cutflow(sample_dir: Path, sample_name: str, events: ak.Array) -> None:
    weight_sum = float(np.sum(to_numpy(events["weight"], np.float64))) if len(events) else 0.0
    record = {
        "step": 0,
        "cut": "initial_total_num_events",
        "events": int(len(events)),
        "weighted_events": weight_sum,
        "efficiency": 1.0,
        "weighted_efficiency": 1.0,
        "relative_efficiency": 1.0,
        "weighted_relative_efficiency": 1.0,
    }
    (sample_dir / f"cutflow_{sample_name}.json").write_text(json.dumps([record], indent=2))


def numeric_field(events: ak.Array, name: str, dtype: Any, fill_value: float | int | bool) -> np.ndarray:
    if name not in events.fields:
        return np.full(len(events), fill_value, dtype=dtype)
    values = ak.fill_none(events[name], fill_value)
    return to_numpy(values, dtype)


def string_field(events: ak.Array, name: str, fill_value: str) -> ak.Array:
    if name not in events.fields:
        return ak.Array([fill_value] * len(events))
    values = ak.fill_none(events[name], fill_value)
    return ak.Array([str(value) for value in ak.to_list(values)])


def final_qi_events(events: ak.Array, sample_name: str, regions: list[str]) -> ak.Array:
    fields: dict[str, Any] = {
        name: numeric_field(events, name, dtype, fill_value)
        for name, dtype, fill_value in FINAL_SCHEMA_FIELDS
    }
    fields["classification_target_name"] = string_field(events, "classification_target_name", sample_name)

    for name in export_observable_names():
        fill_value = CM_ENERGY_GEV if name == "mtautau" else np.nan
        fields[name] = numeric_field(events, name, np.float32, fill_value)
        fields[f"truth_{name}"] = numeric_field(events, f"truth_{name}", np.float32, fill_value)

    fields["flags_valid"] = numeric_field(events, "flags_valid", bool, False)
    fields["mmc_likelihood"] = numeric_field(events, "mmc_likelihood", np.float32, 0.0)
    for vec_name in (
        "lead_a_visible_p4",
        "lead_b_visible_p4",
        "hemisphere_a_visible_p4",
        "hemisphere_b_visible_p4",
    ):
        if vec_name in events.fields:
            fields[vec_name] = events[vec_name]
    for region in regions:
        fields[f"{region}_cut"] = numeric_field(events, f"{region}_cut", bool, False)
    return ak.Array(fields)


def scale_event_weights(events: ak.Array, factor: float) -> ak.Array:
    if factor == 1.0:
        return events
    output = {field: events[field] for field in events.fields}
    for field in ("predict_weight", "weight", "weight_nominal"):
        if field in output:
            output[field] = to_numpy(output[field], np.float32) * np.float32(factor)
    return ak.Array(output)


def write_tree(events: ak.Array, sample_dir: Path, sample_name: str, regions: list[str], compression: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    events = final_qi_events(events, sample_name, regions)
    ak.to_parquet(events, sample_dir / "filtered___raw.parquet", compression=compression)
    for region in regions:
        cut = f"{region}_cut"
        if cut not in events.fields:
            raise KeyError(f"Cannot write region '{region}' because field '{cut}' is missing.")
        mask = to_numpy(events[cut], bool)
        ak.to_parquet(events[mask], sample_dir / f"filtered___{region}.parquet", compression=compression)
    write_cutflow(sample_dir, sample_name, events)


def clean_method_outputs(output_root: Path, methods: list[str]) -> None:
    for method in methods:
        method_dir = output_root / method
        for child_name in ("processed", "_fragments", "run/response_matrices", "run/response_matrices_VR"):
            child = method_dir / child_name
            if child.exists():
                shutil.rmtree(child)
        method_dir.mkdir(parents=True, exist_ok=True)


def sample_fragment_dirs(fragment_root: Path, sample_name: str) -> list[Path]:
    if not fragment_root.exists():
        return []
    output: list[tuple[int, Path]] = []
    for path in fragment_root.iterdir():
        if not path.is_dir():
            continue
        if path.name == sample_name:
            output.append((-1, path))
            continue
        prefix = f"{sample_name}_"
        if not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            output.append((int(suffix), path))
    return [path for _, path in sorted(output)]


def read_parquet_list(paths: list[Path]) -> ak.Array:
    if not paths:
        raise ValueError("Cannot merge an empty parquet list.")
    arrays = [ak.from_parquet(path) for path in paths]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def split_sample_for_train_test(
    method_dir: Path,
    sample_name: str,
    raw_events: ak.Array,
    regions: list[str],
    compression: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    indices = np.arange(len(raw_events), dtype=np.int64)
    split_masks = (
        indices % 2 == 0,
        indices % 2 == 1,
    )
    target_names = (sample_name, f"{sample_name}_000001")
    for target_name, mask in zip(target_names, split_masks):
        split_events = scale_event_weights(raw_events[mask], 2.0)
        sample_dir = method_dir / "processed" / target_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_tree(split_events, sample_dir, sample_name, regions, compression)
        counts["raw"] = counts.get("raw", 0) + len(split_events)
        for region in regions:
            counts[region] = counts.get(region, 0) + int(np.sum(to_numpy(split_events[f"{region}_cut"], bool)))
    return counts


def merge_sample_fragments(
    method_dir: Path,
    sample_name: str,
    regions: list[str],
    compression: str,
) -> dict[str, int]:
    fragment_dirs = sample_fragment_dirs(method_dir / "_fragments", sample_name)
    if not fragment_dirs:
        return {}
    if sample_name == "Ztautau":
        raw_files = [
            fragment_dir / "filtered___raw.parquet"
            for fragment_dir in fragment_dirs
            if (fragment_dir / "filtered___raw.parquet").exists()
        ]
        if not raw_files:
            return {}
        raw_events = read_parquet_list(raw_files)
        return split_sample_for_train_test(method_dir, sample_name, raw_events, regions, compression)

    sample_dir = method_dir / "processed" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for region in ["raw", *regions]:
        files = [
            fragment_dir / f"filtered___{region}.parquet"
            for fragment_dir in fragment_dirs
            if (fragment_dir / f"filtered___{region}.parquet").exists()
        ]
        if not files:
            continue
        events = read_parquet_list(files)
        ak.to_parquet(events, sample_dir / f"filtered___{region}.parquet", compression=compression)
        counts[region] = int(len(events))
        if region == "raw":
            write_cutflow(sample_dir, sample_name, events)
    return counts


def merge_fragment_outputs(
    output_root: Path,
    methods: list[str],
    samples: dict[str, dict[str, Any]],
    regions: list[str],
    compression: str,
) -> dict[str, dict[str, dict[str, int]]]:
    sample_names = list(samples)
    if "data94" not in sample_names:
        sample_names.append("data94")
    merged: dict[str, dict[str, dict[str, int]]] = {}
    for method in methods:
        method_dir = output_root / method
        method_counts: dict[str, dict[str, int]] = {}
        for sample_name_value in sample_names:
            counts = merge_sample_fragments(method_dir, sample_name_value, regions, compression)
            if counts:
                method_counts[sample_name_value] = counts
        fragment_root = method_dir / "_fragments"
        if fragment_root.exists():
            shutil.rmtree(fragment_root)
        merged[method] = method_counts
    return merged


def parquet_schema_names(path: Path) -> set[str]:
    return {field.name for field in pq.ParquetFile(path).schema_arrow}


def existing_columns(path: Path, requested: set[str]) -> list[str]:
    schema_names = parquet_schema_names(path)
    return sorted(column for column in requested if column in schema_names)


def common_export_columns(regions: list[str]) -> set[str]:
    columns = {
        "sample_key",
        "source_sample_index",
        "event_weight",
        "central_weight",
        "classification_target_name",
        "event_category",
        "truth_QI_region",
        "analyzing_power_a",
        "analyzing_power_b",
        "analyzing_power",
        "initial_total_num_events",
        "initial_num_events",
        "nprong",
        "baseline_cut",
        "flags_valid",
        "baseline_flags_valid",
        "mmc_likelihood",
        "baseline_mmc_likelihood",
    }
    columns.update(f"{region}_cut" for region in regions)
    return columns


def p4_columns(prefix: str) -> set[str]:
    return {
        f"{prefix}_p4",
        f"{prefix}_px",
        f"{prefix}_py",
        f"{prefix}_pz",
        f"{prefix}_E",
    }


def prediction_columns(methods: list[str], regions: list[str]) -> set[str]:
    columns = common_export_columns(regions)
    names = export_observable_names()
    columns.update(f"truth_{name}" for name in names)
    if "target" in methods:
        columns.update(p4_columns("lead_a_visible"))
        columns.update(p4_columns("lead_b_visible"))
        columns.update(p4_columns("truth_tau_a"))
        columns.update(p4_columns("truth_tau_b"))
    if "baseline" in methods:
        columns.update(names)
        columns.update(f"baseline_{name}" for name in names)
    if "evenet" in methods:
        columns.update(p4_columns("lead_a_visible"))
        columns.update(p4_columns("lead_b_visible"))
        columns.update(
            {
                "evenet_pred_class_name",
                "evenet_class_index",
                "evenet_invisible_a_theta",
                "evenet_invisible_b_theta",
                "evenet_invisible_a_phi",
                "evenet_invisible_b_phi",
                "evenet_invisible_a_valid",
                "evenet_invisible_b_valid",
            }
        )
    return columns


def raw_columns(regions: list[str]) -> set[str]:
    columns = common_export_columns(regions)
    names = export_observable_names()
    columns.update(names)
    columns.update(f"truth_{name}" for name in names)
    columns.update(f"baseline_{name}" for name in names)
    return columns


def iter_batches(path: Path, batch_size: int, columns: set[str] | None = None):
    parquet = pq.ParquetFile(path)
    selected_columns = None if columns is None else existing_columns(path, columns)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected_columns):
        yield rebuild_vectors(ak.from_arrow(batch))


def fragment_name(sample: str, index: int) -> str:
    return sample if index == 0 else f"{sample}_{index:06d}"


def count_key(method: str, sample_name: str) -> str:
    return f"{method}:{sample_name}"


def record_fragment(
    events: ak.Array,
    output_root: Path,
    method: str,
    sample_name: str,
    fragment_index: int,
    regions: list[str],
    compression: str,
    counts: dict[str, int],
) -> None:
    sample_dir = output_root / method / "_fragments" / fragment_name(sample_name, fragment_index)
    write_tree(events, sample_dir, sample_name, regions, compression)
    counts[count_key(method, sample_name)] = counts.get(count_key(method, sample_name), 0) + len(events)


def export_fragment_outputs(
    events: ak.Array,
    sample_key: str,
    methods: list[str],
    regions: list[str],
    output_root: Path,
    compression: str,
    fragment_index: int,
    counts: dict[str, int],
) -> None:
    for method in methods:
        record_fragment(events, output_root, method, sample_key, fragment_index, regions, compression, counts)


def prediction_sample_keys(events: ak.Array, samples: dict[str, dict[str, Any]]) -> np.ndarray:
    if "sample_key" in events.fields:
        sample_keys = np.asarray(ak.to_list(events["sample_key"]), dtype=object)
    elif "source_sample_index" in events.fields:
        sample_order = tuple(samples.keys())
        source_indices = to_numpy(events["source_sample_index"], np.int64)
        valid = (source_indices >= 0) & (source_indices < len(sample_order))
        if not np.all(valid):
            bad_indices = sorted({int(index) for index in source_indices[~valid]})
            raise ValueError(
                "Prediction parquet source_sample_index contains values outside Samples order: "
                f"{bad_indices}."
            )
        sample_keys = np.asarray([sample_order[int(index)] for index in source_indices], dtype=object)
    else:
        available = ", ".join(events.fields[:40])
        suffix = " ..." if len(events.fields) > 40 else ""
        raise KeyError(
            "Prediction parquet needs sample_key or source_sample_index to identify sample membership. "
            f"Available fields include: {available}{suffix}"
        )

    unknown = sorted({str(key) for key in sample_keys if str(key) not in samples})
    if unknown:
        raise ValueError(f"Prediction parquet contains sample keys not present in analysis config: {unknown}")
    return sample_keys


def export_prediction_file(args: tuple[Any, ...]) -> dict[str, int]:
    pred_path, config, samples, methods, regions, output_root, batch_size, compression, start_index, mc_weight_scale, post_calibration = args
    counts: dict[str, int] = {}
    fragment_index = start_index
    for events in iter_batches(pred_path, batch_size, prediction_columns(methods, regions)):
        sample_keys = prediction_sample_keys(events, samples)
        for sample_key in sorted(set(sample_keys)):
            if sample_key not in samples:
                continue
            sample_cfg = samples[sample_key]
            sample_events = events[sample_keys == sample_key]
            for method in methods:
                method_events = export_method_events(
                    sample_events,
                    method,
                    sample_key,
                    sample_cfg,
                    config,
                    regions,
                    mc_weight_scale,
                    post_calibration,
                )
                record_fragment(method_events, output_root, method, sample_key, fragment_index, regions, compression, counts)
            fragment_index += 1
    return counts


def export_raw_file(args: tuple[Any, ...]) -> dict[str, int]:
    raw_path, sample_key, sample_cfg, config, weight_info, methods, regions, output_root, batch_size, compression, start_index = args
    counts: dict[str, int] = {}
    fragment_index = start_index
    for events in iter_batches(raw_path, batch_size, raw_columns(regions)):
        complement = export_raw_complement(events, sample_key, sample_cfg, config, weight_info, regions)
        if len(complement) == 0:
            continue
        export_fragment_outputs(
            complement,
            sample_key,
            methods,
            regions,
            output_root,
            compression,
            fragment_index,
            counts,
        )
        fragment_index += 1
    return counts


def merge_counts(items: list[dict[str, int]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            output[key] = output.get(key, 0) + value
    return output


def run_jobs(jobs: list[tuple[Any, ...]], fn, workers: int) -> dict[str, int]:
    if workers <= 1:
        return merge_counts([fn(job) for job in jobs])
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return merge_counts(results)


def processor_region_to_signals(config: dict[str, Any], regions: list[str]) -> dict[str, list[str]]:
    mapping = qi_region_to_signals(config, regions)
    return {region: mapping[region] for region in regions}


def write_analysis_config(
    method_dir: Path,
    method: str,
    config: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    regions: list[str],
) -> Path:
    region_to_signals = processor_region_to_signals(config, regions)
    signal_labels = list(dict.fromkeys(
        signal
        for region in regions
        for signal in region_to_signals[region]
    ))
    signal_cfg = signal_categories(config, signal_labels)
    data_loaders = {}
    for sample_key in SAMPLE_ORDER:
        if sample_key not in samples and sample_key != "data94":
            continue
        if sample_key not in samples:
            raise ValueError(f"QI config needs sample '{sample_key}' in analysis config.")
        sample_cfg = samples[sample_key]
        data_loaders[sample_key] = {
            "name": sample_name(sample_key, sample_cfg),
            "is_data": sample_is_data(sample_key, sample_cfg),
            "is_signal": sample_is_signal(sample_key, sample_cfg),
        }
        if "norm_factor" in sample_cfg:
            data_loaders[sample_key]["norm_factor"] = float(sample_cfg["norm_factor"])

    qi_config = {
        "GlobalConfigs": {
            "default_output_dir": str(method_dir / "run"),
            "processed_data_dir": str(method_dir / "processed"),
            "load_regions": ["raw", *regions],
            "verbosity": 1,
            "luminosity": qi_luminosity(config),
            "signal_categories": signal_cfg,
        },
        "DataLoaders": data_loaders,
        "Processors": {
            "QIProcessor": {
                "processor_name": "QIProcessor",
                "output_dir_name": "QI_analysis",
                "asimov_data": True,
                "dict_region_to_signals": region_to_signals,
            },
            "ForwardFoldingProcessor": {
                "processor_name": "ForwardFoldingProcessor",
                "output_dir_name": "ForwardFoldingProcessor",
                "asimov_data": True,
                "dict_region_to_signals": region_to_signals,
            },
        },
    }
    path = method_dir / f"config_{method}.yaml"
    path.write_text(yaml.safe_dump(qi_config, sort_keys=False))
    return path


def main() -> None:
    args = parse_args()
    if args.pseudo_data:
        raise ValueError("--pseudo-data has been removed from export_evenet_qi_inputs.py. Export real data and MC directly.")
    config = read_yaml(args.analysis_config)
    post_calibration = post_calibration_enabled(config)
    samples = sample_configs(config)
    raw_weight_infos = build_raw_weight_info(config, samples)
    qi_mapping = (config.get("QIAnalysis") or {}).get("region_to_signals")
    regions = args.regions or (list(qi_mapping) if isinstance(qi_mapping, dict) else None)
    regions = regions or neutrino_prediction_regions(config) or class_regions(config)
    mc_weight_scale = mc_prediction_weight_scale(args.mc_split_fraction)
    output_root = args.base_dir
    output_root.mkdir(parents=True, exist_ok=True)
    clean_method_outputs(output_root, args.methods)

    prediction_paths = resolve_parquets(args.prediction_parquet)
    if not prediction_paths:
        raise FileNotFoundError("No prediction parquet files were found.")

    raw_jobs = []
    job_index = 1
    for sample_key, sample_cfg in samples.items():
        for raw_path in sample_raw_files(sample_key, sample_cfg):
            raw_jobs.append((
                raw_path,
                sample_key,
                sample_cfg,
                config,
                raw_weight_infos[sample_key],
                args.methods,
                regions,
                output_root,
                args.batch_size,
                args.compression,
                job_index,
            ))
            job_index += 100_000

    pred_jobs = []
    for pred_path in prediction_paths:
        pred_jobs.append((
            pred_path,
            config,
            samples,
            args.methods,
            regions,
            output_root,
            args.batch_size,
            args.compression,
            job_index,
            mc_weight_scale,
            post_calibration,
        ))
        job_index += 100_000

    raw_counts = run_jobs(raw_jobs, export_raw_file, args.num_workers)
    pred_counts = run_jobs(pred_jobs, export_prediction_file, args.num_workers)
    counts = merge_counts([raw_counts, pred_counts])
    merged_outputs = merge_fragment_outputs(output_root, args.methods, samples, regions, args.compression)

    config_paths = {}
    for method in args.methods:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        config_paths[method] = str(write_analysis_config(method_dir, method, config, samples, regions))

    summary = {
        "prediction_files": [str(path) for path in prediction_paths],
        "methods": args.methods,
        "regions": regions,
        "region_to_signals": processor_region_to_signals(config, regions),
        "counts": counts,
        "merged_outputs": merged_outputs,
        "configs": config_paths,
        "mc_split_fraction": args.mc_split_fraction,
        "mc_prediction_weight_scale": mc_weight_scale,
        "post_calibration": post_calibration,
        "raw_weight_info": {sample_key: asdict(info) for sample_key, info in raw_weight_infos.items()},
    }
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
