from pathlib import Path
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import awkward as ak
from dataclasses import dataclass
import vector
import math

PROCESS_LATEX_LABELS = {
    "data94": "Data",
    "data": "Data",
    "Ztautau": r"$Z\to\tau\tau$",
    "Ztautau_pipi": r"$\tau\tau\to\pi\pi$",
    "Ztautau_pirho": r"$\tau\tau\to\pi\rho$",
    "Ztautau_rhopi": r"$\tau\tau\to\rho\pi$",
    "Ztautau_pie": r"$\tau\tau\to\pi e$",
    "Ztautau_epi": r"$\tau\tau\to e\pi$",
    "Ztautau_pimu": r"$\tau\tau\to\pi\mu$",
    "Ztautau_mupi": r"$\tau\tau\to\mu\pi$",
    "Ztautau_rhoe": r"$\tau\tau\to\rho e$",
    "Ztautau_erho": r"$\tau\tau\to e\rho$",
    "Ztautau_rhomu": r"$\tau\tau\to\rho\mu$",
    "Ztautau_murho": r"$\tau\tau\to\mu\rho$",
    "Ztautau_rhorho": r"$\tau\tau\to\rho\rho$",
    "Ztautau_ee": r"$\tau\tau\to ee$",
    "Ztautau_mumu": r"$\tau\tau\to\mu\mu$",
    "Ztautau_emu": r"$\tau\tau\to e\mu$",
    "Ztautau_mue": r"$\tau\tau\to\mu e$",
    "Ztautau_piother": r"$\tau\tau\to\pi+\mathrm{other}$",
    "Ztautau_others": r"$Z\to\tau\tau$ other",
    "Zll": r"$Z\to\ell\ell$",
    "Zqq": r"$Z\to q\bar{q}$",
}

CHANNEL_LATEX_LABELS = {
    "pipi": PROCESS_LATEX_LABELS["Ztautau_pipi"],
    "pirho": PROCESS_LATEX_LABELS["Ztautau_pirho"],
    "rhopi": PROCESS_LATEX_LABELS["Ztautau_rhopi"],
    "rhorho": PROCESS_LATEX_LABELS["Ztautau_rhorho"],
    "pie": PROCESS_LATEX_LABELS["Ztautau_pie"],
    "epi": PROCESS_LATEX_LABELS["Ztautau_epi"],
    "pimu": PROCESS_LATEX_LABELS["Ztautau_pimu"],
    "mupi": PROCESS_LATEX_LABELS["Ztautau_mupi"],
    "rhoe": PROCESS_LATEX_LABELS["Ztautau_rhoe"],
    "erho": PROCESS_LATEX_LABELS["Ztautau_erho"],
    "rhomu": PROCESS_LATEX_LABELS["Ztautau_rhomu"],
    "murho": PROCESS_LATEX_LABELS["Ztautau_murho"],
    "ee": PROCESS_LATEX_LABELS["Ztautau_ee"],
    "mumu": PROCESS_LATEX_LABELS["Ztautau_mumu"],
    "emu": PROCESS_LATEX_LABELS["Ztautau_emu"],
    "mue": PROCESS_LATEX_LABELS["Ztautau_mue"],
    "hadhad": r"$\tau_{\mathrm{had}}\tau_{\mathrm{had}}$",
    "baseline": "baseline",
}

MAX_PART_ENERGY_GEV = 91.25
CM_ENERGY = 91.25
TAU_MASS = 1.77

@dataclass(frozen=True)
class ClassificationLookup:
    class_labels: tuple[str, ...]
    label_to_index: dict[str, int]
    sample_default_label: dict[str, str]
    sample_event_category_to_label: dict[str, dict[int, str]]



def process_latex_label(sample_name: str) -> str:
    return PROCESS_LATEX_LABELS.get(sample_name, sample_name.replace("_", r"\_"))


def channel_latex_label(name: str) -> str:
    channel = name.removeprefix("Ztautau_")
    if channel in CHANNEL_LATEX_LABELS:
        return CHANNEL_LATEX_LABELS[channel]
    if name in PROCESS_LATEX_LABELS:
        return PROCESS_LATEX_LABELS[name]
    return name.replace("_", r"\_")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}

def ordered_class_labels(analysis_config: dict[str, Any], selected_keys: set[str] | None) -> list[str]:
    samples_cfg = analysis_config.get("Samples") or {}
    subcategories_cfg = analysis_config.get("Subcategories") or {}
    labels: list[str] = []

    for sample_key, sample_cfg in samples_cfg.items():
        if selected_keys and sample_key not in selected_keys:
            continue
        if bool(sample_cfg.get("is_data", False)):
            continue
        sample_subcategories = subcategories_cfg.get(sample_key)
        if sample_subcategories:
            labels.extend(str(label) for label in sample_subcategories.keys())
        else:
            sample_name = str(sample_cfg.get("name", sample_key))
            labels.append(sample_name)
    deduplicated: list[str] = []
    seen: set[str] = set()

    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduplicated.append(label)

    return deduplicated

def build_classification_lookup(
    analysis_config: dict[str, Any],
    selected_keys: set[str] | None = None,
) -> ClassificationLookup:
    samples_cfg = analysis_config.get("Samples") or {}
    subcategories_cfg = analysis_config.get("Subcategories") or {}
    class_labels = tuple(ordered_class_labels(analysis_config, selected_keys))
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    sample_default_label: dict[str, str] = {}
    sample_event_category_to_label: dict[str, dict[int, str]] = {}

    for sample_key, sample_cfg in samples_cfg.items():
        if selected_keys and sample_key not in selected_keys:
            continue
        if bool(sample_cfg.get("is_data", False)):
            continue
        sample_name = str(sample_cfg.get("name", sample_key))
        sample_subcategories = subcategories_cfg.get(sample_key)
        if sample_subcategories:
            category_to_label: dict[int, str] = {}
            for label, categories in sample_subcategories.items():
                for category in categories:
                    category_to_label[int(category)] = str(label)
            sample_event_category_to_label[sample_key] = category_to_label
        else:
            sample_default_label[sample_key] = sample_name

    return ClassificationLookup(
        class_labels=class_labels,
        label_to_index=label_to_index,
        sample_default_label=sample_default_label,
        sample_event_category_to_label=sample_event_category_to_label,
    )


def cast_array_like(values: Any, dtype=np.float64):
    if isinstance(values, ak.Array):
        return ak.values_astype(values, dtype)
    return np.asarray(values, dtype=dtype)
def pad_and_flatten_part_feature(values: ak.Array, max_particles: int) -> ak.Array:
    padded = ak.pad_none(values, max_particles, axis=1, clip=True)
    filled = ak.fill_none(padded, 0)
    regular = ak.to_regular(filled, axis=1)
    return ak.values_astype(regular, np.float32)[..., np.newaxis]


def build_momentum4d(px: Any, py: Any, pz: Any, energy: Any) -> ak.Array:
    return ak.zip(
        {
            "px": cast_array_like(px, np.float64),
            "py": cast_array_like(py, np.float64),
            "pz": cast_array_like(pz, np.float64),
            "E": cast_array_like(energy, np.float64),
        },
        with_name="Momentum4D",
    )

def classification_targets_for_sample(
    sample_key: str,
    sample_name: str,
    is_data: bool,
    num_rows: int,
    event_categories: np.ndarray | None,
    lookup: ClassificationLookup,
) -> tuple[np.ndarray, np.ndarray]:
    if is_data:
        return (
            np.full(num_rows, -1, dtype=np.int32),
            np.asarray(["data"] * num_rows, dtype=object),
        )

    if sample_key in lookup.sample_event_category_to_label:
        if event_categories is None:
            raise ValueError(f"Sample '{sample_key}' needs event_category to build classification targets.")
        category_to_label = lookup.sample_event_category_to_label[sample_key]
        names = []
        for category in event_categories.astype(np.int64):
            label = category_to_label.get(int(category))
            if label is None:
                raise ValueError(
                    f"Sample '{sample_key}' has event_category={category} not covered by Subcategories."
                )
            names.append(label)
        name_array = np.asarray(names, dtype=object)
        index_array = np.asarray([lookup.label_to_index[name] for name in names], dtype=np.int32)
        return index_array, name_array

    label = lookup.sample_default_label.get(sample_key, sample_name)
    if label not in lookup.label_to_index:
        raise ValueError(f"Classification label '{label}' for sample '{sample_key}' is not in the class label list.")
    index = lookup.label_to_index[label]
    return (
        np.full(num_rows, index, dtype=np.int32),
        np.asarray([label] * num_rows, dtype=object),
    )

def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return np.ascontiguousarray(ak.to_numpy(values, allow_missing=False).astype(dtype, copy=False))


def event_preselection_mask(events: ak.Array) -> np.ndarray:
    mask = np.ones(len(events), dtype=bool)
    mask &= to_numpy(events["nprong"], np.int64) == 2
    return mask

def build_input_particle_mask(
    events: ak.Array,
    remove_neutral_non_photon: bool,
) -> ak.Array:
    mask = events["Part_fourMomentum_fCoordinates_fT"] < MAX_PART_ENERGY_GEV
    if not remove_neutral_non_photon:
        return mask
    charge = events["Part_charge"]
    abs_pdg_id = abs(events["Part_pdgId"])
    keep_particle = (charge != 0) | (abs_pdg_id == 21)
    return mask & keep_particle

def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(getattr(values, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return vector.zip({"px": values["x"], "py": values["y"], "pz": values["z"], "E": values["t"]})
    raise ValueError(f"Unsupported four-vector fields: {sorted(fields)}")

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    return obj

def post_calibrate_tau_tau(tau_a: ak.Array, tau_b: ak.Array) -> ak.Array:
    # energy = (tau_a + tau_b).mass / 2
    energy = CM_ENERGY / 2
    mass = TAU_MASS
    p = (energy*energy - mass*mass)**0.5

    # reconstruct pt

    pt_a = p / np.cosh(tau_a.eta)
    pt_b = p / np.cosh(tau_b.eta)

    px_a = pt_a * np.cos(tau_a.phi)
    py_a = pt_a * np.sin(tau_a.phi)
    pz_a = pt_a * np.sinh(tau_a.eta)
    px_b = pt_b * np.cos(tau_b.phi)
    py_b = pt_b * np.sin(tau_b.phi)
    pz_b = pt_b * np.sinh(tau_b.eta)

    px_shift = (px_a + px_b)
    py_shift = (py_a + py_b)
    pz_shift = (pz_a + pz_b)

    px_a_corr = px_a - (px_shift/2)
    py_a_corr = py_a - (py_shift / 2)
    pz_a_corr = pz_a - (pz_shift/2)

    # use corrected direction of tau_a only
    norm_a_corr = np.sqrt(px_a_corr ** 2 + py_a_corr ** 2 + pz_a_corr ** 2)

    ux_a = px_a_corr / norm_a_corr
    uy_a = py_a_corr / norm_a_corr
    uz_a = pz_a_corr / norm_a_corr

    # rebuild with fixed |p|
    px_a_final = p * ux_a
    py_a_final = p * uy_a
    pz_a_final = p * uz_a

    # force tau_b to be exactly opposite
    px_b_final = -px_a_final
    py_b_final = -py_a_final
    pz_b_final = -pz_a_final

    tau_a_final = ak.zip(
        {
            "px": px_a_final,
            "py": py_a_final,
            "pz": pz_a_final,
            "energy": np.full_like(px_a_final, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )

    tau_b_final = ak.zip(
        {
            "px": px_b_final,
            "py": py_b_final,
            "pz": pz_b_final,
            "energy": np.full_like(px_b_final, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )
    return tau_a_final, tau_b_final