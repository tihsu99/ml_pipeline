#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import read_yaml, ordered_class_labels

FOUR_VECTOR_FEATURES = ("energy", "pt", "eta", "phi")


# Class Definition
@dataclass(frozen=True)
class FeatureConfig:
    raw_sequential_fields: tuple[str, ...]
    projected_sequential_feature_names: tuple[str, ...]
    global_fields: tuple[str, ...]
    grouped_sequential_config: dict[str, Any] | None
    all_sequential_fields: tuple[str, ...]
@dataclass(frozen=True)
class EveNetConfig:
    process_topologies: dict[str, dict[str, tuple[str, ...]]]
    generation_conditions: tuple[str, ...]
    generation_global_targets: tuple[str, ...]
    generation_events: tuple[str, ...]
    sequential_tags: dict[str, str]
    global_tags: dict[str, str]
    invisible_features: tuple[str, ...]
    invisible_tags: dict[str, str]

def normalize_part_feature_name(raw_name: str) -> str:
    if raw_name in FOUR_VECTOR_FEATURES:
        return f"Part_{raw_name}"
    return raw_name


def is_group_config(raw_value: Any) -> bool:
    return isinstance(raw_value, dict) and "input" in raw_value


def flatten_leaf_features(raw_inputs: list[Any], *, allow_top_level_groups: bool) -> list[str]:
    leaves: list[str] = []
    for child in raw_inputs:
        if isinstance(child, str):
            leaves.append(normalize_part_feature_name(child))
            continue

        if not isinstance(child, dict) or len(child) != 1:
            raise ValueError("Grouped sequential inputs must contain strings or single-key subgroup mappings.")

        _, child_value = next(iter(child.items()))
        if is_group_config(child_value):
            child_inputs = child_value.get("input", [])
        elif isinstance(child_value, list):
            child_inputs = child_value
        else:
            raise ValueError("Subgroup definitions must be a list or a dict with an 'input' field.")
        leaves.extend(flatten_leaf_features(child_inputs, allow_top_level_groups=False))
    return leaves

def normalize_group_children(raw_inputs: list[Any]) -> list[Any]:
    normalized = []
    for child in raw_inputs:
        if isinstance(child, str):
            normalized.append(normalize_part_feature_name(child))
            continue
        if not isinstance(child, dict) or len(child) != 1:
            raise ValueError("Grouped input children must contain strings or single-key subgroup mappings.")
        child_name, child_value = next(iter(child.items()))
        if is_group_config(child_value):
            normalized_child = normalize_group_node(child_name, child_value, top_level=False)
        elif isinstance(child_value, list):
            normalized_child = {
                "name": child_name,
                "output_dim": 1,
                "input": normalize_group_children(child_value),
            }
        else:
            raise ValueError(
                f"Subgroup '{child_name}' must be a list or a dict with an 'input' field."
            )
        normalized.append({child_name: normalized_child})
    return normalized


def normalize_group_node(name: str, raw_value: Any, *, top_level: bool) -> dict[str, Any]:
    if not is_group_config(raw_value):
        raise ValueError(f"Top-level grouped input '{name}' must define 'index' and 'input'.")

    raw_indices = raw_value.get("index")
    if raw_indices is None:
        if top_level:
            raise ValueError(f"Top-level grouped input '{name}' is missing 'index'.")
        indices: tuple[int, ...] = ()
    else:
        indices = tuple(int(index) for index in raw_indices)

    if top_level and len(indices) == 0:
        raise ValueError(f"Top-level grouped input '{name}' must project to at least one slot.")

    raw_inputs = raw_value.get("input", [])
    if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
        raise ValueError(f"Grouped input '{name}' must contain a non-empty 'input' list.")

    explicit_dim = raw_value.get("dim", raw_value.get("output_dim"))
    if explicit_dim is None:
        output_dim = len(indices) if top_level else 1
    else:
        output_dim = int(explicit_dim)

    if output_dim <= 0:
        raise ValueError(f"Grouped input '{name}' must have a positive 'dim'.")
    if top_level and len(indices) > 0 and output_dim != len(indices):
        raise ValueError(
            f"Top-level grouped input '{name}' has dim={output_dim} but index width={len(indices)}."
        )

    normalized = {
        "name": name,
        "output_dim": output_dim,
        "input": normalize_group_children(raw_inputs),
    }
    if len(indices) > 0:
        normalized["index"] = list(indices)
    return normalized

def build_grouped_sequential_config(part_cfg: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]] | None:
    grouped_roots = []
    for root_name, raw_value in part_cfg.items():
        if isinstance(raw_value, list):
            continue
        grouped_roots.append(normalize_group_node(root_name, raw_value, top_level=True))
    if not grouped_roots:
        return None

    for root_name, raw_value in part_cfg.items():
        if isinstance(raw_value, list):
            raise ValueError(
                f"Inputs.Part.{root_name} still uses the legacy flat list while grouped mode is enabled."
            )

    raw_sequential_fields: list[str] = []
    seen_raw_fields: set[str] = set()
    for root in grouped_roots:
        for raw_name in flatten_leaf_features(root["input"], allow_top_level_groups=True):
            if raw_name in seen_raw_fields:
                raise ValueError(f"Raw sequential feature '{raw_name}' is used multiple times in grouped inputs.")
            seen_raw_fields.add(raw_name)
            raw_sequential_fields.append(raw_name)

    occupied_slots: dict[int, str] = {}
    for root in grouped_roots:
        for slot in root["index"]:
            if slot < 0:
                raise ValueError(f"Projected sequential slot {slot} for '{root['name']}' must be non-negative.")
            if slot in occupied_slots:
                raise ValueError(
                    f"Projected sequential slot {slot} is assigned to both '{occupied_slots[slot]}' and '{root['name']}'."
                )
            occupied_slots[slot] = root["name"]

    projected_dim = max(occupied_slots) + 1
    actual_slots = sorted(occupied_slots)
    expected_slots = list(range(projected_dim))
    if actual_slots != expected_slots:
        raise ValueError(
            "Projected sequential slots must cover a dense range starting at 0. "
            f"Expected {expected_slots}, got {actual_slots}."
        )

    projected_feature_names = [f"grouped_{index}" for index in range(projected_dim)]
    for root in grouped_roots:
        root_leaves = flatten_leaf_features(root["input"], allow_top_level_groups=True)
        if root["name"] == "Momentum":
            expected_momentum = [f"Part_{field}" for field in FOUR_VECTOR_FEATURES]
            if root_leaves == expected_momentum and len(root["index"]) == len(expected_momentum):
                for slot, feature_name in zip(root["index"], expected_momentum):
                    projected_feature_names[slot] = feature_name
                continue
        for local_index, slot in enumerate(root["index"]):
            projected_feature_names[slot] = (
                root["name"] if len(root["index"]) == 1 else f"{root['name']}_{local_index}"
            )

    grouped_config = {
        "SEQUENTIAL": {
            "Source": {
                "projected_feature_names": projected_feature_names,
                "groups": {
                    root["name"]: {
                        key: value
                        for key, value in root.items()
                        if key != "name"
                    }
                    for root in grouped_roots
                },
            }
        }
    }
    return grouped_config, tuple(raw_sequential_fields), tuple(projected_feature_names)



def parse_feature_config(config: dict[str, Any]) -> FeatureConfig:
    inputs_cfg = config.get("Inputs") or {}
    part_cfg = inputs_cfg.get("Part") or {}
    global_cfg = inputs_cfg.get("Global") or {}

    grouped_result = build_grouped_sequential_config(part_cfg)
    grouped_sequential_config, raw_sequential_fields, projected_sequential_feature_names = grouped_result

    global_fields = tuple(str(item) for item in global_cfg.get("Fields"))
    return FeatureConfig(
        raw_sequential_fields=tuple(raw_sequential_fields),
        projected_sequential_feature_names=tuple(projected_sequential_feature_names),
        global_fields=global_fields,
        grouped_sequential_config=grouped_sequential_config,
        all_sequential_fields=tuple(raw_sequential_fields),
    )

def merge_tags(default_tags: dict[str, str], raw_tags: dict[str, Any] | None) -> dict[str, str]:
    merged = dict(default_tags)
    if not raw_tags:
        return merged
    for key, value in raw_tags.items():
        merged[key] = str(value)
    return merged

def parse_evenet_config(schema_config: dict[str, Any], analysis_config: dict[str, Any], feature_config: FeatureConfig) -> EveNetConfig:
    normalization_cfg = analysis_config.get("Normalization") or analysis_config.get("EveNet", {}).get("FeatureTags", {})
    generations_cfg = schema_config.get("Generations") or {}
    invisible_cfg = normalization_cfg.get("Invisible") or {}
    invisible_features = tuple(
        key for key in invisible_cfg.keys()
    )

    process_topologies: dict[str, dict[str, tuple[str, ...]]] = {}
    for process_name, resonance_map in (schema_config.get("Processes") or {}).items():
        process_topologies[process_name] = {
            str(resonance_name): tuple(str(item) for item in products)
            for resonance_name, products in resonance_map.items()
        }

    return EveNetConfig(
        process_topologies=process_topologies,
        generation_conditions=tuple(generations_cfg.get("Conditions", feature_config.global_fields)),
        generation_global_targets=tuple(generations_cfg.get("GlobalTargets", ())),
        generation_events=tuple(generations_cfg.get("Events", feature_config.raw_sequential_fields)),
        sequential_tags=merge_tags({key: "none" for key in feature_config.raw_sequential_fields},
                                    normalization_cfg.get("Sequential")),
        global_tags=merge_tags({key: "none" for key in feature_config.global_fields},
                                normalization_cfg.get("Global")),
        invisible_features=tuple(str(item) for item in invisible_features),
        invisible_tags=merge_tags({item: "none" for item in invisible_features},
                                invisible_cfg),
    )


def lookup_feature_tag(feature_name: str, tags: dict[str, str], default: str = "none") -> str:
    if feature_name in tags:
        return tags[feature_name]
    stripped = feature_name[5:] if feature_name.startswith("Part_") else feature_name
    return tags.get(stripped, default)

def build_event_info_payload(
    class_labels: list[str],
    feature_config,
    evenet_config,
) -> dict[str, Any]:
    missing_processes = [label for label in class_labels if label not in evenet_config.process_topologies]
    if missing_processes:
        raise ValueError(
            "Missing EveNet process topology definitions for: " + ", ".join(missing_processes)
        )

    classification_name = "signal"
    payload: dict[str, Any] = {
        "INPUTS": {
            "SEQUENTIAL": {
                "Source": {
                    feature_name: lookup_feature_tag(feature_name, evenet_config.sequential_tags)
                    for feature_name in feature_config.raw_sequential_fields
                }
            },
            "GLOBAL": {
                "Conditions": {
                    feature_name: lookup_feature_tag(feature_name, evenet_config.global_tags)
                    for feature_name in feature_config.global_fields
                }
            },
        },
        "EVENT": {
            label: {
                resonance_name: list(products)
                for resonance_name, products in evenet_config.process_topologies[label].items()
            }
            for label in class_labels
        },
        "CLASSIFICATIONS": {
            "EVENT": [classification_name],
        },
        "CLASSLABEL": {
            "EVENT": {
                classification_name: [class_labels],
            }
        },
        "GENERATIONS": {
            "Conditions": list(evenet_config.generation_conditions),
            "GlobalTargets": list(evenet_config.generation_global_targets),
            "Events": list(evenet_config.generation_events),
            "Neutrinos": {
                feature_name: evenet_config.invisible_tags.get(feature_name, "none")
                for feature_name in evenet_config.invisible_features
            },
        },
    }
    if feature_config.grouped_sequential_config is not None:
        payload["GROUPED_INPUTS"] = feature_config.grouped_sequential_config
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate event info yaml file"
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("config/analysis_config.yaml"),
        help="Analysis YAML with Samples"
    )
    parser.add_argument(
        "--evenet-config",
        type=Path,
        default=Path("config/evenet_schema.yaml"),
        help="EveNet schema YAML with Processes / Generations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/generated_event_info.yaml"),
        help="Output generated event-info YAML path.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of sample keys from analysis.yaml to keep in the class label list.",
    )
    parser.add_argument(
        "--write-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a small JSON summary next to the generated YAML.",
    )
    args = parser.parse_args()

    analysis_config = read_yaml(args.analysis_config)
    evenet_schema = read_yaml(args.evenet_config)

    feature_config = parse_feature_config(analysis_config)
    evenet_config = parse_evenet_config(evenet_schema, analysis_config, feature_config)

    selected_keys = set(args.samples or [])
    class_labels = ordered_class_labels(analysis_config, selected_keys if selected_keys else None)

    payload = build_event_info_payload(class_labels, feature_config, evenet_config)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        yaml.safe_dump(payload, output_file, sort_keys=False)
    print(f"[ml_pipeline_lite] wrote {output_path}")


if __name__ == "__main__":
    main()
