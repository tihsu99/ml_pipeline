"""Opt-in checkpoint bundles; legacy prediction needs none of these settings."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

import yaml


def read_mapping(path):
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def resolve_path(value, base):
    text = os.path.expandvars(str(value))
    if "$" in text:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(text).expanduser()
    return (base / path).resolve()


def resolve_pipeline_model(model, base):
    """Resolve the user-facing predict.model block without accessing remote files."""
    paths = {"evenet_root", "network", "normalization", "event_info",
             "options_default", "classification_network"}
    if not isinstance(model, dict) or set(model) - paths - {"sampler"}:
        raise ValueError("predict.model accepts evenet_root, network, normalization, event_info, options_default, classification_network, sampler")
    for key in ("evenet_root", "network"):
        if key not in model:
            raise ValueError(f"predict.model.{key} is required")
    result = dict(model)
    for key in paths & model.keys():
        if not isinstance(model[key], str) or not model[key].strip():
            raise ValueError(f"predict.model.{key} must be a non-empty path")
        result[key] = str(resolve_path(model[key], base))
    if model.get("sampler", "legacy") not in {"legacy", "stable_v"}:
        raise ValueError("predict.model.sampler must be legacy or stable_v")
    return result


def pipeline_model_runtime(config):
    base = Path(config.get("_resolved", {}).get("repo_root", Path(__file__).resolve().parent))
    model = resolve_pipeline_model(config["predict"]["model"], base)
    disable_ema = config["predict"].get("options", {}).get("disable_ema", True)
    if not isinstance(disable_ema, bool):
        raise ValueError("predict.options.disable_ema must be true or false")
    return {
        "code_root": model["evenet_root"],
        "defaults": {section: model[key] for section, key in
                     (("options", "options_default"), ("network", "classification_network")) if key in model},
        "diffusion": {
            **{key: model[key] for key in ("network", "normalization", "event_info") if key in model},
            "weights": "state_dict" if disable_ema else "ema_state_dict",
        },
        "sampler": {"x0_mode": model.get("sampler", "legacy")},
    }


def read_runtime(path):
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    spec = read_mapping(path)
    if "pipeline" in spec:
        if "model" not in spec.get("predict", {}):
            raise ValueError("Pipeline YAML has no predict.model settings")
        spec = pipeline_model_runtime(spec)
    unknown = set(spec) - {"code_root", "defaults", "diffusion", "sampler"}
    if unknown:
        raise ValueError(f"Unknown model runtime settings: {sorted(unknown)}")
    for name, allowed in (
        ("defaults", {"options", "network"}),
        ("diffusion", {"config", "network", "event_info", "normalization", "weights"}),
        ("sampler", {"x0_mode"}),
    ):
        section = spec.setdefault(name, {})
        if not isinstance(section, dict) or set(section) - allowed:
            raise ValueError(f"Invalid model runtime section: {name}")
    if "code_root" in spec:
        spec["code_root"] = str(resolve_path(spec["code_root"], path.parent))
        if not (Path(spec["code_root"]) / "evenet" / "network" / "evenet_model.py").is_file():
            raise FileNotFoundError(f"code_root must contain the matching evenet package: {spec['code_root']}")
    if not {"config", "network"}.intersection(spec["diffusion"]):
        raise ValueError("Specify a diffusion network YAML (or a complete evaluation config)")
    weights = spec["diffusion"].setdefault("weights", "state_dict")
    if weights not in {"state_dict", "ema_state_dict"}:
        raise ValueError("diffusion.weights must be state_dict or ema_state_dict")
    if spec["sampler"].get("x0_mode", "legacy") not in {"legacy", "stable_v"}:
        raise ValueError("sampler.x0_mode must be legacy or stable_v")
    for section in (spec["defaults"], spec["diffusion"]):
        for key, value in list(section.items()):
            if key == "weights":
                continue
            section[key] = str(resolve_path(value, path.parent))
            if not Path(section[key]).is_file():
                raise FileNotFoundError(section[key])
    spec["source"] = str(path)
    return spec


def runtime_from_argv(argv):
    # Model imports happen before main(), also in spawned GPU workers.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-runtime-config", type=Path)
    args, _ = parser.parse_known_args(argv)
    return read_runtime(args.model_runtime_config)


def merge_mapping(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_mapping(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_config(path, replacements=None):
    """Resolve EveNet section defaults before moving a config to a temp directory."""
    path = Path(path)
    config = read_mapping(path)
    if "extends_overlay" in config:
        raise ValueError(f"Use the complete evaluation/runtime YAML, not an overlay: {path}")
    config.update(replacements or {})
    for key, value in list(config.items()):
        if isinstance(value, dict) and "default" in value:
            value = dict(value)
            default = resolve_path(value.pop("default"), path.parent)
            config[key] = merge_mapping(read_mapping(default), value)
    dataset = config.get("options", {}).get("Dataset", {})
    if dataset.get("normalization_file"):
        dataset["normalization_file"] = str(resolve_path(dataset["normalization_file"], path.parent))
    return config


def prepare_diffusion_config(spec, output, base_config=None):
    bundle = spec["diffusion"]
    # Explicit bundle files replace their entire section. Do not merge a legacy
    # head's overrides into the collaborator's saved network.
    replacements = {section: read_mapping(bundle[section]) for section in ("network", "event_info")
                    if section in bundle}
    source = bundle.get("config", base_config)
    if source is None:
        raise ValueError("Network-only inference requires the pipeline's prepared train config")
    config = expand_config(source, replacements)
    config.pop("_prediction_model_runtime", None)
    if "normalization" in bundle:
        config.setdefault("options", {}).setdefault("Dataset", {})["normalization_file"] = bundle["normalization"]
    for section in ("options", "network", "event_info", "resonance"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Bundle config is missing the complete {section} section")
    normalization = config["options"].get("Dataset", {}).get("normalization_file")
    if not normalization or not Path(normalization).is_file():
        raise FileNotFoundError(f"Bundle normalization_file does not exist: {normalization}")
    output = Path(output)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output


def check_event_schema(expected, actual):
    def signature(event):
        return (
            [(name, kind, [feature.name for feature in event.input_features[name]])
             for name, kind in event.input_types.items()],
            [feature.name for feature in event.invisible_input_features],
            event.class_label,
            event.process_names,
        )
    if signature(expected) != signature(actual):
        raise ValueError("Diffusion bundle input feature order, invisible features, or class labels differ from the analysis schema")


def load_strict_model_state(model, state_dict):
    """Allow disabled task heads, but never drop a backbone or generation branch."""
    import torch

    source = {}
    for key, value in state_dict.items():
        name = key.removeprefix("model.")
        if name in source:
            raise ValueError(f"Duplicate checkpoint key after prefix removal: {name}")
        source[name] = value
    target = model.state_dict()
    inactive = {"Classification", "Regression", "Assignment", "Segmentation",
                "GlobalGeneration", "ReconGeneration", "TruthGeneration", "famo"}
    active_roots = {key.split(".")[0] for key in target}
    ignored_roots = inactive - active_roots
    if "TruthGeneration" not in active_roots:
        ignored_roots.update({"InvisibleInputProjector", "invisible_normalizer"} - active_roots)
    if "ReconGeneration" not in active_roots:
        ignored_roots.update({"num_point_cloud_normalizer"} - active_roots)
    missing = sorted(set(target) - set(source))
    unexpected = sorted(key for key in source if key not in target and key.split(".")[0] not in ignored_roots)
    mismatched = sorted(key for key in target.keys() & source.keys()
                        if target[key].shape != source[key].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}")
    selected = {key: source[key] for key in target}
    for key, value in selected.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
        if "_normalizer." in key and not torch.equal(value.cpu(), target[key].cpu()):
            raise ValueError(f"Bundle normalization disagrees with checkpoint: {key}")
    model.load_state_dict(selected, strict=True)


def sampling_normalizer(model, batch):
    if hasattr(model, "invisible_coordinate_normalizer"):
        return model.invisible_coordinate_normalizer(batch)
    return model.invisible_normalizer
