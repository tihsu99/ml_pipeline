#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for scripts/generate_pipeline_shortcut.py. "
        "Install the existing ml_pipeline Python requirements first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("train", "predict", "eval", "fit")
GPU_STAGES = {"train", "predict"}
CPU_STAGES = {"eval", "fit"}
CHECKPOINT_ROLES = ("classification", "diffusion")
TRAIN_BACKENDS = {"pure-evenet", "dgpo-evenet", "evenet-align"}
PREDICT_RESERVED_OPTIONS = {"num_gpus", "task_num_shards", "task_shard_index"}
PREDICT_PATH_OPTIONS = {
    "converted_parquet",
    "normalization_file",
    "output_dir",
    "shape_metadata",
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    pass


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a YAML mapping.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a YAML list.")
    return value


def _resolve_path(value: Any, *, base: Path, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string.")
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def _validate_scalar(value: Any, label: str) -> None:
    if value is None or isinstance(value, (dict, list)):
        raise ConfigError(f"{label} must be a scalar value.")


def read_pipeline_config(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    with config_path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    _mapping(payload, "pipeline config")
    return resolve_pipeline_config(payload, config_path=config_path, repo_root=repo_root.resolve())


def resolve_pipeline_config(
    payload: dict[str, Any], *, config_path: Path, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    resolved = copy.deepcopy(_mapping(payload, "pipeline config"))
    pipeline = _mapping(resolved.get("pipeline"), "pipeline")
    name = pipeline.get("name")
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
        raise ConfigError(
            "pipeline.name must contain only letters, numbers, dots, underscores, or hyphens."
        )
    output_value = pipeline.get("output_dir", f"generated/{name}")
    pipeline["output_dir"] = _resolve_path(
        output_value, base=repo_root, label="pipeline.output_dir"
    )

    python_command = resolved.get("python", "python3")
    if not isinstance(python_command, str) or not python_command.strip():
        raise ConfigError("python must be a non-empty command string.")
    resolved["python"] = python_command

    configs = _mapping(resolved.get("configs", {}), "configs")
    resolved["configs"] = configs
    for key, value in list(configs.items()):
        configs[key] = _resolve_path(value, base=repo_root, label=f"configs.{key}")

    checkpoints = _mapping(resolved.get("checkpoints", {}), "checkpoints")
    resolved["checkpoints"] = checkpoints
    for role, checkpoint in checkpoints.items():
        checkpoint_data = _mapping(checkpoint, f"checkpoints.{role}")
        source = checkpoint_data.get("source")
        if source not in {"external", "train"}:
            raise ConfigError(
                f"checkpoints.{role}.source must be 'external' or 'train'."
            )
        checkpoint_data["path"] = _resolve_path(
            checkpoint_data.get("path"),
            base=repo_root,
            label=f"checkpoints.{role}.path",
        )

    for stage in STAGES:
        stage_data = _mapping(resolved.get(stage, {}), stage)
        resolved[stage] = stage_data
        enabled = stage_data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{stage}.enabled must be true or false.")
        stage_data["enabled"] = enabled
        if not enabled:
            continue
        _validate_stage(stage, stage_data, resolved, repo_root)

    resolved["_resolved"] = {
        "config_path": str(config_path.resolve()),
        "repo_root": str(repo_root.resolve()),
    }
    return resolved


def _validate_stage(
    stage: str,
    stage_data: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    use_srun = stage_data.get("srun", False)
    if not isinstance(use_srun, bool):
        raise ConfigError(f"{stage}.srun must be true or false.")
    stage_data["srun"] = use_srun
    resources = _mapping(stage_data.get("resources"), f"{stage}.resources")
    for key, minimum in (("nodes", 1), ("gpus_per_node", 0), ("cpus_per_node", 1)):
        value = resources.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigError(f"{stage}.resources.{key} must be an integer >= {minimum}.")

    gpus = resources["gpus_per_node"]
    if stage in GPU_STAGES and gpus < 1:
        raise ConfigError(f"{stage}.resources.gpus_per_node must be >= 1.")
    if stage in CPU_STAGES and gpus != 0:
        raise ConfigError(f"{stage}.resources.gpus_per_node must be 0 (CPU only).")
    if not isinstance(resources.get("time"), str) or not resources["time"].strip():
        raise ConfigError(f"{stage}.resources.time must be a non-empty string.")
    for optional in ("account", "qos", "constraint", "partition"):
        if optional in resources:
            _validate_scalar(resources[optional], f"{stage}.resources.{optional}")

    shifter = _mapping(stage_data.get("shifter", {}), f"{stage}.shifter")
    stage_data["shifter"] = shifter
    shifter_enabled = shifter.get("enabled", False)
    if not isinstance(shifter_enabled, bool):
        raise ConfigError(f"{stage}.shifter.enabled must be true or false.")
    shifter["enabled"] = shifter_enabled
    if shifter_enabled and (
        not isinstance(shifter.get("image"), str) or not shifter["image"].strip()
    ):
        raise ConfigError(f"{stage}.shifter.image is required when Shifter is enabled.")

    _validate_environment(stage, stage_data)
    if stage == "train":
        _validate_train(stage_data, config, repo_root)
    elif stage == "predict":
        _validate_predict(stage_data, config, repo_root)
    else:
        _validate_command_specs(stage, stage_data)


def _validate_environment(stage: str, stage_data: dict[str, Any]) -> None:
    environment = _mapping(stage_data.get("environment", {}), f"{stage}.environment")
    stage_data["environment"] = environment
    for key, value in environment.items():
        if not isinstance(key, str) or not ENV_NAME.fullmatch(key):
            raise ConfigError(f"Invalid environment variable name in {stage}.environment: {key!r}.")
        _validate_scalar(value, f"{stage}.environment.{key}")
    required_env = _list(stage_data.get("required_env", []), f"{stage}.required_env")
    stage_data["required_env"] = required_env
    for key in required_env:
        if not isinstance(key, str) or not ENV_NAME.fullmatch(key):
            raise ConfigError(f"Invalid variable name in {stage}.required_env: {key!r}.")


def _validate_train(
    stage_data: dict[str, Any], config: dict[str, Any], repo_root: Path
) -> None:
    if "train" not in config["configs"]:
        raise ConfigError("configs.train is required when train is enabled.")
    command = _mapping(stage_data.get("command"), "train.command")
    backend = command.get("backend")
    if backend not in TRAIN_BACKENDS:
        raise ConfigError(f"train.command.backend must be one of {sorted(TRAIN_BACKENDS)}.")
    if stage_data["resources"]["nodes"] > 1 and backend != "dgpo-evenet":
        raise ConfigError("Multi-node train is supported only for the existing dgpo-evenet Ray workflow.")
    if "overlay_config" in command:
        command["overlay_config"] = _resolve_path(
            command["overlay_config"], base=repo_root, label="train.command.overlay_config"
        )
    no_overlay = command.get("no_overlay", False)
    if not isinstance(no_overlay, bool):
        raise ConfigError("train.command.no_overlay must be true or false.")
    if no_overlay and "overlay_config" in command:
        raise ConfigError("train.command cannot set both no_overlay and overlay_config.")
    command["no_overlay"] = no_overlay
    args = _list(command.get("args", []), "train.command.args")
    command["args"] = args
    for index, value in enumerate(args):
        _validate_scalar(value, f"train.command.args[{index}]")


def _validate_predict(
    stage_data: dict[str, Any], config: dict[str, Any], repo_root: Path
) -> None:
    for key in ("analysis", "train", "evenet_schema"):
        if key not in config["configs"]:
            raise ConfigError(f"configs.{key} is required when predict is enabled.")
    for role in CHECKPOINT_ROLES:
        if role not in config["checkpoints"]:
            raise ConfigError(f"checkpoints.{role} is required when predict is enabled.")
    options = _mapping(stage_data.get("options", {}), "predict.options")
    stage_data["options"] = options
    reserved = PREDICT_RESERVED_OPTIONS.intersection(options)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ConfigError(f"predict.options cannot override generated shard option(s): {names}.")
    if stage_data["resources"]["nodes"] > 1 and options.get("merge_only"):
        raise ConfigError("predict.options.merge_only cannot be used with multi-node prediction.")
    for key, value in list(options.items()):
        if isinstance(value, dict):
            raise ConfigError(f"predict.options.{key} must be a scalar or list.")
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_scalar(item, f"predict.options.{key}[{index}]")
        if key in PREDICT_PATH_OPTIONS:
            if isinstance(value, list):
                options[key] = [
                    _resolve_path(item, base=repo_root, label=f"predict.options.{key}")
                    for item in value
                ]
            elif value is not None:
                options[key] = _resolve_path(
                    value, base=repo_root, label=f"predict.options.{key}"
                )
def _command_specs(stage: str, stage_data: dict[str, Any]) -> list[dict[str, Any]]:
    if "commands" in stage_data:
        commands = _list(stage_data["commands"], f"{stage}.commands")
        return [_mapping(command, f"{stage}.commands[{index}]") for index, command in enumerate(commands)]
    return [stage_data]


def _validate_command_specs(stage: str, stage_data: dict[str, Any]) -> None:
    commands = _command_specs(stage, stage_data)
    if not commands:
        raise ConfigError(f"{stage}.commands must contain at least one command.")
    for index, command in enumerate(commands):
        label = f"{stage}.commands[{index}]"
        if not isinstance(command.get("entrypoint"), str) or not command["entrypoint"].strip():
            raise ConfigError(f"{label}.entrypoint must be a non-empty string.")
        args = _list(command.get("args", []), f"{label}.args")
        for arg_index, value in enumerate(args):
            _validate_scalar(value, f"{label}.args[{arg_index}]")
        use_python = command.get("python", command["entrypoint"].endswith(".py"))
        if not isinstance(use_python, bool):
            raise ConfigError(f"{label}.python must be true or false.")
        setup_scripts = _list(command.get("setup_scripts", []), f"{label}.setup_scripts")
        for setup_index, value in enumerate(setup_scripts):
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{label}.setup_scripts[{setup_index}] must be a path string.")


def shifter_command(command: list[str], shifter: dict[str, Any]) -> list[str]:
    if not shifter.get("enabled", False):
        return command
    return ["shifter", f"--image={shifter['image']}", *command]


def _cli_options(options: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for key, value in options.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                rendered.append(flag)
        elif value is None:
            continue
        elif isinstance(value, list):
            if value:
                rendered.append(flag)
                rendered.extend(str(item) for item in value)
        else:
            rendered.extend((flag, str(value)))
    return rendered


def train_command(config: dict[str, Any]) -> list[str]:
    command_config = config["train"]["command"]
    command = [
        config["python"],
        str(Path(config["_resolved"]["repo_root"]) / "scripts" / "train_neutrino_backend.py"),
        "--backend",
        command_config["backend"],
        "--base-config",
        config["configs"]["train"],
    ]
    if command_config["no_overlay"]:
        command.append("--no-overlay")
    elif "overlay_config" in command_config:
        command.extend(("--overlay-config", command_config["overlay_config"]))
    if command_config["args"]:
        command.append("--")
        command.extend(str(value) for value in command_config["args"])
    return command


def predict_command(config: dict[str, Any], shard_index: int | None = None) -> list[str]:
    stage = config["predict"]
    resources = stage["resources"]
    command = [
        config["python"],
        str(Path(config["_resolved"]["repo_root"]) / "predict_evenet.py"),
        "--analysis-config",
        config["configs"]["analysis"],
        "--train-config",
        config["configs"]["train"],
        "--evenet-config",
        config["configs"]["evenet_schema"],
        "--classification-checkpoint",
        config["checkpoints"]["classification"]["path"],
        "--diffusion-checkpoint",
        config["checkpoints"]["diffusion"]["path"],
        *_cli_options(stage["options"]),
        "--num-gpus",
        str(resources["gpus_per_node"]),
    ]
    if shard_index is not None:
        command.extend((
            "--task-num-shards", str(resources["nodes"]),
            "--task-shard-index", str(shard_index),
        ))
    return command


def configured_commands(config: dict[str, Any], stage: str) -> list[tuple[Path, list[Path], list[str]]]:
    repo_root = Path(config["_resolved"]["repo_root"])
    output: list[tuple[Path, list[Path], list[str]]] = []
    for spec in _command_specs(stage, config[stage]):
        working_directory = Path(spec.get("working_directory", repo_root)).expanduser()
        if not working_directory.is_absolute():
            working_directory = (repo_root / working_directory).resolve()
        entrypoint = Path(spec["entrypoint"]).expanduser()
        if not entrypoint.is_absolute():
            entrypoint = (working_directory / entrypoint).resolve()
        setup_scripts = []
        for setup in spec.get("setup_scripts", []):
            setup_path = Path(setup).expanduser()
            setup_scripts.append(
                setup_path.resolve()
                if setup_path.is_absolute()
                else (working_directory / setup_path).resolve()
            )
        use_python = spec.get("python", spec["entrypoint"].endswith(".py"))
        command = [str(entrypoint), *(str(value) for value in spec.get("args", []))]
        if use_python:
            command.insert(0, config["python"])
        output.append((working_directory, setup_scripts, command))
    return output


def _sbatch_directives(
    config: dict[str, Any], stage: str, script_dir: Path, *, nodes: int
) -> list[str]:
    resources = config[stage]["resources"]
    directives = [
        f"#SBATCH --job-name={config['pipeline']['name']}-{stage}",
        f"#SBATCH --nodes={nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={resources['cpus_per_node']}",
        f"#SBATCH --time={resources['time']}",
        f"#SBATCH --output={script_dir}/%x-%j.out",
    ]
    if resources["gpus_per_node"]:
        directives.append(f"#SBATCH --gpus-per-task={resources['gpus_per_node']}")
    for key in ("account", "qos", "constraint", "partition"):
        if key in resources:
            directives.append(f"#SBATCH --{key}={resources[key]}")
    shifter = config[stage]["shifter"]
    if shifter["enabled"]:
        directives.append(f"#SBATCH --image={shifter['image']}")
    return directives


def _environment_lines(stage_data: dict[str, Any]) -> list[str]:
    lines = [
        f': "${{{key}:?Set {key} before submitting this job}}"'
        for key in stage_data["required_env"]
    ]
    lines.extend(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in stage_data["environment"].items()
    )
    return lines


def _ray_train_lines(config: dict[str, Any], command: list[str]) -> list[str]:
    stage = config["train"]
    resources = stage["resources"]
    ray = _mapping(stage.get("ray", {}), "train.ray")
    port = ray.get("port", 6379)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ConfigError("train.ray.port must be an integer from 1 to 65535.")
    shifter = stage["shifter"]
    head_start = (
        shlex.join(shifter_command(["ray", "start", "--head"], shifter))
        + f' --node-ip-address="$head_node_ip" --port={port} --dashboard-host=0.0.0.0'
    )
    worker_start = (
        shlex.join(shifter_command(["ray", "start"], shifter))
        + f' --address="$head_node_ip:{port}" --block'
    )
    status = shlex.join(shifter_command(["ray", "status"], shifter))
    launch = shlex.join(shifter_command(command, shifter))
    return [
        "head_node=$(hostname)",
        "head_node_ip=$(hostname --ip-address)",
        'if [[ "$head_node_ip" == *" "* ]]; then',
        "  IFS=' ' read -ra addresses <<<\"$head_node_ip\"",
        '  if [[ ${#addresses[0]} -gt 16 ]]; then',
        "    head_node_ip=${addresses[1]}",
        "  else",
        "    head_node_ip=${addresses[0]}",
        "  fi",
        "fi",
        'export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH:-/tmp}/ray_${SLURM_JOB_ID}}"',
        'mkdir -p "$RAY_TMPDIR"',
        f"{head_start}",
        "for _ in $(seq 1 60); do",
        f"  if {status} >/dev/null 2>&1; then break; fi",
        "  sleep 2",
        "done",
        "worker_num=$((SLURM_JOB_NUM_NODES - 1))",
        'if [[ "$worker_num" -gt 0 ]]; then',
        "  srun \"--nodes=$worker_num\" \"--ntasks=$worker_num\" --ntasks-per-node=1 \\",
        '    "--exclude=$head_node" '
        f"--gpus-per-task={resources['gpus_per_node']} "
        f"--cpus-per-task={resources['cpus_per_node']} \\",
        f"    {worker_start} &",
        "fi",
        f'export RAY_ADDRESS="$head_node_ip:{port}"',
        f"{launch}",
    ]


def _command_stage_lines(config: dict[str, Any], stage: str) -> list[str]:
    lines: list[str] = []
    shifter = config[stage]["shifter"]
    for working_directory, setup_scripts, command in configured_commands(config, stage):
        lines.extend(("(", f"  cd {shlex.quote(str(working_directory))}"))
        if setup_scripts:
            lines.append('  export PYTHONPATH="${PYTHONPATH:-}"')
        lines.extend(f"  source {shlex.quote(str(path))}" for path in setup_scripts)
        lines.append(f"  {shlex.join(shifter_command(command, shifter))}")
        lines.append(")")
    return lines


def interactive_commands(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Build direct commands with an optional per-stage srun prefix."""
    resources = config[stage]["resources"]
    prefix: list[str] = []
    if config[stage]["srun"]:
        tasks = resources["nodes"] if stage == "predict" else 1
        prefix = [
            "srun",
            f"--nodes={resources['nodes']}",
            f"--ntasks={tasks}",
            "--ntasks-per-node=1",
            f"--cpus-per-task={resources['cpus_per_node']}",
            "--kill-on-bad-exit=1",
        ]
        if resources["gpus_per_node"]:
            prefix.append(f"--gpus-per-task={resources['gpus_per_node']}")

    rendered = []
    shifter = config[stage]["shifter"]
    repo_root = Path(config["_resolved"]["repo_root"])
    if stage == "train":
        specs = [(repo_root, [], train_command(config))]
    elif stage == "predict":
        specs = [(repo_root, [], predict_command(config))]
    else:
        specs = configured_commands(config, stage)
    for working_directory, setup_scripts, command in specs:
        launch = shifter_command(command, shifter)
        if stage == "predict" and resources["nodes"] > 1:
            shard_launch = (
                f"exec {shlex.join(launch)} "
                '--task-num-shards "$SLURM_NTASKS" '
                '--task-shard-index "$SLURM_PROCID" --skip-merge'
            )
            rendered.append({
                "stage": stage,
                "cwd": working_directory,
                "command": [*prefix, "bash", "-lc", shard_launch],
            })
            rendered.append({
                "stage": stage,
                "cwd": working_directory,
                "command": [
                    *command, "--merge-only", "--delete-merged-parts",
                ],
            })
            continue
        if setup_scripts:
            shell = " && ".join([
                *(f"source {shlex.quote(str(path))}" for path in setup_scripts),
                f"exec {shlex.join(launch)}",
            ])
            launch = ["bash", "-lc", shell]
        rendered.append({
            "stage": stage,
            "cwd": working_directory,
            "command": [*prefix, *launch],
        })
    return rendered


def print_interactive_summary(
    config: dict[str, Any], stages: Iterable[str], commands: list[dict[str, Any]], *, dry_run: bool,
) -> None:
    print(f"Pipeline: {config['pipeline']['name']}")
    print(f"Mode: {'dry-run' if dry_run else 'run'}")
    for stage in stages:
        print(f"\n[{stage}]")
        if not config[stage]["enabled"]:
            print("disabled")
            continue
        for item in commands:
            if item["stage"] == stage:
                print(f"cwd: {item['cwd']}")
                print(f"command: {shlex.join(item['command'])}")


def run_interactive_commands(config: dict[str, Any], commands: Iterable[dict[str, Any]]) -> None:
    for item in commands:
        stage = item["stage"]
        environment = os.environ.copy()
        environment.update({
            key: str(value) for key, value in config[stage]["environment"].items()
        })
        missing = [
            key for key in config[stage]["required_env"] if not environment.get(key)
        ]
        if missing:
            raise ConfigError(f"Set required environment variable(s): {', '.join(missing)}")
        subprocess.run(
            item["command"], cwd=item["cwd"], env=environment, check=True,
        )


def render_sbatch(
    config: dict[str, Any], stage: str, script_path: Path, *, shard_index: int | None = None
) -> tuple[str, list[str]]:
    stage_data = config[stage]
    nodes = stage_data["resources"]["nodes"] if stage != "predict" else 1
    lines = [
        "#!/bin/bash",
        *_sbatch_directives(config, stage, script_path.parent, nodes=nodes),
        "",
        "set -euo pipefail",
        *_environment_lines(stage_data),
        "",
    ]
    repo_root = config["_resolved"]["repo_root"]
    commands: list[str] = []
    if stage == "train":
        command = train_command(config)
        commands.append(shlex.join(shifter_command(command, stage_data["shifter"])))
        lines.append(f"cd {shlex.quote(repo_root)}")
        if nodes > 1:
            lines.extend(_ray_train_lines(config, command))
        else:
            lines.append(shlex.join(shifter_command(command, stage_data["shifter"])))
    elif stage == "predict":
        if shard_index is None:
            raise ValueError("predict rendering requires a shard index")
        command = predict_command(config, shard_index)
        commands.append(shlex.join(shifter_command(command, stage_data["shifter"])))
        lines.extend(
            (
                f"cd {shlex.quote(repo_root)}",
                shlex.join(shifter_command(command, stage_data["shifter"])),
            )
        )
    else:
        command_specs = configured_commands(config, stage)
        commands.extend(
            shlex.join(shifter_command(command, stage_data["shifter"]))
            for _, _, command in command_specs
        )
        lines.extend(_command_stage_lines(config, stage))
    return "\n".join(lines).rstrip() + "\n", commands


def generate_scripts(config: dict[str, Any], stages: Iterable[str]) -> list[dict[str, Any]]:
    output_root = Path(config["pipeline"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_config.yaml"
    with resolved_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    generated: list[dict[str, Any]] = []
    for stage in stages:
        if not config[stage]["enabled"]:
            continue
        stage_dir = output_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage == "predict":
            paths = [stage_dir / f"predict_{index:03d}.sbatch" for index in range(config[stage]["resources"]["nodes"])]
        else:
            paths = [stage_dir / f"{stage}.sbatch"]
        for index, path in enumerate(paths):
            content, commands = render_sbatch(
                config,
                stage,
                path,
                shard_index=index if stage == "predict" else None,
            )
            path.write_text(content)
            path.chmod(0o755)
            generated.append({"stage": stage, "path": path, "commands": commands})
    return generated


def print_summary(
    config: dict[str, Any], selected_stages: Iterable[str], generated: list[dict[str, Any]], *, mode: str
) -> None:
    print(f"Pipeline: {config['pipeline']['name']}")
    print(f"Mode: {mode}")
    print(f"Resolved config: {Path(config['pipeline']['output_dir']) / 'resolved_config.yaml'}")
    for stage in selected_stages:
        print(f"\n[{stage}]")
        if not config[stage]["enabled"]:
            print("disabled")
            continue
        resources = config[stage]["resources"]
        print(f"nodes: {resources['nodes']}")
        print(f"gpus/node: {resources['gpus_per_node']}")
        if stage == "predict":
            print(f"shards: {resources['nodes']}")
        elif stage in CPU_STAGES:
            print("CPU only")
        print(f"shifter: {'enabled' if config[stage]['shifter']['enabled'] else 'disabled'}")
        for item in generated:
            if item["stage"] != stage:
                continue
            print(item["path"])
            for command in item["commands"]:
                print(f"command: {command}")


def submit_scripts(generated: Iterable[dict[str, Any]]) -> None:
    for item in generated:
        subprocess.run(["sbatch", str(item["path"])], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate thin Slurm shortcuts for the existing ml_pipeline commands."
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline shortcut YAML.")
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands without running or submitting them.")
    mode.add_argument("--submit", action="store_true", help="Generate scripts and submit each one with sbatch.")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run the selected command directly with srun instead of generating sbatch files.",
    )
    args = parser.parse_args(argv)

    try:
        config = read_pipeline_config(args.config)
        stages = list(STAGES) if args.stage == "all" else [args.stage]
        if args.interactive:
            if args.submit:
                raise ConfigError("--interactive cannot be combined with --submit.")
            commands = [
                item
                for stage in stages if config[stage]["enabled"]
                for item in interactive_commands(config, stage)
            ]
            print_interactive_summary(
                config, stages, commands, dry_run=args.dry_run,
            )
            if not args.dry_run:
                run_interactive_commands(config, commands)
            return
        generated = generate_scripts(config, stages)
        selected_mode = "dry-run" if args.dry_run else "submit" if args.submit else "generate"
        print_summary(config, stages, generated, mode=selected_mode)
        if args.submit:
            submit_scripts(generated)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
