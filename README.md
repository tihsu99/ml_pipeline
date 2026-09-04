# ML Pipeline

This repository contains the Ztautau EveNet workflow used to convert central
baseline parquet files into trained EveNet models, predictions, and
QI/unfolding-ready inputs.

Measurement-aware DGPO development uses the side-by-side `EveNet-Align`
backend and an external LEP physics plugin. The existing `evenet_dgpo/` path
remains unchanged as a legacy compatibility implementation until production
parity is established.

The complete workflow is:

1. generate the EveNet event schema
2. build EveNet input shards
3. preprocess the shards into train, validation, test, and data splits
4. train classification and invisible-particle generation models
5. run prediction
6. export predictions to the central QI layout
7. run the central QI and forward-folding processors

## Config-Driven Run Command

`run.sh` is the single pipeline entry point. It reads one YAML and executes the
enabled `train`, `predict`, `eval`, and `fit` commands in that order. It does
not generate or submit `sbatch` files, and prediction is one invocation of the
existing `predict_evenet.py`, matching the commands documented below.

Print every resolved command without running it:

```bash
./run.sh config/pipelines/baseline_pct0p05.yaml --dry-run
```

Run the enabled stages sequentially and stream their output to the terminal:

```bash
./run.sh config/pipelines/baseline_pct0p05.yaml
```

Each stage has an optional YAML switch. The default is direct execution;
enable `srun` only to launch a job step inside an existing interactive Slurm
allocation. Wall time, account, QOS, partition, and constraint are inherited
from that allocation. For multi-node prediction, `resources.nodes` controls
the number of one-task-per-node prediction shards:

```yaml
predict:
  enabled: true
  srun: true

eval:
  enabled: true
  srun: false

fit:
  enabled: true
  srun: false
```

Paths under `configs`, `checkpoints`, and known prediction options are resolved
relative to this repository. A command may also set `working_directory`,
`python`, and `setup_scripts`; this is used to source the parent
`lep_tree_ana/setup.sh` before the central QI command.

For `predict.srun: true` with `predict.resources.nodes: 4`, the central runner
uses one four-node job step:

```bash
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --cpus-per-task=32 --gpus-per-task=4 \
  bash -lc 'python3 predict_evenet.py ... \
    --task-num-shards "$SLURM_NTASKS" \
    --task-shard-index "$SLURM_PROCID" --skip-merge'
```

Each node handles one shard inside the same interactive allocation. After the
job step finishes, `run.sh` performs one `--merge-only --delete-merged-parts`
invocation, leaving the normal merged prediction output rather than four
persistent part files.

## Evaluate Checkpoint A and B

This path uses standard `EveNet-Full`; `evenet_dgpo` is not involved. Run the
following once with checkpoint A and once with checkpoint B. No common random
seed or other A/B alignment is imposed.

From this standalone `ml_pipeline` checkout, set the external checkouts and
artifacts:

```bash
cd /path/to/ml_pipeline

export EVENET_ROOT=/path/to/EveNet-Full
export LEP_TREE_ANA_ROOT=/path/to/lep_tree_ana
export PYTHONPATH="$PWD:$EVENET_ROOT:$LEP_TREE_ANA_ROOT:$PYTHONPATH"

export LABEL=A
export CHECKPOINT=/path/to/checkpoint_A.ckpt
export CLASSIFICATION_CHECKPOINT=/path/to/classification.ckpt
export NORMALIZATION_FILE=/path/to/normalization.pt
export MC_PARQUET=/path/to/evenet_test.parquet
export DATA_PARQUET=/path/to/evenet_data.parquet
export OUTPUT_ROOT=/path/to/checkpoint_evaluation
export RUN_DIR="$OUTPUT_ROOT/$LABEL"
```

Predict MC and data. The split correction is MC-only; change `0.5` if the MC
parquet represents a different fraction of the full sample.

```bash
python3 predict_evenet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --evenet-config config/evenet_schema.yaml \
  --normalization-file "$NORMALIZATION_FILE" \
  --classification-checkpoint "$CLASSIFICATION_CHECKPOINT" \
  --diffusion-checkpoint "$CHECKPOINT" \
  --converted-parquet "$MC_PARQUET" \
  --converted-split-fraction 0.5 \
  --output-dir "$RUN_DIR/prediction/mc" \
  --num-gpus 4 --num-steps 200

python3 predict_evenet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --evenet-config config/evenet_schema.yaml \
  --normalization-file "$NORMALIZATION_FILE" \
  --classification-checkpoint "$CLASSIFICATION_CHECKPOINT" \
  --diffusion-checkpoint "$CHECKPOINT" \
  --converted-parquet "$DATA_PARQUET" \
  --output-dir "$RUN_DIR/prediction/data" \
  --num-gpus 4 --num-steps 200
```

Export and evaluate with the current `lep_tree_ana`:

```bash
python3 export_evenet_qi_inputs.py \
  --analysis-config config/analysis.yaml \
  --prediction-parquet "$RUN_DIR/prediction/mc" "$RUN_DIR/prediction/data" \
  --base-dir "$RUN_DIR/qi" \
  --methods evenet

source "$LEP_TREE_ANA_ROOT/setup.sh"
python3 "$LEP_TREE_ANA_ROOT/bin/tree_ana" \
  --config-yaml "$RUN_DIR/qi/evenet/config_evenet.yaml"
```

Repeat after setting `LABEL=B`, `CHECKPOINT=/path/to/checkpoint_B.ckpt`, and a
new `RUN_DIR`. Other artifacts may also differ; the pipeline does not require A
and B to share them. Finally compare the existing results:

```bash
python3 extract_qi_final_measurements.py \
  --method A:"$OUTPUT_ROOT/A/qi/evenet/run/QI_analysis/results.txt" \
  --method B:"$OUTPUT_ROOT/B/qi/evenet/run/QI_analysis/results.txt" \
  --output-prefix "$OUTPUT_ROOT/A_vs_B"
```

By default the exporter keeps the original one-region/one-signal QI mapping.
If a checkpoint should use the current combined scratch prescription, add a
`QIAnalysis.region_to_signals` mapping to that checkpoint's analysis YAML; the
generated `config_evenet.yaml` passes it directly to `lep_tree_ana`.

## Repository Structure

`ml_pipeline` is a submodule of `lep_tree_ana`. It also contains two nested
submodules:

```text
lep_tree_ana
└── ml_pipeline                          github.com/tihsu99/ml_pipeline
    └── EveNet-Full                      branch: tautau
        └── evenet                       github.com/tihsu99/Core
                                         branch: tautau
```

This repository also vendors a local DGPO compatibility layer under
`ml_pipeline/evenet_dgpo/` so the neutrino-generation backend can be switched
without changing the current prediction and export scripts.

The commands in this README assume that the current working directory is the
top-level `lep_tree_ana` directory:

```bash
cd /path/to/lep_tree_ana
```

Do not run the examples from inside `ml_pipeline` unless the command explicitly
says otherwise.

## Quick Start

Initialize the ML submodule chain:

```bash
git submodule update --init --recursive ml_pipeline
```

Create a Python 3.12 environment and install EveNet:

```bash
python3.12 -m venv .venv-ml
source .venv-ml/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r ml_pipeline/EveNet-Full/requirements.txt
python3 -m pip install -e ml_pipeline/EveNet-Full
python3 -m pip install -r ml_pipeline/evenet_dgpo/requirements.txt
```

Expose the local pipeline modules:

```bash
export PYTHONPATH="$PWD/ml_pipeline:$PWD/ml_pipeline/EveNet-Full:$PYTHONPATH"
```

Verify the installation:

```bash
python3 -c "import evenet, awkward, torch; print(torch.__version__)"
evenet-train --help
git submodule status --recursive ml_pipeline
```

The submodule status should list commits for all three repositories:

```text
ml_pipeline
ml_pipeline/EveNet-Full
ml_pipeline/EveNet-Full/evenet
```

Before running a campaign:

1. edit `ml_pipeline/config/analysis.yaml`
2. generate `ml_pipeline/config/generated_event_info.yaml`
3. update all machine-specific paths in the selected training configs
4. set `WANDB_API_KEY`

## Requirements

- Python 3.12
- Git with submodule support
- CUDA-capable GPUs for normal training and inference
- access to the central selected and raw parquet files
- a writable campaign output directory
- a Weights & Biases API key for the current EveNet training entry point

Set the Weights & Biases key before training:

```bash
export WANDB_API_KEY="<your-key>"
```

The pinned Python dependencies are listed in
`ml_pipeline/EveNet-Full/requirements.txt`.

## Updating Submodules

Update only the ML submodule chain:

```bash
git submodule sync --recursive ml_pipeline
git submodule update --init --recursive ml_pipeline
```

Specifying `ml_pipeline` avoids initializing unrelated top-level submodules.

The parent repository records exact commits. The configured branches are useful
when deliberately updating the nested repositories, but a normal checkout uses
the recorded commits for reproducibility.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `ml_pipeline/config/analysis.yaml` | Samples, input paths, luminosity, labels, regions, and feature layout. |
| `ml_pipeline/config/evenet_schema.yaml` | Process and generation schema used to generate EveNet event information. |
| `ml_pipeline/config/preprocess_config.yaml` | EveNet preprocessing configuration. |
| `ml_pipeline/config/train_*_cls.yaml` | Classification training configurations. |
| `ml_pipeline/config/train_pretrain.yaml` | Pretrained diffusion/generation configuration. |
| `ml_pipeline/config/train_scratch.yaml` | Scratch diffusion/generation configuration. |
| `ml_pipeline/config/ad_stage_overlay.yaml` | Stage-specific overlay for supervised AD/foundation training with its own checkpoint lineage. |
| `ml_pipeline/config/dgpo_ztautau_overlay.yaml` | Ztautau-specific overlay that enables the DGPO backend. |
| `ml_pipeline/config/dgpo_post_training_overlay.yaml` | Stage-specific overlay for DGPO post-training with its own checkpoint lineage. |
| `ml_pipeline/build_evenet_input_from_parquet.py` | Convert central selected parquets into EveNet input shards. |
| `ml_pipeline/generate_event_info_yaml.py` | Generate the EveNet event schema and JSON summary. |
| `ml_pipeline/preprocess_evenet_parquet.py` | Create training splits and normalization metadata. |
| `ml_pipeline/predict_evenet.py` | Run classification and invisible-particle inference. |
| `ml_pipeline/export_evenet_qi_inputs.py` | Export predictions to the central QI/unfolding layout. |
| `ml_pipeline/scripts/train_neutrino_backend.py` | Shared launcher for legacy EveNet/DGPO or the EveNet-Align backend. |
| `ml_pipeline/scripts/run_ad_stage.py` | Stage 1 wrapper: freeze a classification backbone, export `event_token`/`object_token`, and train the latent constraint autoencoder. |
| `ml_pipeline/scripts/run_dgpo_stage.py` | Stage 2 wrapper: launch DGPO/diffusion post-training on the AD-augmented parquet. |
| `ml_pipeline/evenet_dgpo/` | Vendored EveNet + DGPO stack, including RL and latent-SWD constraint tooling. |
| `ml_pipeline/monitor_input.py` | Produce optional input monitoring plots. |
| `ml_pipeline/plot_channel_purity_side_by_side.py` | Compare channel yield, purity, and significance. |
| `ml_pipeline/extract_qi_calibration_magnitude.py` | Summarize post-calibration shifts. |
| `ml_pipeline/extract_qi_final_measurements.py` | Parse and plot final QI measurements. |
| `ml_pipeline/extract_response_matrix_summary.py` | Summarize central response matrices. |
| `ml_pipeline/util/` | Additional validation, debugging, and legacy-compatible utilities. |

## Configure a Campaign

### Analysis Configuration

Edit `ml_pipeline/config/analysis.yaml` before starting a new production.

Check these fields:

- `Samples.<sample>.input_files`: selected central parquet files, usually
  `filtered___baseline.parquet`
- `Samples.<sample>.raw_files`: raw central parquet files, usually
  `filtered___raw.parquet`
- `Samples.<sample>.is_data` and `is_signal`: sample behavior and target setup
- `Samples.<sample>.lumi`: data luminosity in pb
- `Samples.<sample>.norm_factor`: MC cross section or normalization factor in pb
- `Inputs.Part` and `Inputs.Global`: EveNet input features
- `Subcategories`: signal channel labels
- `NeutrinoPrediction`: channels used for invisible-particle training and
  prediction
- `EveNetEvaluation.post_calibration`: whether monitoring and QI export apply
  the back-to-back momentum calibration after prediction; omitted defaults to
  `true`

### Training Configuration

Choose the appropriate configuration:

| Goal | Configuration |
| --- | --- |
| Pretrained classification | `ml_pipeline/config/train_pretrain_cls.yaml` |
| Scratch classification | `ml_pipeline/config/train_scratch_cls.yaml` |
| Pretrained generation | `ml_pipeline/config/train_pretrain.yaml` |
| Scratch generation | `ml_pipeline/config/train_scratch.yaml` |

Every selected training config must be reviewed for machine-specific paths.
At minimum, check:

- `platform.data_parquet_dir`
- `platform.data_parquet_val_dir`
- `logger.wandb.entity`
- `logger.local.save_dir`
- `options.default`
- `options.Training.model_checkpoint_save_path`
- `options.Training.model_checkpoint_load_path`
- `options.Training.pretrain_model_load_path`
- `options.Dataset.normalization_file`
- `network.default`
- `event_info.default`
- `resonance.default`

Some committed configs contain example paths from previous NERSC campaigns.
Replace every `/global/...` and `/pscratch/...` path before training.

Relative `default` paths are resolved relative to the training config file.
For example:

```yaml
options:
  default: ../EveNet-Full/share/options/options.yaml

network:
  default: ../EveNet-Full/share/network/network-20M.yaml

event_info:
  default: generated_event_info.yaml

resonance:
  default: resonance.yaml
```

## Campaign Paths

The examples below use one campaign directory:

```bash
export CAMPAIGN_DIR=/path/to/campaign
export INPUT_DIR="$CAMPAIGN_DIR/evenet-input-shards"
export PREPROCESS_DIR="$CAMPAIGN_DIR/evenet-input"
export PRED_DIR="$CAMPAIGN_DIR/evenet-prediction"
export QI_DIR="$CAMPAIGN_DIR/qi-study"

mkdir -p "$CAMPAIGN_DIR"
```

## Two-Stage AD + DGPO

The wrapped DGPO workflow is intentionally split into two commands:

1. `AD`: use a frozen classification backbone to export `event_token` and
   `object_token`, then train the latent constraint autoencoder
2. `DGPO`: run diffusion/DGPO post-training on the augmented parquet from
   stage 1

Minimal example:

```bash
python3 ml_pipeline/scripts/run_ad_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --ad-backbone-checkpoint /path/to/ad_backbone.ckpt

python3 ml_pipeline/scripts/run_dgpo_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --dgpo-init-checkpoint /path/to/diffusion_init.ckpt
```

Those are only the smallest useful invocations. In practice, both wrappers also
inherit a lot of behavior from their base config and overlay config files, so
you usually need to know which defaults you are accepting.

Stage 1 `run_ad_stage.py` defaults:

- `--ad-base-config ml_pipeline/config/train_pretrain_cls.yaml`
- `--ad-overlay-config ml_pipeline/config/ad_stage_overlay.yaml`
- `--augmented-dirname ad_augmented`
- `--token-batch-size 1024`
- `--token-devices auto`
- if `--ad-backbone-checkpoint` is omitted, the script falls back to
  `options.Training.model_checkpoint_load_path` or
  `options.Training.pretrain_model_load_path` from the AD config
- if `--latent-checkpoint` is omitted, the script trains a new latent
  constraint model and writes it under
  `<stage-root>/latent_constraint/checkpoints`

Stage 2 `run_dgpo_stage.py` defaults:

- `--diffusion-base-config ml_pipeline/config/train_pretrain.yaml`
- `--diffusion-overlay-config ml_pipeline/config/dgpo_post_training_overlay.yaml`
- `--augmented-dirname ad_augmented`
- if `--latent-checkpoint` is omitted, the script expects stage 1 to have
  already produced `<stage-root>/latent_constraint/checkpoints/last.ckpt`
- if `--dgpo-init-checkpoint` is omitted, the diffusion init checkpoint comes
  from `options.Training.model_checkpoint_load_path` in the DGPO config / overlay

Important prerequisites:

- the AD base config must point to a split root in the form
  `platform.data_parquet_dir=<root>/train` and
  `platform.data_parquet_val_dir=<root>/val`; the wrapper validates this and
  exports tokens for both splits together
- the diffusion stage expects the augmented parquet from stage 1 to exist under
  `<stage-root>/<augmented-dirname>/train` and
  `<stage-root>/<augmented-dirname>/val`
- the normalization file, Ray worker count, GPU usage, and most dataset /
  training settings are not passed on the CLI here; they are inherited from the
  selected base config plus overlay config

Recommended explicit invocation:

```bash
python3 ml_pipeline/scripts/run_ad_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --ad-base-config ml_pipeline/config/train_pretrain_cls.yaml \
  --ad-overlay-config ml_pipeline/config/ad_stage_overlay.yaml \
  --ad-backbone-checkpoint /path/to/ad_backbone.ckpt \
  --token-batch-size 1024 \
  --token-devices auto

python3 ml_pipeline/scripts/run_dgpo_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --diffusion-base-config ml_pipeline/config/train_pretrain.yaml \
  --diffusion-overlay-config ml_pipeline/config/dgpo_post_training_overlay.yaml \
  --latent-checkpoint "$CAMPAIGN_DIR/dgpo_run/latent_constraint/checkpoints/last.ckpt" \
  --dgpo-init-checkpoint /path/to/diffusion_init.ckpt
```

Useful variants:

- reuse precomputed token parquet and skip frozen token export:

```bash
python3 ml_pipeline/scripts/run_ad_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --skip-export
```

- reuse an existing latent constraint checkpoint instead of retraining it:

```bash
python3 ml_pipeline/scripts/run_ad_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --ad-backbone-checkpoint /path/to/ad_backbone.ckpt \
  --latent-checkpoint /path/to/existing_latent.ckpt
```

- point stage 2 at a latent checkpoint stored somewhere else:

```bash
python3 ml_pipeline/scripts/run_dgpo_stage.py \
  --stage-root "$CAMPAIGN_DIR/dgpo_run" \
  --latent-checkpoint /path/to/existing_latent.ckpt \
  --dgpo-init-checkpoint /path/to/diffusion_init.ckpt
```

What each stage writes:

- stage 1 token export writes augmented parquet under
  `<stage-root>/<augmented-dirname>/train` and
  `<stage-root>/<augmented-dirname>/val`
- stage 1 latent training writes runtime YAML under
  `<stage-root>/runtime_configs/` and checkpoints / logs under
  `<stage-root>/latent_constraint/`
- stage 2 writes a generated DGPO overlay under
  `<stage-root>/runtime_configs/` and DGPO checkpoints under
  `<stage-root>/diffusion/checkpoints`

Notes:

- `--ad-backbone-checkpoint` and `--dgpo-init-checkpoint` are intentionally
  separate, so AD and DGPO can start from different checkpoints
- `run_ad_stage.py` does not fine-tune the backbone; it only uses the frozen
  checkpoint to export tokens for the AD stack
- both commands share the same `--stage-root`; stage 1 writes the augmented
  parquet and latent checkpoint there, and stage 2 reads them back

## End-to-End Workflow

### 1. Generate EveNet Event Information

Regenerate the event schema after changing sample labels, feature definitions,
subcategories, neutrino-prediction channels, or the EveNet schema.

```bash
python3 ml_pipeline/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline/config/generated_event_info.yaml
```

Outputs:

- `ml_pipeline/config/generated_event_info.yaml`
- `ml_pipeline/config/generated_event_info.summary.json`

The generated class order must match the class order used to build input shards
and train EveNet.

### 2. Build EveNet Input Shards

Convert selected central parquet files into EveNet-ready shards:

```bash
python3 ml_pipeline/build_evenet_input_from_parquet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --output-dir "$INPUT_DIR" \
  --batch-size 50000 \
  --rows-per-shard 100000 \
  --num-workers 4
```

Run only a subset of samples when testing:

```bash
python3 ml_pipeline/build_evenet_input_from_parquet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --output-dir "$INPUT_DIR" \
  --samples Ztautau Zll Zqq \
  --num-workers 4
```

Outputs:

- `$INPUT_DIR/shards/<sample>/*.parquet`
- `$INPUT_DIR/monitoring/<sample>/*.png`
- `$INPUT_DIR/monitoring/comparison/*.png`
- `$INPUT_DIR/manifest.json`

Important behavior:

- data events receive unit event weight
- MC weights use `lumi * norm_factor / sum(initial_total_num_events)`
- signal events receive truth invisible targets and truth angular observables
- data events receive classification index `-1`
- `$INPUT_DIR/manifest.json` is the input to preprocessing

Success check:

```bash
test -f "$INPUT_DIR/manifest.json"
```

### 3. Preprocess for EveNet

Create train, validation, test, diffusion, and data splits. The default split is
`0.4,0.1,0.5`.

```bash
python3 ml_pipeline/preprocess_evenet_parquet.py \
  --manifest "$INPUT_DIR/manifest.json" \
  --config ml_pipeline/config/preprocess_config.yaml \
  --store-dir "$PREPROCESS_DIR" \
  --split-ratio 0.4,0.1,0.5 \
  --num-workers 4 \
  --verbose
```

Outputs:

- `$PREPROCESS_DIR/train/*.parquet`
- `$PREPROCESS_DIR/val/*.parquet`
- `$PREPROCESS_DIR/test/*.parquet`
- `$PREPROCESS_DIR/train-diffusion/*.parquet`
- `$PREPROCESS_DIR/val-diffusion/*.parquet`
- `$PREPROCESS_DIR/test-diffusion/*.parquet`
- `$PREPROCESS_DIR/data/*.parquet`
- `$PREPROCESS_DIR/shape_metadata.json`
- `$PREPROCESS_DIR/normalization.pt`
- `$PREPROCESS_DIR/preprocess_manifest.json`

Use `train`, `val`, and `test` for classification. Use `train-diffusion`,
`val-diffusion`, and `test-diffusion` for invisible-particle generation. The
diffusion splits contain only events with valid invisible targets.

Success check:

```bash
test -f "$PREPROCESS_DIR/normalization.pt"
test -f "$PREPROCESS_DIR/shape_metadata.json"
```

### 4. Train EveNet Models

Before training, update the selected config so that:

- classification configs use `$PREPROCESS_DIR/train` and `$PREPROCESS_DIR/val`
- generation configs use `$PREPROCESS_DIR/train-diffusion` and
  `$PREPROCESS_DIR/val-diffusion`
- `options.Dataset.normalization_file` points to
  `$PREPROCESS_DIR/normalization.pt`
- checkpoint paths point to writable campaign directories
- all included config paths exist on the current machine

Classification:

```bash
evenet-train ml_pipeline/config/train_pretrain_cls.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/pretrain-cls"

evenet-train ml_pipeline/config/train_scratch_cls.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/scratch-cls"
```

Invisible-particle generation:

```bash
evenet-train ml_pipeline/config/train_pretrain.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/pretrain-diffusion"

evenet-train ml_pipeline/config/train_scratch.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/scratch-diffusion"
```

The best checkpoints are written below
`options.Training.model_checkpoint_save_path`.

### 4b. Fine-Tune the Neutrino Backend with DGPO

The DGPO path reuses the same base training YAML, then overlays RL-specific
settings at launch time.

Before running DGPO, make sure:

- `config/train_pretrain.yaml` points to the diffusion train/val parquet and the
  correct `normalization.pt`
- `options.Training.model_checkpoint_load_path` points to the supervised
  diffusion checkpoint you want to fine-tune
- the DGPO parquet carries `event_token` and `object_token`
- `config/dgpo_ztautau_overlay.yaml` is adjusted for the current campaign,
  especially the latent-SWD checkpoint inputs if you enable the frozen
  constraint

Launch the original foundation trainer through the shared wrapper:

```bash
python3 ml_pipeline/scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config ml_pipeline/config/train_pretrain.yaml \
  --overlay-config ml_pipeline/config/ad_stage_overlay.yaml
```

Launch DGPO fine-tuning from the same base config:

```bash
python3 ml_pipeline/scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config ml_pipeline/config/train_pretrain.yaml \
  --overlay-config ml_pipeline/config/dgpo_post_training_overlay.yaml
```

Pass extra trainer arguments after `--`, for example:

```bash
python3 ml_pipeline/scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config ml_pipeline/config/train_pretrain.yaml \
  --ray-dir "$CAMPAIGN_DIR/ray/dgpo-diffusion"
```

Launch the bounded EveNet-Align measurement-DGPO debug configuration with:

```bash
python3 ml_pipeline/scripts/train_neutrino_backend.py \
  --backend evenet-align \
  --base-config ml_pipeline/config/train_pretrain.yaml \
  --overlay-config ml_pipeline/config/measurement_dgpo_cdiag_overlay.yaml \
  --ray-dir ./outputs/measurement_dgpo_cdiag_debug/ray
```

This selects `measurement_dgpo` and loads the external LEP Cdiag physics
plugin. The legacy `evenet_dgpo/` implementation remains available unchanged.

Select the explicit full spin-density-matrix objective with the config-driven
shortcut:

```bash
python3 ml_pipeline/scripts/train_neutrino_backend.py \
  --backend evenet-align \
  --measurement-objective sdm_frobenius \
  --base-config ml_pipeline/config/train_pretrain.yaml \
  --ray-dir ./outputs/measurement_dgpo_sdm_debug/ray
```

This chooses `measurement_dgpo_sdm_overlay.yaml`; an explicit
`--overlay-config` always takes precedence.

The wrapper accepts both `--ray-dir` and `--ray_dir`, then forwards the spelling
required by the selected backend. Arguments after `--` remain a raw pass-through
escape hatch for other trainer options.

The launcher writes a temporary merged runtime YAML, sets `PYTHONPATH` so the
vendored `evenet_dgpo` package is importable, and then dispatches to either
`evenet_dgpo/evenet/train.py` or
`evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py`.

Recommended stage split:

- `config/ad_stage_overlay.yaml`: use a dedicated `pretrain_model_load_path`
  and output directory for the supervised AD/foundation stage.
- `config/dgpo_post_training_overlay.yaml`: use a dedicated
  `model_checkpoint_load_path` and output directory for DGPO post-training.
- `config/dgpo_ztautau_overlay.yaml`: keep as the minimal domain/RL overlay
  when you only want to switch the base config into DGPO mode without defining
  a separate stage lineage.

### 5. Run EveNet Prediction

Run MC prediction on the converted test split:

```bash
python3 predict_evenet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --evenet-config config/evenet_schema.yaml \
  --classification-checkpoint /path/to/classification/best.ckpt \
  --diffusion-checkpoint /path/to/diffusion/best.ckpt \
  --converted-parquet "$PREPROCESS_DIR/test" \
  --shape-metadata "$PREPROCESS_DIR/shape_metadata.json" \
  --output-dir "$PRED_DIR/mc" \
  --converted-split-fraction 0.5 \
  --batch-size 8192 \
  --num-gpus 4 \
  --num-steps [NUM STEP] \
```

Run data prediction separately:

```bash
python3 predict_evenet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --evenet-config config/evenet_schema.yaml \
  --classification-checkpoint /path/to/classification/best.ckpt \
  --diffusion-checkpoint /path/to/diffusion/best.ckpt \
  --converted-parquet "$PREPROCESS_DIR/data" \
  --shape-metadata "$PREPROCESS_DIR/shape_metadata.json" \
  --output-dir "$PRED_DIR/data" \
  --batch-size 8192 \
  --num-gpus 4 \
  --num-steps [NUM STEP] \

```

Useful options:

- `--task-num-shards` and `--task-shard-index`: distribute prediction across
  independent scheduler jobs
- `--skip-merge`: keep `*.part*.parquet` outputs
- `--merge-only`: merge existing parts without rerunning inference
- `--delete-merged-parts`: remove parts after a successful merge
- `--use-truth-classification`: skip classification inference and use stored
  truth classes

Prediction outputs:

- `<input_stem>__evenet_pred.parquet`
- optional `<input_stem>__evenet_pred.partNNN.parquet`

`--converted-split-fraction 0.5` rescales MC weights because the test split
contains half of the original MC events. Do not use it for data.

### 6. Export Predictions to QI Inputs

Export prediction parquets to the central processed-parquet layout:

```bash
python3 ml_pipeline/export_evenet_qi_inputs.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --prediction-parquet \
    "$PRED_DIR/mc/*__evenet_pred.parquet" \
    "$PRED_DIR/data/*__evenet_pred.parquet" \
  --base-dir "$QI_DIR" \
  --methods evenet \
  --num-workers 4 \
  --batch-size 50000
```

The exporter expands quoted glob patterns internally.

Use `--mc-split-fraction` only when the prediction step did not already apply
the split correction. Never apply both `--converted-split-fraction` and
`--mc-split-fraction` to the same MC prediction files.

Outputs:

- `$QI_DIR/evenet/processed/<sample>/filtered___raw.parquet`
- `$QI_DIR/evenet/processed/<sample>/filtered___<region>.parquet`
- `$QI_DIR/evenet/processed/<sample>/cutflow_<sample>.json`
- `$QI_DIR/evenet/config_evenet.yaml`
- `$QI_DIR/export_summary.json`

### 7. Run Central QI and Forward Folding

Run from the top-level central analysis environment:

```bash
python3 bin/tree_ana \
  -c "$QI_DIR/evenet/config_evenet.yaml"
```

The central workflow writes QI and response-matrix outputs under the `run`
directory configured by `config_evenet.yaml`.

## Common Validation Commands

### Monitor EveNet Inputs

```bash
python3 ml_pipeline/monitor_input.py \
  --data-dir "$INPUT_DIR/shards/data94" \
  --mc-dir \
    "$INPUT_DIR/shards/Ztautau" \
    "$INPUT_DIR/shards/Zll" \
    "$INPUT_DIR/shards/Zqq" \
  --config ml_pipeline/config/analysis.yaml \
  --output-dir "$CAMPAIGN_DIR/monitor-input" \
  --num-workers 4
```

### Compare Exported Channel Purity

```bash
python3 ml_pipeline/plot_channel_purity_side_by_side.py \
  --method EveNet:"$QI_DIR/evenet" \
  --baseline-xlsx data/baseline_yield.xlsx \
  --output "$CAMPAIGN_DIR/channel_purity_side_by_side.png"
```

Add `--unblind` only when data/MC panels should be shown.

### Summarize Calibration Magnitude

```bash
python3 ml_pipeline/extract_qi_calibration_magnitude.py \
  --method EveNet:"$QI_DIR/evenet" \
  --output-prefix "$CAMPAIGN_DIR/calibration_magnitude"
```

### Extract Final QI Measurements

```bash
python3 ml_pipeline/extract_qi_final_measurements.py \
  --method EveNet:"$QI_DIR/evenet/run/QI_analysis/results.txt" \
  --output-prefix "$CAMPAIGN_DIR/qi-results/evenet"
```

Compare multiple methods:

```bash
python3 ml_pipeline/extract_qi_final_measurements.py \
  --method Baseline:/path/to/baseline/results.txt \
  --method EveNet:"$QI_DIR/evenet/run/QI_analysis/results.txt" \
  --output-prefix "$CAMPAIGN_DIR/qi-results/baseline-vs-evenet"
```

### Summarize Response Matrices

```bash
python3 ml_pipeline/extract_response_matrix_summary.py \
  --method EveNet:"$QI_DIR/evenet/run/ForwardFoldingProcessor/response_matrices" \
  --output-prefix "$CAMPAIGN_DIR/response-matrix/evenet"
```

### Compare Diffusion Predictions Directly

Compare two or more prediction campaigns before QI reconstruction:

```bash
python3 ml_pipeline/compare_evenet_predictions.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --comparison-config ml_pipeline/config/prediction_comparison.yaml \
  --method Baseline="$BASELINE_PREDICTION_DIR" \
  --method DGPO="$DGPO_PREDICTION_DIR" \
  --output-dir "$CAMPAIGN_DIR/prediction-comparison"
```

The comparison uses each method's own truth target and therefore does not
require row alignment between methods. It reads both nested `x_invisible` and
EveNet-preprocessed flattened columns such as `x_invisible:0:0`. If both forms
are absent, the
default `target_source: auto` reconstructs the direction offsets from the
visible and target-missing three-vectors stored in the prediction parquet. It
automatically skips data parquets that have no truth target. It writes:

- normalized 1D target and prediction distributions;
- a target-versus-prediction 2D map for every method, leg, and diffusion feature;
- residual distributions and target-binned bias/resolution profiles;
- target, prediction, and prediction-minus-target density maps in the configured
  two-feature plane (by default delta-theta versus delta-phi);
- `comparison_metrics.json` and `comparison_metrics.csv` with bias, resolution,
  MAE, RMSE, Pearson correlation, and Jensen-Shannon divergence.

These plots validate the diffusion target space. In the current analysis, the
Invisible features are direction offsets relative to the visible tau candidate;
they are not absolute generator-level neutrino four-vectors. Binning, feature
selection, weights, and the per-method event cap are controlled in
`ml_pipeline/config/prediction_comparison.yaml`. Pass `--max-events 0` to use all
events.

## Troubleshooting

### Nested Submodule Is Missing

```bash
git submodule sync --recursive ml_pipeline
git submodule update --init --recursive ml_pipeline
git submodule status --recursive ml_pipeline
```

### `ModuleNotFoundError: evenet`

Verify that the nested Core repository exists:

```bash
test -f ml_pipeline/EveNet-Full/evenet/__init__.py
```

Then reinstall EveNet and set `PYTHONPATH`:

```bash
python3 -m pip install -e ml_pipeline/EveNet-Full
export PYTHONPATH="$PWD/ml_pipeline:$PWD/ml_pipeline/EveNet-Full:$PYTHONPATH"
```

### `evenet-train: command not found`

```bash
python3 -m pip install -e ml_pipeline/EveNet-Full
```

Confirm that the active environment is the expected one:

```bash
command -v python3
command -v evenet-train
```

### Included Config File Does Not Exist

Inspect every `default:` entry in the selected training config:

```bash
grep -n "default:" ml_pipeline/config/train_pretrain.yaml
```

Replace stale absolute paths with valid absolute paths or paths relative to
`ml_pipeline/config`.

### Class Labels Do Not Match

Regenerate the event information:

```bash
python3 ml_pipeline/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline/config/generated_event_info.yaml
```

Then rebuild input shards and rerun preprocessing.

### Prediction Weights Are Too Small or Too Large

Apply the MC split correction exactly once:

- prediction: `--converted-split-fraction 0.5`
- export: `--mc-split-fraction 0.5`

Do not apply either option to data.

### Export Cannot Find Raw Complement Events

Check that every sample in `ml_pipeline/config/analysis.yaml` has valid
`raw_files`. Export uses these files to rebuild the raw complement outside the
selected baseline rows.

## Updating the ML Repositories

Normal users should use the commits recorded by the parent repositories.
Maintainers updating a nested repository must commit and push from the inside
out:

1. commit and push `ml_pipeline/EveNet-Full/evenet`
2. update, commit, and push `ml_pipeline/EveNet-Full`
3. update, commit, and push `ml_pipeline`
4. update and commit the `ml_pipeline` pointer in `lep_tree_ana`

Each parent repository stores the exact commit of its child submodule.
