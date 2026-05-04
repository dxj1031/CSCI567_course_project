# CS567 CCT20 Baseline

This repository is organized for the workflow we agreed on:

- local machine + VSCode for editing
- GitHub for shared code
- USC CARC for Conda environments and GPU training
- Google Drive for the shared dataset source

The repo intentionally keeps code, configs, job scripts, and lightweight artifacts only. Dataset files should stay on Google Drive and on CARC storage, not in GitHub.

## Repository Layout

- `configs/`: experiment configs for baseline runs
- `scripts/`: setup, sync, submit, and training entrypoints
- `slurm/`: CARC job payload scripts
- `src/cs567_cct20/`: reusable training code
- `data_process/`: local handoff notebook and notation docs

## 1. Clone To CARC

Use a shared project directory on CARC:

```bash
module purge
module load git
cd /project2/<PI>_<project_id>
git clone <your-github-shared-repo-url> cs567-cct20
cd cs567-cct20
```

## 2. Build The CARC Conda Environment

Create the environment under the shared project directory:

```bash
cd /project2/<PI>_<project_id>/cs567-cct20
bash scripts/carc_setup_env.sh /project2/<PI>_<project_id>/envs/cs567-baseline
```

The setup script installs:

- Python 3.11
- CUDA-enabled `torch`, `torchvision`, `torchaudio`
- `pandas`, `scikit-learn`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`

## 3. Configure Google Drive Access

Follow USC CARC's `rclone` guide to configure the Google Drive remote on your local machine, then copy `rclone.conf` to CARC.

Once `rclone` works on CARC, pull the dataset into a shared project location:

```bash
export DRIVE_REMOTE=mydrive
export DRIVE_PATH=shared/CS567/CCT20
export DEST_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
bash scripts/carc_sync_data.sh
```

This uses `rclone copy`, not `sync`, so it will not delete CARC-side files by accident.

## 4. Launch An Interactive GPU Session

Use this before the first full run to smoke test the environment and data paths:

```bash
export ACCOUNT=<project_id>
bash scripts/carc_interactive_gpu.sh
```

Inside the allocation:

```bash
cd /project2/<PI>_<project_id>/cs567-cct20
module purge
module load conda

export PYTHONPATH=$PWD/src
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
/project2/<PI>_<project_id>/envs/cs567-baseline/bin/python scripts/train_baseline.py --config configs/cross_location_resnet18.yaml --validate-only
```

## 5. Submit GPU Jobs

Smoke run:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
bash scripts/submit_smoke.sh configs/cross_location_resnet18.yaml
```

Full baseline run:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
bash scripts/submit_train.sh configs/cross_location_resnet18.yaml
```

Outputs land in `/scratch1/$USER/cs567_runs/<experiment>_<timestamp>/`. Copy the final metrics, plots, and checkpoints you want to keep into a shared artifacts directory under `/project2/<PI>_<project_id>/cs567-cct20/artifacts/`.

## 6. Current Baseline Configs

- `configs/cross_location_resnet18.yaml`
- `configs/cross_location_resnet34.yaml`
- `configs/day_to_night_resnet18.yaml`
- `configs/day_to_night_resnet34.yaml`
- `configs/night_to_day_resnet18.yaml`
- `configs/night_to_day_resnet34.yaml`
- `configs/cross_location_resnet50.yaml`
- `configs/day_to_night_resnet50.yaml`
- `configs/night_to_day_resnet50.yaml`
- `configs/cross_location_resnet101.yaml`
- `configs/day_to_night_resnet101.yaml`
- `configs/night_to_day_resnet101.yaml`

The day/night configs already remove `squirrel` so the label space stays consistent across the 10-class cross-time experiments.

## 7. Capacity Follow-Up

To test the hypothesis that generalization failure is partly caused by insufficient model capacity, rerun the same three experiment scenarios with deeper backbones while keeping the same training recipe:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash scripts/submit_train.sh configs/cross_location_resnet34.yaml
bash scripts/submit_train.sh configs/day_to_night_resnet34.yaml
bash scripts/submit_train.sh configs/night_to_day_resnet34.yaml

bash scripts/submit_train.sh configs/cross_location_resnet50.yaml
bash scripts/submit_train.sh configs/day_to_night_resnet50.yaml
bash scripts/submit_train.sh configs/night_to_day_resnet50.yaml

bash scripts/submit_train.sh configs/cross_location_resnet101.yaml
bash scripts/submit_train.sh configs/day_to_night_resnet101.yaml
bash scripts/submit_train.sh configs/night_to_day_resnet101.yaml
```

The code now supports `resnet18`, `resnet34`, `resnet50`, and `resnet101`, so no further training-code changes are needed for this follow-up. The configs only switch the model backbone while keeping the baseline training recipe fixed.

## 8. Compare Capacity Trends

Once the backbone runs have finished, aggregate the summaries into comparison tables:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

$ENV_PREFIX/bin/python scripts/compare_capacity.py \
  --results-root "$OUTPUT_ROOT" \
  --output-dir "$PROJECT_ROOT/artifacts/capacity_comparison"
```

This writes:

- `capacity_runs.csv`: one row per experiment run
- `capacity_split_metrics.csv`: one row per split
- `capacity_deltas.csv`: paired `resnet50 - resnet18` deltas for backward-compatible reporting
- `capacity_drop_comparison.csv`: paired `resnet50 - resnet18` generalization-drop deltas
- `capacity_trend.csv`: per-scenario backbone trend table across depths
- `capacity_trend_summary.csv`: scenario-level trend direction summary
- `capacity_comparison.md`: a markdown summary you can share with teammates or reuse in the report

When interpreting the results, compare whether increasing depth shrinks the same-domain-to-shifted-domain gap. If deeper models improve both in-domain and shifted-domain metrics but the shifted-domain drop remains large or grows with depth, then capacity alone is unlikely to fully explain the generalization failure.

## 9. Plot Capacity Results

To generate more intuitive figures for the report, create plots directly from the comparison CSVs:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline

$ENV_PREFIX/bin/python scripts/plot_capacity_results.py \
  --comparison-dir "$PROJECT_ROOT/artifacts/capacity_comparison" \
  --output-dir "$PROJECT_ROOT/artifacts/capacity_plots"
```

This writes:

- `capacity_trend_lines.png`: OOD accuracy and normalized gap (`gap / in-domain accuracy`) as a function of model depth
- `capacity_tradeoff_scatter.png`: trade-off view of OOD accuracy vs normalized gap
- `capacity_in_out_bar_grid.png`: per-scenario in-domain vs out-of-domain accuracy bars across backbones

## 10. Train-Time Domain Generalization Interventions

The active intervention suite now uses distribution diversification rather than information suppression. The goal is to expose the model to more train-time appearance diversity while preserving object structure. The validation and test domains are always read from the original CCT20 images and are never perturbed, aligned, blurred, or used to fit transformation statistics.

Supported `training.train_intervention` values:

- `none`: original training images
- `photometric_randomization`: for training images only, randomly perturb brightness, contrast, gamma, saturation/color balance, and mild RGB noise
- `background_perturbation`: for training images only, preserve scaled annotation bbox regions and apply light blur/noise/contrast perturbation outside the bbox
- `combined`: apply `background_perturbation` followed by `photometric_randomization`

Legacy `bbox_blur` and `brightness_aligned` configs remain in the repo for provenance, but `experiments/submit_train_time_interventions.sh` no longer submits them.

The same architecture, optimizer, epochs, seed, splits, and evaluation code are used across variants. The experiment grid covers all combinations of:

- backbones: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- scenarios: `cross_location`, `day_to_night`, `night_to_day`
- variants: `original`, `photometric_randomization`, `background_perturbation`, `combined`

Install the baseline dependencies into the CARC environment if needed:

```bash
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
$ENV_PREFIX/bin/python -m pip install -r requirements-baseline.txt
```

Then submit the full train-time intervention suite:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash experiments/submit_train_time_interventions.sh
```

Before submitting long jobs, run the validate-only matrix in an interactive allocation to verify paths, split filters, interventions, and one forward pass for every config:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_validate_only

bash experiments/validate_train_time_intervention_matrix.sh
```

Every run writes `dataset_summary.json` with `train_intervention`, `split_interventions`, and sanity checks confirming validation/test splits use no intervention.

## 11. Aggregate And Plot Intervention Results

After the diversification runs finish, aggregate the train-time intervention results:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash experiments/build_train_time_intervention_report.sh
```

By default this writes artifacts to a new timestamped directory under `$PROJECT_ROOT/artifacts/`, for example `$PROJECT_ROOT/artifacts/train_time_diversification_20260503_121500`. Set `ARTIFACT_ROOT` only when you intentionally want a specific new output directory.

- `intervention_runs.csv`
- `intervention_split_metrics.csv`
- `intervention_metrics.csv` with tidy columns including `backbone`, `scenario`, `variant`, `in_domain_acc`, `ood_acc`, `gap`, `normalized_gap`
- `intervention_effect_metrics.csv`
- `experiment_matrix.csv`
- `missing_runs.csv`
- `RUN_MANIFEST.md`
- `intervention_comparison.md`
- `intervention_summary_zh.md`
- `sanity_checks.json`
- `intervention_ood_accuracy_by_variant.png`
- `intervention_normalized_gap_by_variant.png`
- `intervention_tradeoff_scatter.png`
- `intervention_delta_bar_grid.png`
- `intervention_delta_ood_accuracy_vs_original.png`
- `intervention_delta_normalized_gap_vs_original.png`
- `intervention_backbone_lines_by_scenario.png`
- `intervention_backbone_comparison.png`
- `intervention_all_backbones_variants_grid.png`

The scatter plot uses:

- X-axis: normalized gap (`gap / in-domain accuracy`)
- Y-axis: out-of-domain accuracy
- Color: train-time intervention variant
- Marker/size: scenario and backbone

## 12. Visibility Hypothesis Experiments

The visibility suite tests whether low-light or low-visibility inputs explain the remaining generalization failures better than background or brightness mismatch alone.

Supported visibility flags on `scripts/train_baseline.py`:

- `--visibility-mode {original,gamma,clahe,gamma_clahe}`
- `--visibility-scope {test_only,train_test_consistent,night_only}`
- `--night-only-flag-source <metadata_field_or_rule>`; default: `day_night`
- `--visibility-gamma <float>`; default: `0.7`
- `--visibility-clahe-clip-limit <float>`; default: `2.0`
- `--visibility-clahe-tile-grid-size <WxH>`; default: `8,8`
- `--eval-only-checkpoint <path>` or `--checkpoint-results-root <dir>` for fixed-checkpoint test-time enhancement

Experiment scope semantics:

- `test_only`: load a fixed baseline checkpoint and enhance only evaluation rows identified as night or low-light. Training is not rerun.
- `train_test_consistent`: retrain from scratch and apply the selected visibility mode to train, validation, and test images.
- `night_only`: retrain from scratch and apply the selected visibility mode only to night or low-light rows across train, validation, and test.

Submit the three requested experiment groups on CARC:

```bash
export ACCOUNT=vsharan_1861
export PROJECT_ROOT=/home1/xdeng713/CSCI567_course_project
export ENV_PREFIX=/home1/xdeng713/envs/cs567-baseline
export DATA_ROOT=/scratch1/$USER/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

# Experiment 1: fixed-checkpoint test-time enhancement only.
# BASELINE_RESULTS_ROOT should contain the original baseline checkpoint run dirs.
export BASELINE_RESULTS_ROOT="$OUTPUT_ROOT"
bash "$PROJECT_ROOT/experiments/submit_visibility_test_only.sh"

# Experiment 2: retrain with consistent train/eval enhancement.
bash "$PROJECT_ROOT/experiments/submit_visibility_train_test_consistent.sh"

# Experiment 3: retrain with night-only enhancement.
bash "$PROJECT_ROOT/experiments/submit_visibility_night_only.sh"
```

To submit all three groups in one shell:

```bash
bash "$PROJECT_ROOT/experiments/submit_visibility_experiments.sh"
```

Before long submissions, run validate-only checks in an interactive allocation:

```bash
export OUTPUT_ROOT=/scratch1/$USER/cs567_visibility_validate_only
bash "$PROJECT_ROOT/experiments/validate_visibility_matrix.sh"
```

After the runs finish, aggregate and plot the results:

```bash
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
bash "$PROJECT_ROOT/experiments/build_visibility_report.sh"
```

The report writes a new artifact directory under `$PROJECT_ROOT/artifacts/visibility_hypothesis_<timestamp>` containing:

- `visibility_summary.csv`: tidy in-domain/OOD/gap/normalized-gap table
- `visibility_runs.csv`
- `visibility_split_metrics.csv`
- `visibility_effect_metrics.csv`
- `visibility_split_effect_metrics.csv`
- `experiment_matrix.csv`
- `missing_runs.csv`
- `visibility_summary_zh.md`
- `RUN_MANIFEST.md`
- `sanity_checks.json`
- plots for OOD accuracy, normalized gap, deltas, trade-off scatter, scenario-specific summaries, and best-mode heatmap

Do not interpret improvements unless they appear in `visibility_summary.csv` and the paired delta files. Scenario-dependent or backbone-dependent effects should be reported as such.

## 13. Notes

- Set `PYTHONPATH=$PROJECT_ROOT/src` before running Python entrypoints on CARC.
- Prefer calling `$ENV_PREFIX/bin/python` directly on CARC so user-site packages do not leak in.
- Keep the original Google Drive dataset read-only.
- Train-time interventions must not preprocess validation/test data or fit statistics from validation/test domains.
- Keep editing code locally in VSCode and sync through GitHub.
- Use CARC OnDemand only when you need remote IDE access or quick remote inspection.
