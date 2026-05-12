#  CSCI 567 final project

This repository is organized for the workflow we agreed on:

- local machine + VSCode for editing
- GitHub for shared code
- USC CARC for Conda environments and GPU training
- Official public CCT-20 dataset release for the raw dataset source

The repo intentionally keeps code, configs, job scripts, and lightweight artifacts only. 
Raw CCT-20 dataset files should be downloaded from the official public benchmark release and stored locally or on CARC storage, not in GitHub.

## Repository Layout

- `src/`: reusable Python package for CCT20 training, interventions, visibility, and image ablations
- `configs/`: active expexriment configs; legacy blur/brightness configs live in `configs/legacy/`
- `scripts/`: direct command-line entrypoints for training, submission helpers, aggregation, and plotting
- `jobs/`: CARC batch jobs and experiment matrices
- `data/`: dataset preparation notebook, notes, and lightweight preprocessing utilities
- `examples/`: small tracked visual/debug samples only
- `results/`: curated report-ready CSV, figure, and markdown outputs
- `outputs/`: ignored local outputs, paper assets, CARC result copies, and caches

## Dataset Source

This project uses the CCT-20 benchmark subset of the Caltech Camera Traps dataset. The benchmark subset was introduced in Beery et al. 2018, and the benchmark images were downsized to a maximum of 1024 pixels on one side.

The raw image files and official annotation metadata are not stored in this GitHub repository. Download them from the official public CCT-20 benchmark release:

- Benchmark images, about 6GB: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz
- Metadata files for train/val/cis/trans splits, about 3MB: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz

After downloading and extracting the files, organize the dataset as:

```text
CCT20/
├── images/
└── annotations/
    ├── train_annotations.json
    ├── cis_val_annotations.json
    ├── trans_val_annotations.json
    ├── cis_test_annotations.json
    └── trans_test_annotations.json
```

## Data Preparation Contribution

This repository uses the official CCT-20 split annotation files and converts them into standardized image-level CSV tables before training. The data preparation work is tracked in:

- `data/cct20_data_prep.ipynb`: notebook that builds the processed split CSV files, label mapping, and preprocessing metadata.
- `data/CCT20_DATA_NOTATION.md`: handoff document that defines the split construction, label-collapse rule, day/night labeling rule, cleaning rules, output contract, and known caveats.

To run the data preparation notebook locally:

```bash
export DATA_ROOT=/path/to/CCT20
jupyter notebook data/cct20_data_prep.ipynb
```

The generated CSV and JSON metadata files are written to `$DATA_ROOT/processed/`.

## Intervention Coverage

The repo contains code and CARC job coverage for the 11 non-original intervention conditions used in the report story:

| Group | Conditions | Primary job | Validation job |
|---|---|---|---|
| Legacy appearance/context probes | `bbox_blur`, `brightness_aligned` | `jobs/legacy_alignment.sh` | `jobs/validate_legacy_alignment.sh` |
| Train-time diversification | `photometric_randomization`, `background_perturbation`, `combined` | `jobs/train_interventions.sh` | `jobs/validate_interventions.sh` |
| Visibility enhancement | `gamma`, `clahe`, `gamma_clahe` | `jobs/visibility_all.sh` | `jobs/validate_visibility.sh` |
| Object-centric diagnostics | `foreground_only`, `background_only`, `object_crop` | `jobs/object_diagnostics.sh` | `jobs/validate_followup.sh` |

To submit the full 11-condition matrix from one entrypoint:

```bash
bash jobs/interventions_all.sh
```

To run the validate-only checks for all intervention families:

```bash
bash jobs/validate_all_interventions.sh
```

## 1. Clone To CARC

Use a CARC project directory:

```bash
module purge
module load git
cd /project2/<PI>_<project_id>
git clone <your-github-shared-repo-url> cs567-cct20
cd cs567-cct20
```

## 2. Build The CARC Conda Environment

Create the environment under the CARC project directory:

```bash
cd /project2/<PI>_<project_id>/cs567-cct20
bash scripts/setup_env.sh /project2/<PI>_<project_id>/envs/cs567-baseline
```

The setup script installs:

- Python 3.11
- CUDA-enabled `torch`, `torchvision`, `torchaudio`
- `pandas`, `scikit-learn`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`

## 3. Prepare The CCT-20 Dataset

Download the official CCT-20 benchmark images and metadata from the links in the Dataset Source section.

On CARC, one possible setup is:

```bash
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
mkdir -p "$DATA_ROOT"

cd "$DATA_ROOT"

wget https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz
wget https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz

tar -xzf eccv_18_all_images_sm.tar.gz
tar -xzf eccv_18_annotations.tar.gz
```

After extraction, arrange the files so that the final dataset layout is:

```text
$DATA_ROOT/
├── images/
└── annotations/
    ├── train_annotations.json
    ├── cis_val_annotations.json
    ├── trans_val_annotations.json
    ├── cis_test_annotations.json
    └── trans_test_annotations.json
```

Then run the preprocessing notebook:

```bash
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
jupyter notebook data/cct20_data_prep.ipynb
```

The processed CSV and JSON metadata files will be written to:

```text
$DATA_ROOT/processed/
```

Do not commit raw images, official annotation files, or generated processed CSV files to GitHub.

## 4. Launch An Interactive GPU Session

Use this before the first full run to smoke test the environment and data paths:

```bash
export ACCOUNT=<project_id>
bash scripts/gpu_shell.sh
```

Inside the allocation:

```bash
cd /project2/<PI>_<project_id>/cs567-cct20
module purge
module load conda

export PYTHONPATH=$PWD/src
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
/project2/<PI>_<project_id>/envs/cs567-baseline/bin/python scripts/train.py --config configs/cross_location_resnet18.yaml --validate-only
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

Outputs land in `/scratch1/$USER/cs567_runs/<experiment>_<timestamp>/`. Copy the final metrics and plots you want to keep into an artifact directory under `/project2/<PI>_<project_id>/cs567-cct20/outputs/artifacts/`. Large checkpoints should remain outside GitHub.

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
  --output-dir "$PROJECT_ROOT/outputs/artifacts/capacity_comparison"
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

$ENV_PREFIX/bin/python scripts/plot_capacity.py \
  --comparison-dir "$PROJECT_ROOT/outputs/artifacts/capacity_comparison" \
  --output-dir "$PROJECT_ROOT/outputs/artifacts/capacity_plots"
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

Legacy `bbox_blur` and `brightness_aligned` configs remain in `configs/legacy/` for provenance and can be submitted through `jobs/legacy_alignment.sh`. The active distribution-diversification matrix in `jobs/train_interventions.sh` keeps only train-time perturbations that preserve the original validation/test images.

The same architecture, optimizer, epochs, seed, splits, and evaluation code are used across variants. The experiment grid covers all combinations of:

- backbones: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- scenarios: `cross_location`, `day_to_night`, `night_to_day`
- variants: `original`, `photometric_randomization`, `background_perturbation`, `combined`

Install the baseline dependencies into the CARC environment if needed:

```bash
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
$ENV_PREFIX/bin/python -m pip install -r requirements.txt
```

Then submit the full train-time intervention suite:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash jobs/train_interventions.sh
```

Before submitting long jobs, run the validate-only matrix in an interactive allocation to verify paths, split filters, interventions, and one forward pass for every config:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_validate_only

bash jobs/validate_interventions.sh
```

Every run writes `dataset_summary.json` with `train_intervention`, `split_interventions`, and sanity checks confirming validation/test splits use no intervention.

## 11. Aggregate And Plot Intervention Results

After the diversification runs finish, aggregate the train-time intervention results:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash jobs/report_interventions.sh
```

By default this writes artifacts to a new timestamped directory under `$PROJECT_ROOT/outputs/artifacts/`, for example `$PROJECT_ROOT/outputs/artifacts/train_time_diversification_20260503_121500`. Set `ARTIFACT_ROOT` only when you intentionally want a specific new output directory.

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

Supported visibility flags on `scripts/train.py`:

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
bash "$PROJECT_ROOT/jobs/visibility_test.sh"

# Experiment 2: retrain with consistent train/eval enhancement.
bash "$PROJECT_ROOT/jobs/visibility_train_test.sh"

# Experiment 3: retrain with night-only enhancement.
bash "$PROJECT_ROOT/jobs/visibility_night.sh"
```

To submit all three groups in one shell:

```bash
bash "$PROJECT_ROOT/jobs/visibility_all.sh"
```

Before long submissions, run validate-only checks in an interactive allocation:

```bash
export OUTPUT_ROOT=/scratch1/$USER/cs567_visibility_validate_only
bash "$PROJECT_ROOT/jobs/validate_visibility.sh"
```

After the runs finish, aggregate and plot the results:

```bash
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
bash "$PROJECT_ROOT/jobs/report_visibility.sh"
```

The report writes a new artifact directory under `$PROJECT_ROOT/outputs/artifacts/visibility_hypothesis_<timestamp>` containing:

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

## 13. Follow-Up Experiments For The Final Analysis

The next experiment suite is designed for a careful negative-result story: quantify uncertainty, test whether rare classes drive the gap, and diagnose whether the model is using animal pixels or scene context.

### 13.1 Seed sweep for headline results

Rerun the scenario-winning backbone from the capacity study with multiple seeds:

- `cross_location -> resnet101`
- `day_to_night -> resnet101`
- `night_to_day -> resnet50`

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
export SEEDS="42 43 44"

bash jobs/seed_sweep.sh
```

Purpose: estimate run-to-run variation so small OOD changes are not overinterpreted as real effects.

### 13.2 Rare-class and class-balance interventions

Submit class-weight and sampling alternatives on the same scenario-winning backbones:

```bash
bash jobs/class_balance.sh
```

The compared methods are:

- `baseline_weighted`: current weighted cross-entropy behavior
- `no_class_weights`: unweighted cross-entropy
- `class_balanced_sampler`: inverse-frequency sampling with unweighted cross-entropy
- `focal_balanced`: focal loss with balanced class weights

Purpose: test whether classes such as `rodent`, `bird`, and `cat` fail mainly because of imbalance rather than because of the domain shift alone.

### 13.3 Object-centric diagnostics

Submit bbox-based image views on the same scenario-winning backbones:

```bash
bash jobs/object_diagnostics.sh
```

The compared image views are:

- `none`: original full image
- `object_crop`: crop to the union of annotation boxes
- `foreground_only`: preserve annotated animal boxes and replace background with gray
- `background_only`: replace annotated animal boxes with gray and preserve background

Purpose: distinguish animal-shape reliance from background/context reliance. A strong `background_only` result would be evidence of shortcut learning; a strong `foreground_only` or `object_crop` result would support object-centric robustness.

### 13.4 Validate and aggregate follow-ups

The validate script only builds the matrix and performs the existing validate-only forward checks:

```bash
bash jobs/validate_followup.sh
```

After jobs finish:

```bash
bash jobs/report_followup.sh
```

This writes:

- `followup_runs.csv`
- `followup_split_metrics.csv`
- `followup_generalization_metrics.csv`
- `followup_seed_summary.csv`
- `followup_per_class_f1.csv`
- `followup_comparison.md`

## 14. Notes

- Set `PYTHONPATH=$PROJECT_ROOT/src` before running Python entrypoints on CARC.
- Prefer calling `$ENV_PREFIX/bin/python` directly on CARC so user-site packages do not leak in.
- Keep the original downloaded CCT-20 dataset read-only.
- Train-time interventions must not preprocess validation/test data or fit statistics from validation/test domains.
- Keep editing code locally in VSCode and sync through GitHub.
- Use CARC OnDemand only when you need remote IDE access or quick remote inspection.
