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

## 10. ResNet50 Data Intervention Follow-Up

To test whether scenario-dependent generalization differences are driven by background reliance or illumination shift, keep the backbone fixed to `resnet50` and create two new dataset copies:

- `dataset_bbox_bg`: use the official annotation bounding boxes, keep the boxed animal region unchanged, and blur everything outside the boxes
- `dataset_histmatch`: compute train-only day/night brightness histograms on the HSV value channel and histogram-match each image toward the combined train target distribution

These scripts only read the original dataset and write new copies elsewhere. Before generating the new copies, delete the previous CARC-only intervention datasets:

```bash
export VARIANT_DATA_ROOT=/scratch1/$USER/cct20_variants
bash experiments/cleanup_old_resnet50_interventions.sh
```

Install the baseline dependencies into the CARC environment if needed:

```bash
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
$ENV_PREFIX/bin/python -m pip install -r requirements-baseline.txt
```

Then generate the new variants:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export SOURCE_DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export VARIANT_DATA_ROOT=/scratch1/$USER/cct20_variants

bash experiments/prepare_dataset_variants.sh
```

This creates:

- `$VARIANT_DATA_ROOT/dataset_bbox_bg`
- `$VARIANT_DATA_ROOT/dataset_histmatch`

Then submit the intervention suite:

```bash
export ACCOUNT=<project_id>
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export VARIANT_DATA_ROOT=/scratch1/$USER/cct20_variants
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash experiments/submit_resnet50_data_interventions.sh
```

The original ResNet50 baseline summaries are reused from the existing baseline/capacity runs:

- `configs/cross_location_resnet50.yaml`
- `configs/day_to_night_resnet50.yaml`
- `configs/night_to_day_resnet50.yaml`

The intervention configs are:

- `configs/cross_location_resnet50_bbox_bg.yaml`
- `configs/day_to_night_resnet50_bbox_bg.yaml`
- `configs/night_to_day_resnet50_bbox_bg.yaml`
- `configs/cross_location_resnet50_histmatch.yaml`
- `configs/day_to_night_resnet50_histmatch.yaml`
- `configs/night_to_day_resnet50_histmatch.yaml`

## 11. Aggregate And Plot Intervention Results

After the ResNet50 original/background/brightness runs finish, aggregate the intervention results:

```bash
export PROJECT_ROOT=/project2/<PI>_<project_id>/cs567-cct20
export ENV_PREFIX=/project2/<PI>_<project_id>/envs/cs567-baseline
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs

bash experiments/build_resnet50_intervention_report.sh
```

This writes artifacts to `$PROJECT_ROOT/artifacts/resnet50_interventions`:

- `intervention_runs.csv`
- `intervention_split_metrics.csv`
- `intervention_metrics.csv`
- `intervention_comparison.md`
- `intervention_tradeoff_scatter.png`
- `intervention_in_out_bar_grid.png`

The scatter plot uses:

- X-axis: normalized gap (`gap / in-domain accuracy`)
- Y-axis: out-of-domain accuracy
- Color: scenario
- Marker/text label: dataset variant (`Original`, `BBox Background`, `Histogram Match`)

## 12. Notes

- Set `PYTHONPATH=$PROJECT_ROOT/src` before running Python entrypoints on CARC.
- Prefer calling `$ENV_PREFIX/bin/python` directly on CARC so user-site packages do not leak in.
- Keep the original Google Drive dataset read-only. All intervention datasets should be written to CARC-only directories such as `/scratch1/$USER/cct20_variants`.
- Keep editing code locally in VSCode and sync through GitHub.
- Use CARC OnDemand only when you need remote IDE access or quick remote inspection.
