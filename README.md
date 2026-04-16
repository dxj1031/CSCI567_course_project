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
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project2/<PI>_<project_id>/envs/cs567-baseline

export PYTHONPATH=$PWD/src
export DATA_ROOT=/project2/<PI>_<project_id>/datasets/CCT20
export OUTPUT_ROOT=/scratch1/$USER/cs567_runs
python scripts/train_baseline.py --config configs/cross_location_resnet18.yaml --validate-only
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
- `configs/day_to_night_resnet18.yaml`
- `configs/night_to_day_resnet18.yaml`

The day/night configs already remove `squirrel` so the label space stays consistent across the 10-class cross-time experiments.

## 7. Notes

- Set `PYTHONPATH=$PROJECT_ROOT/src` before running Python entrypoints on CARC.
- Keep editing code locally in VSCode and sync through GitHub.
- Use CARC OnDemand only when you need remote IDE access or quick remote inspection.

