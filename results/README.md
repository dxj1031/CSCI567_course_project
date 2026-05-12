# Curated Results

This directory stores lightweight, report-ready outputs that should be versioned with the code. Large raw run folders, checkpoints, caches, and generated scratch artifacts remain under ignored `outputs/` paths.

## Contents

- `research_assessment.md`: short audit of how the repository supports the final project story.
- `tables/`: final CSV tables used by the manuscript figures and result narrative.
- `figures/`: final PDF/PNG figures and the 11-condition sample intervention gallery.
- `followup/`: aggregated CARC follow-up results for seed, class-balance, and object-centric diagnostic experiments.

## 11-Condition Intervention Map

| Condition | Family | Code entrypoint | Job coverage |
|---|---|---|---|
| `bbox_blur` | Legacy context probe | `src/cs567_cct20/interventions.py` | `jobs/legacy_alignment.sh` |
| `brightness_aligned` | Legacy appearance probe | `src/cs567_cct20/interventions.py` | `jobs/legacy_alignment.sh` |
| `photometric_randomization` | Train-time diversification | `src/cs567_cct20/interventions.py` | `jobs/train_interventions.sh` |
| `background_perturbation` | Train-time diversification | `src/cs567_cct20/interventions.py` | `jobs/train_interventions.sh` |
| `combined` | Train-time diversification | `src/cs567_cct20/interventions.py` | `jobs/train_interventions.sh` |
| `gamma` | Visibility enhancement | `src/cs567_cct20/visibility.py` | `jobs/visibility_all.sh` |
| `clahe` | Visibility enhancement | `src/cs567_cct20/visibility.py` | `jobs/visibility_all.sh` |
| `gamma_clahe` | Visibility enhancement | `src/cs567_cct20/visibility.py` | `jobs/visibility_all.sh` |
| `foreground_only` | Object-centric diagnostic | `src/cs567_cct20/image_ablation.py` | `jobs/object_diagnostics.sh` |
| `background_only` | Object-centric diagnostic | `src/cs567_cct20/image_ablation.py` | `jobs/object_diagnostics.sh` |
| `object_crop` | Object-centric diagnostic | `src/cs567_cct20/image_ablation.py` | `jobs/object_diagnostics.sh` |

Use `jobs/interventions_all.sh` for the full submission path and `jobs/validate_all_interventions.sh` for validate-only checks.
