# CCT-20 Data Processing Notation (Handoff Doc)

This document defines the exact data processing pipeline used by `cct20_data_prep.ipynb` in this project.

## 1. Purpose

The goal is to produce reproducible, image-level classification tables for:

- cross-location evaluation (`train` -> `cis`, `trans`)
- cross-time evaluation (`day` vs `night`)
- robust baseline training with consistent class space

## 2. Input Files

Expected directory:

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

Each annotation file is COCO-like with keys:

- `images`: image metadata rows
- `annotations`: object annotations (category + bbox)
- `categories`: category id/name mapping

## 3. Symbols

- `I`: set of image rows after split-file loading
- `A_i`: annotation set for image `i`
- `y_i`: final image-level class label for image `i`
- `s_i`: split label for image `i`, in `{train, val, cis, trans}`
- `t_i`: parsed timestamp for image `i`
- `h_i`: hour component from `t_i`
- `d_i`: day/night label in `{day, night, unknown}`
- `q_i`: sequence id (`seq_id`) for image `i`

## 4. Split Construction

Default split assignment:

- `train_annotations.json` -> `train`
- `cis_val_annotations.json` -> `val`
- `trans_val_annotations.json` -> `val` (if `MERGE_TRANS_VAL_INTO_VAL=True`)
- `cis_test_annotations.json` -> `cis`
- `trans_test_annotations.json` -> `trans`

If `MERGE_TRANS_VAL_INTO_VAL=False`, `trans_val_annotations.json` is assigned to `trans`.

## 5. Image-Level Label Collapse

Because one image may contain multiple annotation rows, we collapse annotation-level labels into one primary label per image.

For each image `i`, choose `y_i` using:

1. Prefer non-`empty` category over `empty`.
2. If still multiple candidates, prefer larger bbox area (`w * h`).
3. If still tied, take first row after sorting.

Additional tracking columns:

- `num_annotations`: count of annotation rows in `A_i`
- `num_unique_categories`: number of unique categories in `A_i`
- `has_multi_category`: whether `num_unique_categories > 1`

## 6. Timestamp and Day/Night

Timestamp source:

- use `datetime` if present
- otherwise fallback to `date_captured`

Then parse into `datetime_parsed` and minute-of-day `m_i`.

Manual day/night ranges:

- Define `R_day` from `DAY_TIME_RANGES = [(start, end), ...]`
- Define `R_night` from `NIGHT_TIME_RANGES = [(start, end), ...]`
- Time format is `HH:MM` (24-hour clock)
- Wrap-around ranges are supported (example: `('22:00', '02:00')`)

Label rule:

- if `m_i ∈ R_day` and `m_i ∉ R_night`, then `d_i = day`
- if `m_i ∈ R_night` and `m_i ∉ R_day`, then `d_i = night`
- if `m_i ∈ R_day ∩ R_night`, then `d_i = OVERLAP_POLICY`
- if `m_i ∉ (R_day ∪ R_night)`, then `d_i = UNCOVERED_LABEL`
- if timestamp parse fails, `d_i = unknown`

Important:

- `R_day` and `R_night` do not need to cover all 24 hours.
- Uncovered intervals are intentionally allowed for selective experiments.

## 7. Cleaning Rules

Config defaults:

- `REMOVE_EMPTY = True`
- `MIN_TRAIN_IMAGES = 50`
- `KEEP_ONE_FRAME_PER_SEQ = True`
- `TOP_K_CLASSES = None`
- `EXCLUDE_CLASSES = ['car']`

Rules:

1. If `REMOVE_EMPTY=True`, drop rows with `y_i='empty'`.
2. If `EXCLUDE_CLASSES` is non-empty, drop rows with `y_i ∈ EXCLUDE_CLASSES`.
3. Compute train class counts:
   - `N_c = |{i : s_i=train and y_i=c}|`
4. Keep class `c` if `N_c >= MIN_TRAIN_IMAGES`.
5. Apply kept class set to all splits (consistent class space).
6. If `TOP_K_CLASSES` is not `None`, keep only top-K among already kept train classes.

## 8. Sequence Deduplication

If `KEEP_ONE_FRAME_PER_SEQ=True`:

- group by `(s_i, q_i)`
- keep earliest row by `frame_num` (fallback to sorted order if missing)

Rationale: remove near-duplicate burst frames from camera trap sequences.

## 9. Output Contract

Outputs under `CCT20/processed/`:

- `cct20_clean_all.csv`: cleaned full dataset
- `cct20_train.csv`: training set
- `cct20_val.csv`: validation set
- `cct20_cis.csv`: same-location test set
- `cct20_trans.csv`: cross-location test set
- `cct20_train_day.csv`: daytime subset of the training set, defined by `DAY_TIME_RANGES`
- `cct20_train_night.csv`: nighttime subset of the training set, defined by `NIGHT_TIME_RANGES`
- `label_mapping.json`: mapping from class names to numeric labels
- `data_processing_spec.json`: preprocessing configuration, input metadata, and output metadata

Per-dataset class statistics:

- The notebook provides an analysis cell that prints class counts for every exported dataset listed above.
- It also outputs a wide summary table where rows are classes and columns are dataset files.

## 10. Required Columns in Processed CSV

Expected core columns used downstream:

- identity: `image_id`, `file_name`
- label: `category_name`, `label`
- split/domain: `split`, `day_night`
- time: `datetime_raw`, `datetime_parsed`, `hour`
- camera/sequence: `location`, `seq_id`, `seq_num_frames`, `frame_num`
- quality/meta: `num_annotations`, `num_unique_categories`, `has_multi_category`, `is_empty`, `source_file`

## 11. Reproducibility Checklist

When sharing with collaborators, provide:

1. this document
2. `cct20_data_prep.ipynb`
3. `CCT20/processed/data_processing_spec.json`
4. `CCT20/processed/label_mapping.json`
5. exact git commit hash used to generate processed CSV files

## 12. Known Caveats

- A small number of images can contain multiple categories; single-label collapse may introduce noise.
- Day/night labels are defined by manual time ranges; they are not astronomy-based sunrise/sunset.
- If `REMOVE_EMPTY=True`, negative/background-only examples are removed from training tables.
