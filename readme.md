# Waterbirds Preprocessing & Training Pipeline

This pipeline reproduces the Waterbirds bias-amplification experiments across multiple values of ρ (training set bias). Run the two scripts in order from the root of this repo.

## Prerequisites

- Python environment with `pandas`, `numpy`, `requests`, `tqdm`, `tabulate` installed
- A cloned copy of the StyleGAN2-based GAN repo (e.g. `Diffusion-GAN/diffusion-stylegan2`) containing `dataset_tool.py` and `train.py`
- No manual dataset download needed — Waterbirds downloads automatically on first run (~1GB)

## Step 1: Preprocessing

```bash
python waterbirds_preprocessing.py --dataset_tool_path /path/to/dataset_tool.py
```

This single script runs the full preprocessing pipeline:

1. **Download** — fetches and extracts the Waterbirds dataset if not already present
2. **Generate per-ρ metadata CSVs** — creates resampled training splits at ρ = 0.70, 0.80, 0.95 (edit `RHO_VALUES` at the top of the script to change this)
3. **Copy images** — copies the relevant images into `data/waterbirds_rho_images/waterbirds_{rho}/`
4. **Write dataset.json** — generates label files required by `dataset_tool.py`
5. **Run dataset_tool.py** — converts each image folder into the `.zip`/folder format StyleGAN2 expects, output to `data/waterbirds_rho_gan/bffhq_256_{rho}_balanced/`

> **Naming note**: despite the `bffhq_` prefix, this folder contains Waterbirds data. The naming follows the convention used by the downstream training command; if this is confusing for collaborators, consider renaming to something like `waterbirds_256_{rho}_balanced` in both `preprocess_waterbirds.py` and `train_waterbirds.py`.

Each step is idempotent: if its output already exists, it's skipped on rerun. To force a step to rerun, delete its output first or use `--skip_step` to skip steps you don't want to repeat:

```bash
python waterbirds_preprocessing.py --dataset_tool_path /path/to/dataset_tool.py --skip_step 0 1
```

### Verifying dataset_tool.py options

This repo's `dataset_tool.py` may differ from upstream StyleGAN2-ADA. Before running, check the available flags:

```bash
python /path/to/dataset_tool.py --help
```

If the resolution flag differs from `--width`/`--height`, update `step4_run_dataset_tool` in `waterbirds_preprocessing.py` accordingly.

## Step 2: Training

```bash
python train_waterbirds.py --train_script_path /path/to/train.py
```

**By default this runs in dry-run mode** (`--dry-run=1`), meaning no actual training occurs — it validates the command and data path without launching a real run. To launch real training, pass `--dry-run 0`:

```bash
python train_waterbirds.py --train_script_path /path/to/train.py --dry-run 0
```

This trains StyleGAN2 once per ρ value, reading from the folders produced in Step 1 (`data/waterbirds_rho_gan/bffhq_256_{rho}_balanced/`) and writing checkpoints to `output_runs_{rho}/` in the repo root.

Default hyperparameters (override via flags):

| Flag | Default | Meaning |
|---|---|---|
| `--gpus` | 1 | number of GPUs |
| `--batch` | 32 | batch size |
| `--gamma` | 8 | R1 regularization weight |
| `--kimg` | 3500 | training length (thousands of images) |
| `--cond` | 1 | class-conditional training |
| `--mirror` | 1 | horizontal flip augmentation |
| `--aug` | ada | adaptive discriminator augmentation |

Example overriding batch size and training length:

```bash
python train_waterbirds.py --train_script_path /path/to/train.py --batch=64 --kimg=5000
```

**Note for ProjectedGAN:** the `--cfg=stylegan2` flag is hardcoded in `train_waterbirds.py` to preserve W-space truncation semantics. If training ProjectedGAN, this is required and already set — but if you adapt this script for other architectures, check whether their default `--cfg` needs to be overridden too.

## Output Layout

```
data/
├── waterbirds/                       # raw downloaded dataset
├── waterbirds_rho/                   # held-out test split metadata
├── waterbirds_rho_csv/               # per-rho metadata CSVs
├── waterbirds_rho_images/            # per-rho copied images + dataset.json
└── waterbirds_rho_gan/               # StyleGAN2-formatted datasets (output of dataset_tool.py)

output_runs_{rho}/                    # training checkpoints and logs, one folder per rho
```

## Troubleshooting

- **FileNotFoundError on metadata.csv**: the dataset hasn't downloaded yet, or `IMAGE_ROOT` in `waterbirds_preprocessing.py` doesn't match where it landed. Run Step 1 from scratch.
- **Image counts don't match a previous run**: dataset size depends on `RHO_VALUES` and the rescaling logic in `step1_generate_csvs`. If you need to exactly reproduce an older training run, confirm the CSV row count matches before proceeding to Step 2 — don't assume two runs with the same ρ values produce the same dataset size unless the underlying pool and rescaling code are also identical.
- **dataset_tool.py errors on `--resolution`**: this fork may use `--width`/`--height` instead. Run `--help` to check.