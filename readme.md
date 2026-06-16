# Waterbirds Preprocessing & Training Pipeline

This pipeline reproduces the Waterbirds bias-amplification experiments across multiple values of ρ (training set bias). Run the two scripts in order from the root of this repo.

## Step 1: Preprocessing

```bash
python waterbirds_preprocessing.py 
```

This single script runs the full preprocessing pipeline:

1. **Download** — fetches and extracts the Waterbirds dataset if not already present
2. **Generate per-ρ metadata CSVs** — creates resampled training splits at ρ = 0.70, 0.80, 0.95 (edit `RHO_VALUES` at the top of the script to change this)
3. **Copy images** — copies the relevant images into `data/waterbirds_rho_images/waterbirds_{rho}/`
4. **Write dataset.json** — generates label files required by `dataset_tool.py`
5. **Run dataset_tool.py** — converts each image folder into the `.zip`/folder format StyleGAN2 expects, output to `data/waterbirds_rho_gan/waterbirds_256_{rho}/`

Each step is idempotent: if its output already exists, it's skipped on rerun. To force a step to rerun, delete its output first or use `--skip_step` to skip steps you don't want to repeat:

```bash
python waterbirds_preprocessing.py --skip_step 0 1
```

## Step 2: Training

```bash
python train_waterbirds.py 
```

**By default this runs in dry-run mode** (`--dry-run=1`), meaning no actual training occurs — it validates the command and data path without launching a real run. To launch real training, pass `--dry-run 0`:

```bash
python train_waterbirds.py --dry-run 0
```

This trains StyleGAN2 once per $\rho$ value, reading from the folders produced in Step 1 (`data/waterbirds_rho_gan/waterbirds_256_{rho}/`) and writing checkpoints to `output_waterbirds_{rho}/` in the repo root.

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
python train_waterbirds.py --batch=64 --kimg=5000
```

## Output Layout

```
data/
├── waterbirds/                       # raw downloaded dataset
├── waterbirds_rho/                   # held-out test split metadata
├── waterbirds_rho_csv/               # per-rho metadata CSVs
├── waterbirds_rho_images/            # per-rho copied images + dataset.json
└── waterbirds_rho_gan/               # StyleGAN2-formatted datasets (output of dataset_tool.py)

output_waterbirds_{rho}/                    # training checkpoints and logs, one folder per rho
```
