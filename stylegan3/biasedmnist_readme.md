# BiasedMNIST Preprocessing & Training Pipeline

This pipeline reproduces the BiasedMNIST bias-amplification experiments across multiple values of $\rho$ (label–bias correlation). Run the two scripts in order from the root of this repository.

## Step 1: Preprocessing

```bash
python biasedmnist_preprocessing.py
```

This script runs the full preprocessing pipeline:

1. **Generate BiasedMNIST datasets** — creates BiasedMNIST datasets with $\rho$ = 0.70, 0.80, 0.95
2. **Export images and metadata** — saves all generated images and writes a metadata CSV for each $\rho$
3. **Write dataset.json** — generates the label files required by `dataset_tool.py`
4. **Run dataset_tool.py** — converts each image folder into the format expected by StyleGAN2, output to `data/biasedmnist_gan/biasedmnist_{rho}/`

The generated metadata CSV contains:

```text
img_id,img_filename,y,bias,rho
```

where:

* `img_id` is the sample index
* `img_filename` is the image filename
* `y` is the digit class
* `bias` is the background colour class
* `rho` is the target bias level used to generate the dataset

To skip the GAN dataset conversion step:

```bash
python biasedmnist_preprocessing.py --skip_gan
```

## Verifying the Bias Level

The effective bias level can be verified directly from the generated CSV:

```python
import pandas as pd

df = pd.read_csv("biasedmnist_0.70.csv")

empirical_rho = (df["y"] == df["bias"]).mean()

print(f"Empirical rho: {empirical_rho:.4f}")
```

Since aligned samples satisfy:

```text
y == bias
```

the empirical rho should be close to the requested value (0.70, 0.80, or 0.95).

## Step 2: Training

Training is executed one $\rho$ value at a time using a shell wrapper. Each run trains a StyleGAN2 model on a single dataset corresponding to one bias level.

```bash
bash train.sh <rho_tag>
```

For example:

```bash
bash train.sh 95
```

By default this runs in dry-run mode (`--dry-run`), meaning no actual training occurs. The command is validated and the dataset path is checked without launching a real training run.

To launch a real training run, remove the `--dry-run` flag from `train.sh` or pass the corresponding option supported by your training script.

## Dataset Requirements

The script expects preprocessing outputs from Step 1:

```bash
data/biasedmnist_gan/biasedmnist_{rho_tag}
```

For example:

```bash
data/biasedmnist_gan/biasedmnist_95
```

If the folder does not exist, run preprocessing first:

```bash
python biasedmnist_preprocessing.py
```

## Default Hyperparameters

| Flag       | Default | Meaning                               |
| ---------- | ------- | ------------------------------------- |
| `--gpus`   | 1       | Number of GPUs                        |
| `--batch`  | 32      | Batch size                            |
| `--kimg`   | 3500    | Training length (thousands of images) |
| `--cond`   | 1       | Class-conditional training            |
| `--mirror` | 1       | Horizontal flip augmentation          |
| `--gamma`  | 8       | R1 regularization weight              |
| `--aug`    | ada     | Adaptive augmentation                 |

## Optional Overrides

Additional arguments are forwarded directly to the underlying training script, allowing hyperparameter overrides without modifying `train.sh`.

For example:

```bash
bash biasedmnist_train.sh 95 --gpus=2 --batch=64 --kimg=5000
```

This overrides:

* number of GPUs
* batch size
* training length

## Output Layout

```text
data/
├── mnist/                              # raw MNIST dataset
├── biasedmnist_images/                 # generated images + dataset.json
│   ├── rho_0.70/
│   ├── rho_0.80/
│   └── rho_0.95/
├── biasedmnist_metadata/               # metadata CSV files
│   ├── biasedmnist_0.70.csv
│   ├── biasedmnist_0.80.csv
│   └── biasedmnist_0.95.csv
└── biasedmnist_gan/                    # StyleGAN2-formatted datasets
    ├── biasedmnist_70/
    ├── biasedmnist_80/
    └── biasedmnist_95/

output_biasedmnist_70/
output_biasedmnist_80/
output_biasedmnist_95/
```
