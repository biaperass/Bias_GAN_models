# BiasedMNIST Preprocessing & Training Pipeline

This pipeline reproduces the BiasedMNIST bias-amplification experiments across multiple values of $\rho$ (label–bias correlation). Run the two scripts in order from the root of this repository.

## Prerequisites

Before running the preprocessing or training pipeline, create and activate a Conda environment using one of the environment files provided in the repository root:

* `environment_cu12.yml` for systems with CUDA 12
* `environment_cu11.yml` for systems with CUDA 11

Choose the environment file that matches the CUDA version available on your machine or server.

## Step 1: Preprocessing

```bash
python biasedmnist_preprocessing.py
```

This script runs the full preprocessing pipeline:

1. **Generate BiasedMNIST datasets** — creates BiasedMNIST datasets with $\rho$ = 0.70, 0.80, 0.95
2. **Export images and metadata** — saves all generated images and writes a metadata CSV for each $\rho$
3. **Write dataset.json** — generates the label files required by `dataset_tool.py`
4. **Run dataset_tool.py** — converts each image folder into the format expected by the model, output to `data/biasedmnist_gan/biasedmnist_{rho}/`

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


## Step 2: Training

Training is executed one $\rho$ value at a time using a shell wrapper. Each run trains a StyleGAN2 model on a single dataset corresponding to one bias level.

```bash
bash biasedmnist_train.sh <rho_tag>
```

For example:

```bash
bash biasedmnist_train.sh 95
```

By default, the training script launches a real training run using the specified dataset and hyperparameters.

If you want to validate the configuration, verify the dataset path, and inspect the generated command without starting training, add the `--dry-run` flag when invoking the training script or enable it in `biasedmnist_train.sh`.

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
