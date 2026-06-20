import os
import tarfile
from typing import Tuple, Union, Dict, List, Generator, Literal

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from tabulate import tabulate

Strategy = Literal["downsample", "oversample", "borrow_from_val", "borrow_from_val_fixed"]

class WaterbirdsDynamicRho(Dataset):
    """
    Waterbirds dataset with adjustable training set bias via different strategies.

    This class can be used in two ways:
    1. As a standard Dataset loader for a given split (train/val/test).
    2. To generate new, complete metadata files with a modified training set
       using the `create_and_save_metadata` static method.

    Args:
        env (str): The dataset split, one of ["train", "val", "test"].
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform to apply to the images.
        bias_amount (float): The desired correlation for the training set (e.g., 0.95, 0.70).
                             Only used when creating a new training set.
        strategy (Strategy): The resampling strategy for the training set.
        metadata_path (str): Path to the metadata CSV file to use.
        random_seed (int): The random seed for resampling the dataset.
    """
    DOWNLOAD_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    DATASET_NAME = "waterbirds"

    def __init__(self,
                 env: str,
                 root: str = "./data",
                 transform=None,
                 bias_amount: float = 0.95,
                 strategy: Strategy = "downsample",
                 metadata_path: str = None,
                 random_seed: int = 42):

        self.root: str = os.path.join(root, WaterbirdsDynamicRho.DATASET_NAME)
        self.env: str = env
        self.transform = transform
        self.num_classes = 2
        self.n_groups = 4
        self.env_to_split = {"train": 0, "val": 1, "test": 2}

        self.data_dir = os.path.join(self.root, "waterbird_complete95_forest2water2")
        if not os.path.exists(self.data_dir):
            self.__download_dataset()

        # If no specific metadata path is given, use the default one.
        if metadata_path is None:
            metadata_path = os.path.join(self.data_dir, "metadata.csv")

        if not os.path.exists(metadata_path):
             raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"You may need to generate it first using `create_and_save_metadata`."
            )

        # Load the specified metadata and filter for the environment (train/val/test)
        metadata_df = pd.read_csv(metadata_path)
        self.metadata = metadata_df.query(f"split == {self.env_to_split[self.env]}")

        self.samples = []
        for _, row in self.metadata.iterrows():
            self.samples.append({
                "image_path":  os.path.join(self.data_dir, row["img_filename"]),
                "class_label": int(row["y"]),
                "bias_label":  int(row["place"]), # here it is not used other than evaluation purposes
            })

    @staticmethod
    def create_and_save_metadata(
        output_path: str,
        strategy: Strategy,
        bias_amount: float,
        root: str = "./data",
        random_seed: int = 0,
        n_per_class: Dict[int, int] = None,
        extra_pool_df: pd.DataFrame = None
    ):
        """
        Generates a new metadata file with a resampled training split and saves it.
        This is a static method, so it can be called directly from the class.
        """
        print(f"Generating new metadata with bias={bias_amount}, strategy='{strategy}'...")
        
        # Instantiate the class to ensure data is downloaded
        base_data_dir = os.path.join(root, WaterbirdsDynamicRho.DATASET_NAME, "waterbird_complete95_forest2water2")
        if not os.path.exists(base_data_dir):
            WaterbirdsDynamicRho(env="test", root=root) # Triggers download if needed
            
        original_metadata_path = os.path.join(base_data_dir, "metadata.csv")
        original_df = pd.read_csv(original_metadata_path)

        # 1. Define the pool of data to sample from
        train_df = original_df.query("split == 0")
        sampling_pool = train_df
        if strategy == "borrow_from_val":
            print("=> Strategy: Borrowing from validation set.")
            val_df = original_df.query("split == 1")
            sampling_pool = pd.concat([train_df, val_df])
        
        elif strategy == "borrow_from_val_fixed":
            val_df = original_df.query("split == 1")
            sampling_pool = pd.concat([train_df, val_df])
            if extra_pool_df is not None:
                print(f"=> Including {len(extra_pool_df)} extra samples from test remainder.")
                sampling_pool = pd.concat([sampling_pool, extra_pool_df])

        # 2. Resample the pool to create the new training data
        new_train_df = WaterbirdsDynamicRho._resample_pool(
            pool=sampling_pool,
            original_train_df=train_df,
            target_bias=bias_amount,
            strategy=strategy,
            seed=random_seed,
            n_per_class=n_per_class
        )
        new_train_df['split'] = 0

        # 3. Get the other splits
        test_df = original_df.query("split == 2").copy()
        # If we borrowed from val, the new val split is empty
        val_df = pd.DataFrame(columns=original_df.columns) if strategy == "borrow_from_val" else original_df.query("split == 1").copy()

        # 4. Combine and save
        final_metadata = new_train_df.reset_index(drop=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_metadata.to_csv(output_path, index=False)
        print(f"Successfully saved new metadata to: {output_path}")
        print("\nNew split distribution:")
        print(final_metadata['split'].value_counts().sort_index())


    @staticmethod
    def _resample_pool(pool: pd.DataFrame, original_train_df: pd.DataFrame, target_bias: float, strategy: Strategy, seed: int, n_per_class: Dict[int, int] = None) -> pd.DataFrame:
        """Static helper method to resample a dataframe to a target bias."""
        print(f"=> Adjusting training set bias to {target_bias:.2f} using '{strategy}' strategy.")
        pool_groups = {(y, p): pool[(pool['y'] == y) & (pool['place'] == p)] for y in [0, 1] for p in [0, 1]}

        if strategy == "downsample":
            n_minority_lb = len(original_train_df[(original_train_df['y'] == 0) & (original_train_df['place'] == 1)])
            n_minority_wb = len(original_train_df[(original_train_df['y'] == 1) & (original_train_df['place'] == 0)])
            target_counts = {
                (0, 0): int(n_minority_lb * target_bias / (1.0 - target_bias)),
                (0, 1): n_minority_lb,
                (1, 0): n_minority_wb,
                (1, 1): int(n_minority_wb * target_bias / (1.0 - target_bias)),
            }
            
        elif strategy == "borrow_from_val_fixed":
            assert n_per_class is not None, "borrow_from_val_fixed requires n_per_class"
            target_counts = {
                (0, 1): int(n_per_class[0] * (1 - target_bias)),
                (0, 0): int(n_per_class[0] * target_bias),
                (1, 0): int(n_per_class[1] * (1 - target_bias)),
                (1, 1): int(n_per_class[1] * target_bias),
            }
            
        else:  # "oversample" or "borrow_from_val"
            class_counts = original_train_df['y'].value_counts()
            target_counts = {
                (0, 1): int(class_counts[0] * (1 - target_bias)),
                (0, 0): int(class_counts[0] * target_bias),
                (1, 0): int(class_counts[1] * (1 - target_bias)),
                (1, 1): int(class_counts[1] * target_bias),
            }
        
        sampled_groups = []
        for (y, p), n_target in target_counts.items():
            group_pool = pool_groups[(y, p)]
            replace = n_target > len(group_pool)
            if replace and strategy != "downsample":
                print(f"   - Oversampling group (y={y}, p={p}) from {len(group_pool)} to {n_target}")
            sampled_groups.append(group_pool.sample(n=n_target, random_state=seed, replace=replace))

        final_df = pd.concat(sampled_groups)
        return final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "waterbirds.tar.gz")
        print(f"=> Downloading {WaterbirdsDynamicRho.DATASET_NAME} from {WaterbirdsDynamicRho.DOWNLOAD_URL}")
        try:
            response = requests.get(WaterbirdsDynamicRho.DOWNLOAD_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(output_path, "wb") as f, tqdm(desc=output_path, total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except requests.RequestException as e:
            raise RuntimeError(f"Unable to complete dataset download: {e}")
        print(f"=> Extracting {output_path} to {self.root}")
        try:
            with tarfile.open(output_path, "r:gz") as tar:
                tar.extractall(path=self.root)
        except tarfile.TarError as e:
            raise RuntimeError(f"Unable to extract {output_path}: {e}")
        finally:
            os.remove(output_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[int, int], int]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, (sample["class_label"], sample["bias_label"]), index

    def get_group_counts(self) -> torch.Tensor:
        group_counts = torch.zeros(self.n_groups, dtype=torch.long)
        for sample in self.samples:
            group_idx = sample["class_label"] * 2 + sample["bias_label"]
            group_counts[group_idx] += 1
        return group_counts

    def __repr__(self) -> str:
        return f"WaterbirdsDynamicRho(env={self.env}, num_samples={len(self)})"
    
    @staticmethod
    def compute_group_counts_for_rho(
        rho: float,
        pool_df: pd.DataFrame,
    ) -> Dict[Tuple[int,int], int]:
        """
        Calcola i target count per ogni gruppo dato rho,
        rispettando i campioni disponibili nel pool.
        Ritorna None se il pool è insufficiente.
        """
        pool_groups = {
            (y, p): pool_df[(pool_df['y'] == y) & (pool_df['place'] == p)]
            for y in [0, 1] for p in [0, 1]
        }
        class_counts = pool_df['y'].value_counts()

        target_counts = {
            (0, 0): int(class_counts[0] * rho),
            (0, 1): int(class_counts[0] * (1 - rho)),
            (1, 1): int(class_counts[1] * rho),
            (1, 0): int(class_counts[1] * (1 - rho)),
        }

        clipped = {
            key: min(n, len(pool_groups[key]))
            for key, n in target_counts.items()
        }

        # Ricalcola rho effettivo e aggiusta per mantenerlo coerente
        for y in [0, 1]:
            aligned_key   = (y, y)        # (0,0) o (1,1)
            conflict_key  = (y, 1 - y)    # (0,1) o (1,0)
            n_align   = clipped[aligned_key]
            n_conf    = clipped[conflict_key]
            actual_rho = n_align / (n_align + n_conf)

            if actual_rho > rho:  # troppi aligned
                n_conf = clipped[conflict_key]
                n_align = int(n_conf * rho / (1 - rho))
            elif actual_rho < rho:  # troppi conflict
                n_align = clipped[aligned_key]
                n_conf = int(n_align * (1 - rho) / rho)

            clipped[aligned_key]  = n_align
            clipped[conflict_key] = n_conf

        return clipped
    
    @staticmethod
    def create_held_out_test_split(
        root: str = "./data",
        n_per_group: int = 100,
        random_seed: int = 42,
        output_path: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the test set into:
        - A fixed held-out evaluation set (n_per_group samples per group)
        - A remainder that can be used as extra training pool

        Returns (held_out_df, remainder_df), and optionally saves a new metadata CSV.
        """
        base_data_dir = os.path.join(
            root, WaterbirdsDynamicRho.DATASET_NAME,
            "waterbird_complete95_forest2water2"
        )
        original_df = pd.read_csv(os.path.join(base_data_dir, "metadata.csv"))
        test_df = original_df.query("split == 2").copy()

        held_out_parts = []
        remainder_parts = []

        for y in [0, 1]:
            for p in [0, 1]:
                group = test_df[(test_df['y'] == y) & (test_df['place'] == p)]
                assert len(group) >= n_per_group, (
                    f"Not enough samples for group (y={y}, p={p}): "
                    f"found {len(group)}, need {n_per_group}"
                )
                sampled = group.sample(n=n_per_group, random_state=random_seed)
                held_out_parts.append(sampled)
                remainder_parts.append(group.drop(sampled.index))

        held_out_df = pd.concat(held_out_parts).reset_index(drop=True)
        remainder_df = pd.concat(remainder_parts).reset_index(drop=True)

        print(f"Held-out test set: {len(held_out_df)} samples ({n_per_group} per group)")
        print(f"Remainder (available for training pool): {len(remainder_df)} samples")

        if output_path is not None:
            # Save a new metadata CSV where:
            # split=2 -> held-out test
            # split=3 -> remainder (to be used as extra pool)
            held_out_df['split'] = 2
            remainder_df['split'] = 3
            train_val_df = original_df.query("split != 2").copy()
            combined = pd.concat([train_val_df, held_out_df, remainder_df], ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined.to_csv(output_path, index=False)
            print(f"Saved updated metadata to: {output_path}")

        return held_out_df, remainder_df


if __name__ == "__main__":
    rhos = [0.70, 0.80, 0.95]
    root = "./data"

    base_data_dir = os.path.join(
        root, WaterbirdsDynamicRho.DATASET_NAME,
        "waterbird_complete95_forest2water2"
    )
    original_df = pd.read_csv(os.path.join(base_data_dir, "metadata.csv"))

    # --- STEP 0: carve out held-out test set, get remainder ---
    held_out_metadata_path = "./data/waterbirds_rho/metadata_with_held_out.csv"
    _, remainder_df = WaterbirdsDynamicRho.create_held_out_test_split(
        root=root,
        n_per_group=100,
        random_seed=42,
        output_path=held_out_metadata_path,
    )

    # Pool = train + val + test remainder
    pool_df = original_df.query("split == 0 or split == 1").copy()
    pool_df = pd.concat([pool_df, remainder_df]).reset_index(drop=True)

    # --- STEP 1: compute counts ---
    all_counts = {}
    all_sizes  = {}

    for rho in rhos:
        counts = WaterbirdsDynamicRho.compute_group_counts_for_rho(rho, pool_df)
        total  = sum(counts.values())
        all_counts[rho] = counts
        all_sizes[rho]  = total
        print(f"rho={rho:.2f}: counts={counts}, total={total}")

    min_per_class_per_class = {
        0: min(counts[(0,0)] + counts[(0,1)] for counts in all_counts.values()),
        1: min(counts[(1,0)] + counts[(1,1)] for counts in all_counts.values()),
    }
    shared_min = min(min_per_class_per_class.values())
    min_per_class = {0: shared_min, 1: shared_min}
    print(f"shared_min per class = {shared_min}, total dataset size = {shared_min * 2}")

    # --- STEP 2: generate datasets ---
    os.makedirs("./data/waterbirds_rho", exist_ok=True)

    for rho in rhos:
        counts  = all_counts[rho]
        n_class = {
            0: counts[(0,0)] + counts[(0,1)],
            1: counts[(1,0)] + counts[(1,1)],
        }

        final_counts = {}
        for y in [0, 1]:
            aligned_key  = (y, y)
            conflict_key = (y, 1 - y)
            current_total_y = n_class[y]

            if current_total_y > min_per_class[y]:   # <-- per-class min
                scale   = min_per_class[y] / current_total_y
                n_align = int(counts[aligned_key] * scale)
                n_conf  = min_per_class[y] - n_align
            else:
                n_align = counts[aligned_key]
                n_conf  = counts[conflict_key]

            effective_rho = n_align / (n_align + n_conf)
            final_counts[aligned_key]  = n_align
            final_counts[conflict_key] = n_conf
            print(f"rho={rho:.2f}, class={y}: align={n_align}, conflict={n_conf}, "
                  f"effective_rho={effective_rho:.3f}")

        out_path = f"./data/waterbirds_rho/waterbirds_{rho:.2f}_fixed.csv"
        if not os.path.exists(out_path):
            WaterbirdsDynamicRho.create_and_save_metadata(
                output_path=out_path,
                strategy="borrow_from_val_fixed",
                bias_amount=rho,
                n_per_class=min_per_class,
                root=root,
                extra_pool_df=remainder_df,  
            )

        print(min_per_class_per_class)
        print(shared_min)
        print(pool_df.groupby(["y", "place"]).size())

        train = WaterbirdsDynamicRho(env="train", metadata_path=out_path)
        print(f"total dataset: {len(train)} samples, groups={train.get_group_counts().tolist()}\n")