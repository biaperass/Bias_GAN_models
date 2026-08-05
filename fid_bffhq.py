import os
import re
import csv
import glob
import argparse
import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from pytorch_fid.fid_score import ImagePathDataset
from pytorch_fid.inception import InceptionV3

# ----------------------------- CONFIG -----------------------------
# Real BFFHQ images are NOT behind a metadata CSV -- aligned/conflicting
# and y are already encoded by directory structure for the train split:
#   <REAL_ROOT>/<pct_dir>/align/<y>/*.png     -> bias-aligned
#   <REAL_ROOT>/<pct_dir>/conflict/<y>/*.png  -> bias-conflicting
#
# valid/ and test/ are FLAT (no align/conflict subdirs) and are NOT
# per-rho -- they are the same fixed pool regardless of which pct_dir
# you're evaluating. Condition + class are recovered from the filename
# suffix: "..._{y}_{bias}.png". y == bias -> aligned, y != bias -> conflicting.
REAL_ROOT = "data_bffhq/bffhq/bffhq"
VALID_DIR = os.path.join(REAL_ROOT, "valid")
TEST_DIR = os.path.join(REAL_ROOT, "test")

# CONFIRM THIS MAPPING ON DISK before trusting results. BFFHQ's folder
# names are conflict-percentage, not rho, and the two are easy to get
# backwards: 0.5pct (very biased) -> rho=0.995, 5pct -> rho=0.95.
# If your data only has one of these levels, delete the other entry --
# do not leave a guessed mapping in place.
RHO_TO_PCT_DIR = {
    "95": "0.5pct",
    "99": "0.5pct",
}
RHO_MAP = {
    "95": "0.95",
    "99": "0.99",
}

# Because valid/test are NOT rho-specific, pooling them into every rho's
# conflicting reference means that pool is constant across rho tokens
# while only the tiny train-conflict contribution actually varies with
# rho. That can mask true rho-dependent differences in the conflicting
# distribution. Keep this in mind when reading FID deltas across rho.
AUGMENT_CONFLICTING_WITH_VALID_TEST = True

# Off by default. Turning this on keeps aligned/conflicting symmetric
# (same source mixture for both), which removes the split-source
# confound in point 2, but also dilutes whatever signal currently
# exists in the aligned condition. Decide deliberately, don't just flip
# it because it "seems more consistent."
AUGMENT_ALIGNED_WITH_VALID_TEST = False

ap = argparse.ArgumentParser()
ap.add_argument("--gen_root", required=True)
ap.add_argument("--detections_dir", required=True)
ap.add_argument("--output_csv", required=True)
args = ap.parse_args()
GEN_ROOT = args.gen_root
DETECTIONS_DIR = args.detections_dir
OUTPUT_CSV = args.output_csv

DETECTIONS_FILE_RE = re.compile(
    r"^detections_gen_bffhq_rho(\d+)_trunc([0-9.]+)_class([01])\.csv$"
)

# y and bias digit are the last two underscore-separated tokens before
# the extension, e.g. "12345_0_1.png" -> y=0, bias=1 -> conflicting.
FILENAME_COND_RE = re.compile(r"_(\d)_(\d)\.(png|jpg|jpeg)$", re.IGNORECASE)

MIN_N_WARN = 200  # below this, FID is unstable -- flag it, don't silently report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIMS = 2048
BATCH_SIZE = 32
IMG_SIZE = 299
# --------------------------------------------------------------------


def classify_by_filename(path):
    """Returns (y, condition) parsed from filename, or None if it doesn't match."""
    m = FILENAME_COND_RE.search(os.path.basename(path))
    if not m:
        return None
    y, bias = int(m.group(1)), int(m.group(2))
    condition = "aligned" if y == bias else "conflicting"
    return y, condition


def load_split_paths_by_condition(split_dir, class_id):
    """Flat-folder (valid/test) variant: condition comes from filename, not directory."""
    aligned, conflicting = [], []
    if not os.path.isdir(split_dir):
        print(f"  [warn] split dir not found: {split_dir}")
        return aligned, conflicting

    unmatched = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in glob.glob(os.path.join(split_dir, ext)):
            parsed = classify_by_filename(p)
            if parsed is None:
                unmatched += 1
                continue
            y, condition = parsed
            if y != class_id:
                continue
            (aligned if condition == "aligned" else conflicting).append(p)

    if unmatched:
        print(f"  [warn] {unmatched} file(s) in {split_dir} didn't match the "
              f"'..._{{y}}_{{bias}}.ext' naming convention -- skipped, not counted")

    return sorted(aligned), sorted(conflicting)


def load_real_paths_by_condition(rho_token, class_id):
    """
    Train split: directory-based (existing behavior).
    Optionally extended with valid/test (filename-based), which are the
    SAME pool for every rho_token -- see AUGMENT_* flags above.

    Returns (aligned_paths, conflicting_paths, provenance) where
    provenance is a dict of counts by source, for transparency in the
    output CSV.
    """
    if rho_token not in RHO_TO_PCT_DIR:
        raise KeyError(
            f"rho token '{rho_token}' has no entry in RHO_TO_PCT_DIR -- "
            f"add it after confirming the correct pct folder on disk."
        )
    pct_dir = RHO_TO_PCT_DIR[rho_token]

    aligned_dir = os.path.join(REAL_ROOT, pct_dir, "align", str(class_id))
    conflicting_dir = os.path.join(REAL_ROOT, pct_dir, "conflict", str(class_id))

    if not os.path.isdir(aligned_dir):
        print(f"  [warn] aligned dir does not exist: {aligned_dir}")
    if not os.path.isdir(conflicting_dir):
        print(f"  [warn] conflicting dir does not exist: {conflicting_dir}")

    aligned_train = sorted(glob.glob(os.path.join(aligned_dir, "*.png")))
    conflicting_train = sorted(glob.glob(os.path.join(conflicting_dir, "*.png")))

    aligned_valid, conflicting_valid = load_split_paths_by_condition(VALID_DIR, class_id)
    aligned_test, conflicting_test = load_split_paths_by_condition(TEST_DIR, class_id)

    provenance = {
        "n_aligned_train": len(aligned_train),
        "n_aligned_valid": 0,
        "n_aligned_test": 0,
        "n_conflicting_train": len(conflicting_train),
        "n_conflicting_valid": 0,
        "n_conflicting_test": 0,
    }

    aligned_all = list(aligned_train)
    if AUGMENT_ALIGNED_WITH_VALID_TEST:
        aligned_all += aligned_valid + aligned_test
        provenance["n_aligned_valid"] = len(aligned_valid)
        provenance["n_aligned_test"] = len(aligned_test)

    conflicting_all = list(conflicting_train)
    if AUGMENT_CONFLICTING_WITH_VALID_TEST:
        conflicting_all += conflicting_valid + conflicting_test
        provenance["n_conflicting_valid"] = len(conflicting_valid)
        provenance["n_conflicting_test"] = len(conflicting_test)

    for label, paths in (("aligned", aligned_all), ("conflicting", conflicting_all)):
        basenames = [os.path.basename(p) for p in paths]
        if len(basenames) != len(set(basenames)):
            dupes = len(basenames) - len(set(basenames))
            print(f"  [warn] {dupes} duplicate filename(s) in pooled {label} set for "
                  f"rho={rho_token}, class={class_id} -- check for train/valid/test overlap")

    return sorted(aligned_all), sorted(conflicting_all), provenance


def discover_detection_files():
    entries = sorted(os.listdir(DETECTIONS_DIR))
    matched = []
    unmatched = []
    for name in entries:
        full = os.path.join(DETECTIONS_DIR, name)
        if not os.path.isfile(full):
            continue
        m = DETECTIONS_FILE_RE.match(name)
        if m:
            rho_token, trunc_val, class_token = m.group(1), m.group(2), m.group(3)
            matched.append((full, rho_token, trunc_val, int(class_token)))
        else:
            unmatched.append(name)
    if unmatched:
        print(f"[warn] skipping {len(unmatched)} file(s) not matching "
              f"'detections_gen_bffhq_rho<R>_trunc<T>_class<C>.csv' "
              f"in {DETECTIONS_DIR}: {unmatched}")
    return matched


def load_generated_paths_by_condition(detection_csv_path, image_folder, class_id):
    """
    Single pass over one detection CSV, returns both conditions:
      aligned:     y == class_id and detected == y
      conflicting: y == class_id and detected != y

    detected is the SigLIP2 bias-attribute call (0/1), directly
    comparable to y for this dataset per your labeling convention.
    """
    aligned, conflicting = [], []
    y_class_mismatches = 0
    with open(detection_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row["y"])
            detected = int(row["detected"])
            if y != class_id:
                y_class_mismatches += 1
                continue
            img_path = os.path.join(image_folder, row["img_filename"])
            if y == detected:
                aligned.append(img_path)
            else:
                conflicting.append(img_path)
    if y_class_mismatches:
        print(f"  [warn] {y_class_mismatches} row(s) in {os.path.basename(detection_csv_path)} "
              f"have y != class implied by filename ({class_id}) -- check this file")
    return aligned, conflicting


def get_inception_model():
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
    model = InceptionV3([block_idx]).to(DEVICE)
    model.eval()
    return model


def get_activations_resized(files, model, batch_size, dims, device):
    if batch_size > len(files):
        batch_size = len(files)

    transform = TF.Compose([TF.Resize((IMG_SIZE, IMG_SIZE)), TF.ToTensor()])
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1
    )

    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(dataloader, leave=False):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]
    return pred_arr


def stats_for_paths(paths, model):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(paths)} files not found, e.g.:\n  {missing[0]}"
        )
    act = get_activations_resized(
        paths, model, batch_size=min(BATCH_SIZE, len(paths)), dims=DIMS, device=DEVICE
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def frechet_distance_robust(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    for exp in range(-6, 5):
        eps = 10.0 ** exp
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            max_imag = np.max(np.abs(covmean.imag))
            if max_imag > 1e-3:
                continue
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return fid, eps

    raise RuntimeError("Frechet distance failed to converge with eps up to 1e4.")


def main():
    args = ap.parse_args()
    model = get_inception_model()

    # cache real (mu, sigma, n, provenance) per (rho_token, class, condition) --
    # computed once per condition, not once per truncation value
    real_stats_cache = {}

    detection_files = discover_detection_files()
    if not detection_files:
        print(f"No detection files found in {DETECTIONS_DIR} -- nothing to do.")
        return

    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as out_f:
        writer = csv.writer(out_f)
        if write_header:
            writer.writerow([
                "rho", "class", "trunc", "condition", "n_real", "n_gen",
                "n_real_train", "n_real_valid", "n_real_test",
                "fid", "reg_eps", "warning"
            ])

        for detection_csv_path, rho_token, trunc_val, class_id in detection_files:
            if rho_token not in RHO_MAP:
                print(f"  [skip] unrecognized rho token '{rho_token}' in "
                      f"{os.path.basename(detection_csv_path)} -- add it to RHO_MAP if valid")
                continue
            rho_str = RHO_MAP[rho_token]

            # --- real stats, both conditions, cached per (rho_token, class) ---
            cache_key = (rho_token, class_id)
            if cache_key not in real_stats_cache:
                try:
                    real_aligned_paths, real_conflicting_paths, provenance = load_real_paths_by_condition(
                        rho_token, class_id
                    )
                except KeyError as e:
                    print(f"  [skip] {e}")
                    real_stats_cache[cache_key] = {"aligned": None, "conflicting": None}
                    continue
                print(f"[rho={rho_str}, class={class_id}] real aligned: {len(real_aligned_paths)} "
                      f"(train={provenance['n_aligned_train']}, valid={provenance['n_aligned_valid']}, "
                      f"test={provenance['n_aligned_test']}), "
                      f"real conflicting: {len(real_conflicting_paths)} "
                      f"(train={provenance['n_conflicting_train']}, valid={provenance['n_conflicting_valid']}, "
                      f"test={provenance['n_conflicting_test']}) (computing once, cached)")

                entry = {"provenance": provenance}
                for cond, paths in (("aligned", real_aligned_paths), ("conflicting", real_conflicting_paths)):
                    if len(paths) < 2:
                        print(f"  [skip-real] not enough real {cond} images for rho={rho_str}, class={class_id}")
                        entry[cond] = None
                    else:
                        mu, sigma = stats_for_paths(paths, model)
                        entry[cond] = (mu, sigma, len(paths))
                real_stats_cache[cache_key] = entry

            real_entry = real_stats_cache[cache_key]
            provenance = real_entry.get("provenance", {})

            # --- generated paths, both conditions, single CSV read ---
            image_folder = os.path.join(
                GEN_ROOT,
                f"bffhq_rho{rho_token}",
                f"bffhq_trunc{trunc_val}_class{class_id}",
            )
            if not os.path.isdir(image_folder):
                print(f"  [skip] image folder not found: {image_folder}")
                continue

            gen_aligned_paths, gen_conflicting_paths = load_generated_paths_by_condition(
                detection_csv_path, image_folder, class_id
            )
            gen_paths_by_cond = {"aligned": gen_aligned_paths, "conflicting": gen_conflicting_paths}

            for condition in ("aligned", "conflicting"):
                real_stats = real_entry.get(condition)
                if real_stats is None:
                    continue
                mu_real, sigma_real, n_real = real_stats

                gen_paths = gen_paths_by_cond[condition]
                if len(gen_paths) < 2:
                    print(f"  [skip] not enough {condition} generated images in "
                          f"{os.path.basename(detection_csv_path)}")
                    continue

                try:
                    mu_gen, sigma_gen = stats_for_paths(gen_paths, model)
                except FileNotFoundError as e:
                    print(f"  [skip] {os.path.basename(detection_csv_path)} ({condition}): {e}")
                    continue

                fid, eps_used = frechet_distance_robust(mu_real, sigma_real, mu_gen, sigma_gen)

                n_gen = len(gen_paths)
                warning = ""
                if n_real < MIN_N_WARN or n_gen < MIN_N_WARN:
                    warning = f"n<{MIN_N_WARN}, FID unstable"

                if condition == "aligned":
                    n_train = provenance.get("n_aligned_train", "")
                    n_valid = provenance.get("n_aligned_valid", "")
                    n_test = provenance.get("n_aligned_test", "")
                else:
                    n_train = provenance.get("n_conflicting_train", "")
                    n_valid = provenance.get("n_conflicting_valid", "")
                    n_test = provenance.get("n_conflicting_test", "")

                print(f"  rho={rho_str} class={class_id} trunc={trunc_val} [{condition}]: "
                      f"FID={fid:.3f} (n_real={n_real} [train={n_train}, valid={n_valid}, test={n_test}], "
                      f"n_gen={n_gen}, eps={eps_used:.0e})"
                      + (f"  [{warning}]" if warning else ""))

                writer.writerow([rho_str, class_id, trunc_val, condition, n_real, n_gen,
                                  n_train, n_valid, n_test,
                                  f"{fid:.6f}", f"{eps_used:.0e}", warning])
                out_f.flush()

    print(f"\nDone. Results appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()