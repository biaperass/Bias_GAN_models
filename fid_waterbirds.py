import os
import re
import csv
import glob
import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from pytorch_fid.fid_score import ImagePathDataset
from pytorch_fid.inception import InceptionV3

# ----------------------------- CONFIG -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--gen_root", required=True)
ap.add_argument("--detections_dir", required=True)
ap.add_argument("--output_csv", required=True)
args = ap.parse_args()
GEN_ROOT = args.gen_root
DETECTIONS_DIR = args.detections_dir
OUTPUT_CSV = args.output_csv

DETECTIONS_FILE_RE = re.compile(
    r"^detections_gen_waterbirds_rho(\d+)_trunc([0-9.]+)_class([01])\.csv$"
)

RHO_MAP = {
    "70": "0.70",
    "80": "0.80",
    "95": "0.95",
}

MIN_N_WARN = 200  # below this, FID is unstable -- flag it, don't silently report

OUTPUT_CSV = "fid_diff-gan_results_aligned_vs_conflicting_seed_2.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIMS = 2048
BATCH_SIZE = 32
IMG_SIZE = 299
# --------------------------------------------------------------------

def train_original_csv_path():
    return os.path.join(TRAIN_IMG_ROOT, f"metadata.csv")

def train_csv_path(rho_str):
    return os.path.join(TRAIN_CSV_DIR, f"waterbirds_{rho_str}_fixed.csv")


def load_real_paths_by_condition(rho_str, class_id):
    """
    Single pass over the real CSV, returns both conditions at once
    for a given class_id:
      aligned:     y == class_id and y == place
      conflicting: y == class_id and y != place
    """
    csv_path = train_original_csv_path()
    aligned, conflicting = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row["y"])
            place = int(row["place"])
            if y != class_id:
                continue
            img_path = os.path.join(TRAIN_IMG_ROOT, row["img_filename"])
            if y == place:
                aligned.append(img_path)
            else:
                conflicting.append(img_path)
    return aligned, conflicting


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
              f"'detections_gen_waterbirds_rho<R>_trunc<T>_class<C>.csv' "
              f"in {DETECTIONS_DIR}: {unmatched}")
    return matched


def load_generated_paths_by_condition(detection_csv_path, image_folder, class_id):
    """
    Single pass over one detection CSV, returns both conditions:
      aligned:     y == class_id and detected == y
      conflicting: y == class_id and detected != y

    NOTE: this assumes "detected" encodes the VLM's inferred bias attribute
    (place-equivalent), mirroring y == place for real images -- same
    assumption the original script already made for the aligned case.
    Verify this before trusting the conflicting-side numbers.
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
    model = get_inception_model()

    # cache real (mu, sigma, n) per (rho_str, class, condition) -- computed
    # once per condition, not once per truncation value
    real_stats_cache = {}

    detection_files = discover_detection_files()
    if not detection_files:
        print(f"No detection files found in {DETECTIONS_DIR} -- nothing to do.")
        return

    from collections import defaultdict
    grouped = defaultdict(dict)  # (rho_token, trunc_val) -> {class_id: detection_csv_path}
    for detection_csv_path, rho_token, trunc_val, class_id in detection_files:
        grouped[(rho_token, trunc_val)][class_id] = detection_csv_path

    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as out_f:
        writer = csv.writer(out_f)
        if write_header:
            writer.writerow([
                "rho", "trunc", "condition",
                "n_real_class0", "n_real_class1", "n_gen_class0", "n_gen_class1",
                "fid_class0", "fid_class1", "fid_mean",
                "warning"
            ])

        for (rho_token, trunc_val), class_map in sorted(grouped.items()):
            if rho_token not in RHO_MAP:
                print(f"  [skip] unrecognized rho token '{rho_token}' -- add it to RHO_MAP if valid")
                continue
            rho_str = RHO_MAP[rho_token]

            per_class_results = {"aligned": {}, "conflicting": {}}

            for class_id, detection_csv_path in sorted(class_map.items()):
                cache_key = (rho_str, class_id)
                if cache_key not in real_stats_cache:
                    real_aligned_paths, real_conflicting_paths = load_real_paths_by_condition(rho_str, class_id)
                    print(f"[rho={rho_str}, class={class_id}] real aligned: {len(real_aligned_paths)}, "
                        f"real conflicting: {len(real_conflicting_paths)} (computing once, cached)")

                    entry = {}
                    for cond, paths in (("aligned", real_aligned_paths), ("conflicting", real_conflicting_paths)):
                        if len(paths) < 2:
                            print(f"  [skip-real] not enough real {cond} images for rho={rho_str}, class={class_id}")
                            entry[cond] = None
                        else:
                            mu, sigma = stats_for_paths(paths, model)
                            entry[cond] = (mu, sigma, len(paths))
                    real_stats_cache[cache_key] = entry

                real_entry = real_stats_cache[cache_key]

                image_folder = os.path.join(
                    GEN_ROOT, f"generated_10k_rho{rho_token}", f"waterbirds_trunc{trunc_val}_class{class_id}"
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
                        print(f"  [skip] not enough {condition} generated images, class={class_id}")
                        continue

                    try:
                        mu_gen, sigma_gen = stats_for_paths(gen_paths, model)
                    except FileNotFoundError as e:
                        print(f"  [skip] {e}")
                        continue

                    fid, eps_used = frechet_distance_robust(mu_real, sigma_real, mu_gen, sigma_gen)
                    per_class_results[condition][class_id] = (fid, n_real, len(gen_paths), eps_used)

            for condition in ("aligned", "conflicting"):
                results = per_class_results[condition]
                if not results:
                    continue
                fids = [v[0] for v in results.values()]
                fid_mean = sum(fids) / len(fids)

                n_real_0, n_gen_0, n_real_1, n_gen_1 = "", "", "", ""
                fid_0, fid_1 = "", ""
                if 0 in results:
                    fid_0, n_real_0, n_gen_0, _ = results[0]
                if 1 in results:
                    fid_1, n_real_1, n_gen_1, _ = results[1]

                warning = ""
                if len(results) < 2:
                    warning = "only 1 class contributed"
                elif any(v[1] < MIN_N_WARN or v[2] < MIN_N_WARN for v in results.values()):
                    warning = f"n<{MIN_N_WARN} in at least one class, FID unstable"

                print(f"  rho={rho_str} trunc={trunc_val} [{condition}]: "
                    f"fid_class0={fid_0}, fid_class1={fid_1}, mean={fid_mean:.3f}"
                    + (f"  [{warning}]" if warning else ""))

                writer.writerow([rho_str, trunc_val, condition,
                                n_real_0, n_real_1, n_gen_0, n_gen_1,
                                f"{fid_0:.6f}" if fid_0 != "" else "",
                                f"{fid_1:.6f}" if fid_1 != "" else "",
                                f"{fid_mean:.6f}", warning])
                out_f.flush()

    print(f"\nDone. Results appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()