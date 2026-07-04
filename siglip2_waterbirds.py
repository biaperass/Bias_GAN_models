#!/usr/bin/env python3
"""
Apply a SigLIP2 detector to generated Waterbirds images.

The script computes the detected alignment ratio (rho_hat) for each
(rho, truncation, class) split.

Expected directory layout:

  GEN_ROOT/
    waterbirds_10k_rho{RHO_TAG}/
      waterbirds_trunc{TRUNC}_class{Y}/
        *.png
"""

import os
import re
import csv
import glob
import argparse

RHO_TAG_TO_NOMINAL = {
    "70": 0.70,
    "80": 0.80,
    "95": 0.95,
}

IMG_EXTS = (".png", ".jpg", ".jpeg")

TOP_LEVEL_PREFIX = "waterbirds_10k"   # matches waterbirds_10k_rho{N}
SUBFOLDER_PREFIX = "waterbirds"       # matches waterbirds_trunc{T}_class{Y}


def normalize_trunc(trunc_str):
    if "." in trunc_str:
        return float(trunc_str)
    n = int(trunc_str)
    if n == 1:
        return 1.0
    if 0 < n < 100:
        return n / 100.0
    raise ValueError(
        f"Can't confidently normalize truncation string {trunc_str!r} -- "
        f"doesn't match the '0.5'/'50'/'1' patterns seen on disk. Check "
        f"manually rather than letting this guess."
    )


def find_class_dirs(gen_root):
    rho_pat = re.compile(rf"^{re.escape(TOP_LEVEL_PREFIX)}_rho(\d+)$")
    class_pat = re.compile(rf"^{re.escape(SUBFOLDER_PREFIX)}_trunc([0-9.]+)_class([01])$")

    for rho_dir in sorted(os.listdir(gen_root)):
        rho_match = rho_pat.match(rho_dir)
        if not rho_match:
            continue
        rho_tag = rho_match.group(1)
        rho_dir_path = os.path.join(gen_root, rho_dir)
        if not os.path.isdir(rho_dir_path):
            continue

        for class_dir in sorted(os.listdir(rho_dir_path)):
            class_match = class_pat.match(class_dir)
            if not class_match:
                continue
            trunc_str, y = class_match.group(1), int(class_match.group(2))
            trunc_val = normalize_trunc(trunc_str)
            yield rho_tag, trunc_str, trunc_val, y, os.path.join(rho_dir_path, class_dir)


def list_images(dir_path):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    return sorted(files)


def detect_dir(image_paths, y, out_csv, run):
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            done = {r["img_filename"] for r in csv.DictReader(f)}
        print(f"    resume: {len(done)} already done")
    mode = "a" if done else "w"
    skipped = []
    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["img_filename", "y", "score", "detected"])
        for i, path in enumerate(image_paths, 1):
            fname = os.path.basename(path)
            if fname in done:
                continue
            try:
                det, score = run(path)
            except Exception as e:
                skipped.append(fname)
                print(f"    !! skipping {fname}: {e}")
                continue
            w.writerow([fname, y, f"{score:.6f}", "" if det is None else det])
            if i % 500 == 0:
                f.flush()
                print(f"    [{i}/{len(image_paths)}]")
    if skipped:
        skip_log = out_csv.replace(".csv", "_skipped.txt")
        with open(skip_log, "a") as f:
            f.write("\n".join(skipped) + "\n")
        print(f"    {len(skipped)} skipped, logged to {skip_log}")


def evaluate_generated(det_csv, rho_tag, nominal_rho, trunc_val, y, flip):
    rows = []
    with open(det_csv, newline="") as f:
        for r in csv.DictReader(f):
            d = r["detected"]
            rows.append(int(d) if d in ("0", "1") else None)

    n_total = len(rows)
    valid = [d for d in rows if d is not None]
    n_null = n_total - len(valid)

    def aligned(a):
        return (a == y) if not flip else (a != y)

    rho_hat_slice = (sum(1 for d in valid if aligned(d)) / len(valid)
                      if valid else float("nan"))

    print(f"  rho_tag={rho_tag} (nominal {nominal_rho}) trunc={trunc_val} y={y}: "
          f"rho_hat_slice={rho_hat_slice:.4f}  (valid {len(valid)}, null {n_null}, "
          f"total {n_total})")

    return {"rho_tag": rho_tag, "rho_nominal": nominal_rho, "trunc": trunc_val,
            "y": y, "rho_hat_slice": round(rho_hat_slice, 4),
            "n_valid": len(valid), "n_null": n_null, "n_total": n_total}


def append_summary(row, path):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def build_run(model_id, class0_prompts, class1_prompts, margin_thresh):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageFile
    from transformers import AutoModel, AutoProcessor

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # generated PNGs can be truncated

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    def _feat(out):
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.inference_mode()
    def _text_proto(prompts):
        t = proc(text=prompts, return_tensors="pt", padding="max_length")
        t = {k: v.to(device) for k, v in t.items()}
        e = F.normalize(_feat(model.get_text_features(**t)).float(), dim=-1)
        return F.normalize(e.mean(0, keepdim=True), dim=-1)

    proto0 = _text_proto(class0_prompts)
    proto1 = _text_proto(class1_prompts)

    @torch.inference_mode()
    def run(image_path):
        img = Image.open(image_path).convert("RGB")
        px = proc(images=img, return_tensors="pt")
        px = {k: (v.to(device).to(model.dtype) if v.is_floating_point()
                  else v.to(device)) for k, v in px.items()}
        e = F.normalize(_feat(model.get_image_features(**px)).float(), dim=-1)
        s0 = (e @ proto0.T).item()
        s1 = (e @ proto1.T).item()
        margin = s1 - s0
        if abs(margin) < margin_thresh:
            return None, margin
        return (1 if margin > 0 else 0), margin
    return run


BIAS_PROMPTS_CLASS0 = [  # class 0 = land
    "a photo of a bird in a forest",
    "a bird on land, on a branch or in the woods",
    "a photograph with a natural land background",
]
BIAS_PROMPTS_CLASS1 = [  # class 1 = water
    "a photo of a bird in water",
    "a bird on the ocean, a lake, or a river",
    "a photograph with a natural water background",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="diff-proj_waterbirds_10k")
    ap.add_argument("--output_dir", default="result_diff-proj_waterbirds_siglip2")
    ap.add_argument("--model_id", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--margin_thresh", type=float, default=0.0)
    ap.add_argument("--flip_alignment", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "siglip_generated_waterbirds.csv")

    run = None
    if not args.analyze_only:
        run = build_run(args.model_id, BIAS_PROMPTS_CLASS0, BIAS_PROMPTS_CLASS1,
                         args.margin_thresh)

    n_dirs = 0
    for rho_tag, trunc_str, trunc_val, y, dir_path in find_class_dirs(args.gen_root):
        n_dirs += 1
        if rho_tag not in RHO_TAG_TO_NOMINAL:
            raise ValueError(f"rho_tag {rho_tag!r} (dir: {dir_path}) not in "
                              f"RHO_TAG_TO_NOMINAL.")
        nominal_rho = RHO_TAG_TO_NOMINAL[rho_tag]

        det_csv = os.path.join(
            args.output_dir,
            f"detections_gen_waterbirds_rho{rho_tag}_trunc{trunc_val}_class{y}.csv"
        )

        if not args.analyze_only:
            images = list_images(dir_path)
            print(f"\n{dir_path}: {len(images)} images "
                  f"(trunc_str={trunc_str!r} -> trunc_val={trunc_val})")
            detect_dir(images, y, det_csv, run)

        if os.path.exists(det_csv):
            append_summary(
                evaluate_generated(det_csv, rho_tag, nominal_rho, trunc_val, y,
                                    args.flip_alignment),
                summary_path,
            )

    if n_dirs == 0:
        print(f"\n!! No matching directories found under {args.gen_root} -- "
              f"check TOP_LEVEL_PREFIX/SUBFOLDER_PREFIX against actual folder "
              f"names before assuming this ran correctly.")
    print(f"\nappended to {summary_path} ({n_dirs} rho x trunc x y slices found)")


if __name__ == "__main__":
    main()