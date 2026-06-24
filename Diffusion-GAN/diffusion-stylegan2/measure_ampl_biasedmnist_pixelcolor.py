# import os, re, csv
# import numpy as np
# from PIL import Image
# from collections import defaultdict

# COLOUR_MAP = [
#     [255,0,0],[0,255,0],[0,0,255],[225,225,0],[225,0,225],
#     [0,255,255],[255,128,0],[255,0,128],[128,0,255],[128,128,128],
# ]
# COLOUR_NAMES = ["red","green","blue","yellow","magenta",
#                 "cyan","orange","rose","electric violet","grey"]
# PALETTE = np.array(COLOUR_MAP, dtype=np.float32)
# K = 10  # class i aligned with COLOUR_MAP[i]

# PARENT = "."
# BORDER_FRAC = 0.12
# DIST_THRESHOLD = 90.0

# def background_color(path, border_frac=BORDER_FRAC):
#     img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
#     h, w, _ = img.shape
#     bh, bw = max(1, round(h*border_frac)), max(1, round(w*border_frac))
#     border = np.concatenate([
#         img[:bh].reshape(-1,3), img[-bh:].reshape(-1,3),
#         img[:, :bw].reshape(-1,3), img[:, -bw:].reshape(-1,3),
#     ], axis=0)
#     return np.median(border, axis=0)

# def classify(rgb):
#     d = np.linalg.norm(PALETTE - rgb[None,:], axis=1)
#     i = int(np.argmin(d)); return i, float(d[i])

# def parse_rho(name):
#     m = re.search(r"rho(\d+)", name); return int(m.group(1))/100.0 if m else None

# def parse_leaf(name):
#     m = re.search(r"trunc([0-9.]+)_class(\d+)", name)
#     return (float(m.group(1)), int(m.group(2))) if m else (None, None)

# cells = defaultdict(lambda: {"aligned":0, "total":0, "hist":[0]*K})  # (rho,trunc,class)

# rho_dirs = sorted(d for d in os.listdir(PARENT)
#                   if os.path.isdir(os.path.join(PARENT,d)) and parse_rho(d) is not None)

# for rdir in rho_dirs:
#     rho = parse_rho(rdir); rpath = os.path.join(PARENT, rdir)
#     for leaf in sorted(os.listdir(rpath)):
#         lpath = os.path.join(rpath, leaf)
#         if not os.path.isdir(lpath): continue
#         trunc, cls = parse_leaf(leaf)
#         if cls is None: print(f"SKIP {leaf}"); continue
#         for fn in sorted(os.listdir(lpath)):
#             if not fn.lower().endswith((".png",".jpg",".jpeg")): continue
#             ci, dist = classify(background_color(os.path.join(lpath, fn)))
#             c = cells[(rho,trunc,cls)]
#             c["total"] += 1
#             if dist <= DIST_THRESHOLD:
#                 c["hist"][ci] += 1
#                 if ci == cls: c["aligned"] += 1


# truncs = sorted({t for (_,t,_) in cells})
# rhos   = sorted({r for (r,_,_) in cells})

# with open("aligned_per_class.csv","w",newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["rho","trunc","class","n_aligned","n_total","frac_aligned"])
#     for rho in rhos:
#         for trunc in truncs:
#             print(f"\n=== rho={rho:.2f}  trunc={trunc} ===")
#             fracs, tot_a, tot_n = [], 0, 0
#             for cls in range(K):
#                 c = cells.get((rho,trunc,cls))
#                 if not c or c["total"]==0: continue
#                 a,n = c["aligned"], c["total"]
#                 fr = a/n; fracs.append(fr); tot_a += a; tot_n += n
#                 print(f"  class {cls}: {a:4d}/{n:<4d}  = {fr:.3f}")
#                 w.writerow([rho,trunc,cls,a,n,round(fr,4)])
#             if fracs:
#                 macro = float(np.mean(fracs))          # mean of per-class fractions
#                 micro = tot_a/tot_n                     # total aligned / total samples
#                 print(f"  --> per-class avg (macro) = {macro:.3f}")
#                 print(f"  --> overall    (micro)    = {tot_a}/{tot_n} = {micro:.3f}")

# with open("color_matrix.csv","w",newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["rho","trunc","class"]+[f"n_{c}" for c in COLOUR_NAMES])
#     for (rho,trunc,cls) in sorted(cells):
#         w.writerow([rho,trunc,cls]+cells[(rho,trunc,cls)]["hist"])

import os, re, csv
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

# Save the attached snippet as color_utils.py next to this script.
from biased_mnist_utils import COLOUR_MAP, COLOUR_NAMES, get_color_idx_from_batch

K = 10  # class i aligned with COLOUR_MAP[i]
PARENT = "."
THRESHOLD = 0.7      # same non-white threshold the utils use (on the [0,1] scale)
BATCH = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tensor(path):
    """PIL RGB -> (3,H,W) float tensor in [-1,1], the range the utils expect."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0  # [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)                                   # (3,H,W)
    return t * 2.0 - 1.0                                                          # [-1,1]

def has_background(t):
    """True if the image has any non-white pixel (else get_color_from_batch -> NaN)."""
    x = (t + 1) / 2
    return bool(((x[0] < THRESHOLD) | (x[1] < THRESHOLD) | (x[2] < THRESHOLD)).any())

def parse_rho(name):
    m = re.search(r"rho(\d+)", name); return int(m.group(1))/100.0 if m else None

def parse_leaf(name):
    m = re.search(r"trunc([0-9.]+)_class(\d+)", name)
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)

cells = defaultdict(lambda: {"aligned":0, "total":0, "failed":0, "hist":[0]*K})  # (rho,trunc,class)

rho_dirs = sorted(d for d in os.listdir(PARENT)
                  if os.path.isdir(os.path.join(PARENT,d)) and parse_rho(d) is not None)

for rdir in rho_dirs:
    rho = parse_rho(rdir); rpath = os.path.join(PARENT, rdir)
    for leaf in sorted(os.listdir(rpath)):
        lpath = os.path.join(rpath, leaf)
        if not os.path.isdir(lpath): continue
        trunc, cls = parse_leaf(leaf)
        if cls is None: print(f"SKIP {leaf}"); continue
        c = cells[(rho,trunc,cls)]

        files = sorted(f for f in os.listdir(lpath)
                       if f.lower().endswith((".png",".jpg",".jpeg")))
        for i in range(0, len(files), BATCH):
            chunk = files[i:i+BATCH]
            tensors, valid = [], []
            for fn in chunk:
                t = load_tensor(os.path.join(lpath, fn))
                c["total"] += 1
                if has_background(t):
                    tensors.append(t); valid.append(fn)
                else:
                    c["failed"] += 1          # degenerate (all-white) -> not labeled
            if not tensors: continue
            # stack assumes a fixed image size within the run (true for a generation batch)
            batch = torch.stack(tensors, dim=0).to(device)
            idx = get_color_idx_from_batch(batch).cpu().tolist()   # <-- their function
            for ci in idx:
                c["hist"][ci] += 1
                if ci == cls: c["aligned"] += 1

# ---- MAIN RESULT: aligned per class + per-class average, per (rho,trunc) ----
truncs = sorted({t for (_,t,_) in cells})
rhos   = sorted({r for (r,_,_) in cells})

with open("aligned_per_class.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["rho","trunc","class","n_aligned","n_total","frac_aligned"])
    for rho in rhos:
        for trunc in truncs:
            print(f"\n=== rho={rho:.2f}  trunc={trunc} ===")
            fracs, tot_a, tot_n, tot_f = [], 0, 0, 0
            for cls in range(K):
                c = cells.get((rho,trunc,cls))
                if not c or c["total"]==0: continue
                a,n = c["aligned"], c["total"]
                fr = a/n; fracs.append(fr); tot_a += a; tot_n += n; tot_f += c["failed"]
                print(f"  class {cls}: {a:4d}/{n:<4d}  = {fr:.3f}"
                      + (f"   (failed={c['failed']})" if c['failed'] else ""))
                w.writerow([rho,trunc,cls,a,n,round(fr,4)])
            if fracs:
                macro = float(np.mean(fracs))
                micro = tot_a/tot_n
                print(f"  --> per-class avg (macro) = {macro:.3f}")
                print(f"  --> overall    (micro)    = {tot_a}/{tot_n} = {micro:.3f}"
                      + (f"   (failed={tot_f})" if tot_f else ""))

# ---- BYPRODUCT: color histogram per cell ----
with open("color_matrix.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["rho","trunc","class"]+[f"n_{c}" for c in COLOUR_NAMES])
    for (rho,trunc,cls) in sorted(cells):
        w.writerow([rho,trunc,cls]+cells[(rho,trunc,cls)]["hist"])