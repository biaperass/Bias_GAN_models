import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os
import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path",    type=str, required=True,  help="Path al metadata CSV di Waterbirds")
parser.add_argument("--image_root",  type=str, required=True,  help="Root dir delle immagini originali")
parser.add_argument("--output_csv",  type=str, default="results_original_waterbirds.csv")
parser.add_argument("--split",       type=int, default=None,   help="Filtra per split: 0=train, 1=val, 2=test. Default: tutti.")
args = parser.parse_args()

# --- Carica modello ---
model_path = "Salesforce/blip2-flan-t5-xl"
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).float()
model.eval()


def run_inference(image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=5
        )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def parse_place(raw_answer: str):
    text = raw_answer.strip().lower()
    has_water = "yes" in text
    has_land  = "no"  in text
    if has_water and not has_land:
        return "yes"
    if has_land and not has_water:
        return "no"
    return None


# --- Carica metadata ---
rows = []
with open(args.csv_path, newline="") as f:
    for row in csv.DictReader(f):
        if args.split is not None and int(row["split"]) != args.split:
            continue
        rows.append({
            "img_filename": row["img_filename"],
            "y":     int(row["y"]),
            "place": int(row["place"]),   # 0=land, 1=water
        })

print(f"Immagini da processare: {len(rows)}"
      + (f" (split={args.split})" if args.split is not None else " (tutti i split)"))

prompt = "Is the background predominantly water? Answer with one word: yes or no. Answer:"

# --- Riprendi da dove eravamo ---
processed = set()
if os.path.exists(args.output_csv):
    with open(args.output_csv, newline="") as f:
        for row in csv.DictReader(f):
            processed.add(row["img_filename"])
    print(f"Resume: trovate {len(processed)} righe già processate.")

mode = "a" if os.path.exists(args.output_csv) else "w"

with open(args.output_csv, mode, newline="") as f:
    writer = csv.writer(f)
    if mode == "w":
        writer.writerow(["img_filename", "y", "place", "raw_answer", "parsed_place"])

    for i, meta in enumerate(rows, start=1):
        if meta["img_filename"] in processed:
            continue

        img_path = os.path.join(args.image_root, meta["img_filename"])
        raw    = run_inference(img_path, prompt)
        parsed = parse_place(raw)

        writer.writerow([
            meta["img_filename"],
            meta["y"],
            meta["place"],
            raw,
            parsed,
        ])

        if i % 100 == 0:
            f.flush()
            print(f"[{i}/{len(rows)}] processate")

# --- Metriche post-hoc ---
# ground truth: place=1 -> "yes" (water), place=0 -> "no" (land)
GT = {1: "yes", 0: "no"}

counts = defaultdict(lambda: {"correct": 0, "wrong": 0, "null": 0})

with open(args.output_csv, newline="") as f:
    for row in csv.DictReader(f):
        y      = int(row["y"])
        place  = int(row["place"])
        parsed = row["parsed_place"] or None
        gt     = GT[place]

        for key in ["overall", f"y={y}", f"place={place}", f"y={y},place={place}"]:
            if parsed is None:
                counts[key]["null"] += 1
            elif parsed == gt:
                counts[key]["correct"] += 1
            else:
                counts[key]["wrong"] += 1

print("\n=== Accuracy BLIP-2 su immagini reali ===")
print(f"{'Slice':<25} {'Correct':>8} {'Valid':>8} {'Acc':>8} {'Null':>6} {'Total':>7}")
print("-" * 65)

key_order = ["overall",
             "y=0", "y=1",
             "place=0", "place=1",
             "y=0,place=0", "y=0,place=1",
             "y=1,place=0", "y=1,place=1"]

for key in key_order:
    if key not in counts:
        continue
    c     = counts[key]
    valid = c["correct"] + c["wrong"]
    total = valid + c["null"]
    acc   = c["correct"] / valid if valid > 0 else float("nan")
    print(f"{key:<25} {c['correct']:>8} {valid:>8} {acc:>7.1%} {c['null']:>6} {total:>7}")