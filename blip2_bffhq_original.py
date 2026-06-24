import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path",   type=str, required=True,  help="Path al metadata CSV di BFFHQ")
parser.add_argument("--output_csv", type=str, default="results_original_bffhq.csv")
args = parser.parse_args()

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


def parse_gender(raw_answer: str):
    text = raw_answer.strip().lower()
    is_male   = "yes" in text
    is_female = "no"  in text
    if is_male and not is_female:
        return "yes"
    if is_female and not is_male:
        return "no"
    return None


rows = []
with open(args.csv_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append({
            "img_filename": row["img_filename"],
            "src_path":     row["src_path"],
            "class_label":  row["class_label"],
            "bias_label":   row["bias_label"],
        })

print(f"Immagini da processare: {len(rows)}")

prompt = "Is the person a male? (answer only with yes or no)."

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
        writer.writerow(["img_filename", "class_label", "bias_label", "raw_answer", "parsed_place"])

    for i, meta in enumerate(rows, start=1):
        if meta["img_filename"] in processed:
            continue

        raw    = run_inference(meta["src_path"], prompt)
        parsed = parse_gender(raw)

        writer.writerow([
            meta["img_filename"],
            meta["class_label"],
            meta["bias_label"],
            raw,
            parsed,
        ])

        if i % 100 == 0:
            f.flush()
            print(f"[{i}/{len(rows)}] processate")