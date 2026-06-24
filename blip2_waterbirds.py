import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os
import csv
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rho", type=int, required=True)
parser.add_argument("--truncation", type=float, required=True)
parser.add_argument("--class", dest="cls", type=int, required=True)

args = parser.parse_args()

rho = args.rho
truncation = args.truncation
cls = args.cls

model_path = "Salesforce/blip2-flan-t5-xl"

processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model = model.float()
model.eval()


def run_inference(image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=5,
        )

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def parse_place(raw_answer: str):
    text = raw_answer.strip().lower()

    has_water = "yes" in text
    has_land = "no" in text

    if has_water and not has_land:
        return "yes"
    if has_land and not has_water:
        return "no"
    return None  


# --- batch over the generated samples ---

prompt = "Is the background predominantly water? Answer with one word: yes or no. Answer:" 

image_dir = f"stylegan3/generated_10k_rho{rho}/waterbirds_trunc{truncation}_class{cls}"
results = {}

filenames = sorted(
    f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

output_csv = f"results_10k_waterbirds_rho{rho}_trunc{truncation}_class{cls}.csv"

processed = set()

if os.path.exists(output_csv):
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["filename"])

mode = "a" if os.path.exists(output_csv) else "w"

with open(output_csv, mode, newline="") as f:
    writer = csv.writer(f)

    if mode == "w":
        writer.writerow(["filename", "raw_answer", "parsed_place"])

    for i, fname in enumerate(filenames, start=1):

        if fname in processed:
            continue

        raw = run_inference(os.path.join(image_dir, fname), prompt)
        parsed = parse_place(raw)

        writer.writerow([fname, raw, parsed])

        if i % 500 == 0:
            f.flush()
            print(f"Salvate {i} immagini")