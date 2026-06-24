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

    is_man = "yes" in text
    is_female = "no" in text

    if is_man and not is_female:
        return "yes"
    if is_female and not is_man:
        return "no"
    return None  


# --- batch over the generated samples ---

prompt = "Is the person a male? (answer only with yes or no)." 


image_dir = f"Diffusion-GAN/diffusion-stylegan2/generated_bffhq_10k/bffhq_rho{rho}/bffhq_trunc{truncation:.2f}_class{cls}"
results = {}

filenames = sorted(
    f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

output_csv = f"results_diffusion_10k_bffhq_rho{rho}_trunc{truncation:.2f}_class{cls}_cuda.csv"

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