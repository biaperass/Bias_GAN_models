import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os
import csv
from collections import Counter

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

image_dir = "stylegan3/generated_balance_waterbirds/waterbirds_256_0_70_50"
results = {}

filenames = sorted(
    f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

for fname in filenames:
    raw = run_inference(os.path.join(image_dir, fname), prompt)
    results[fname] = raw
    print(f"{fname}: raw='{raw}'")

# normalize / parse
normalized = {k: parse_place(v) for k, v in results.items()}

unparsed = [k for k, v in normalized.items() if v is None]
if unparsed:
    print(f"\nWARNING: {len(unparsed)} images could not be parsed into water/land:")
    for fname in unparsed:
        print(f"  {fname}: raw='{results[fname]}'")

# save csv (raw + parsed, so you can audit failures later)
with open("results_256_0_70_50_vito.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "raw_answer", "parsed_place"])
    for fname in filenames:
        writer.writerow([fname, results[fname], normalized[fname]])

# count
counts = Counter(normalized.values())
print(f"\nWater: {counts.get('yes', 0)}")
print(f"Land: {counts.get('no', 0)}")
print(f"Unparsed: {counts.get(None, 0)}")