from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.eval.run_llava import eval_model
import os
import torch
import copy
from PIL import Image
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import csv
from collections import Counter


model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

PROMPT = "Is the background of this image a water scene (ocean, lake, river) or a land scene (forest, field, grass)? Answer with one word: water or land."

def run_inference(image_path: str, prompt: str) -> str:
    conv = copy.deepcopy(conv_templates["llava_v1"])
    conv.messages = list(conv.messages)

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
             #image_sizes=[image.size],
            do_sample=False,
            num_beams=1,
            max_new_tokens=5,
        )
        
    #print("raw output_ids:", output_ids)
    #print("input length:", input_ids.shape[1])
    #print("output length:", output_ids.shape[1])
    #print("new tokens:", output_ids[0, input_ids.shape[1]:])

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

def caption_directory(image_dir: str, y_value: int, id_start: int = 0):
    rows = []
    img_id = id_start
    unparsed = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        raw = run_inference(os.path.join(image_dir, fname), PROMPT)
        place = parse_place(raw)
        if place is None:
            unparsed.append((fname, raw))
        rows.append({
            "img_id": img_id,
            "img_filename": fname,      # relative to image_dir
            "y": y_value,
            "split": -1,                # sentinel: not a real train/val/test split
            "place": place,             # None if unparsed — filter before using
            "place_filename": "",       # no Places-source provenance for generated images
        })
        img_id += 1

    if unparsed:
        print(f"[{image_dir}] {len(unparsed)}/{len(rows)} unparsed VLM answers:")
        for fname, raw in unparsed[:10]:
            print(f"  {fname}: {raw!r}")
    return rows, img_id

GENERATION_RUNS = [
    ("waterbirds_256_0_95", 0),
    ("waterbirds_256_1_95", 1),
]

all_rows = []
next_id = 0
for image_dir, y_value in GENERATION_RUNS:
    rows, next_id = caption_directory(image_dir, y_value, id_start=next_id)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows, columns=["img_id", "img_filename", "y", "split", "place", "place_filename"])
df.to_csv("generated_metadata.csv", index=False)

print(f"Total rows: {len(df)}, unparsed/missing place: {df['place'].isna().sum()}")
print(df.groupby(["y", "place"], dropna=False).size())

# --- batch over the generated samples ---
"""
# image_file = "test_images/seed0001.png" # quick test, then remove 
# print(eval_model(args)) # quick test, then remove 
image_dir = "waterbirds_256_0_95"
results = {}

for fname in os.listdir(image_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        results[fname] = run_inference(os.path.join(image_dir, fname), PROMPT)


print(results)

# normalize
normalized = {k: v.strip().split(",")[0].strip() for k, v in results.items()}

# save csv
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "answer"])
    for fname, answer in normalized.items():
        writer.writerow([fname, answer])

# count
counts = Counter(normalized.values())
print(f"Land: {counts['Land']}")
print(f"Water: {counts['Water']}")
"""

