import torch
import numpy as np
import dnnlib
import legacy
import PIL.Image
import os
from torchvision import transforms
from PIL import Image
from scipy.stats import ttest_ind


device = "cuda"

network_pkl = "output_runs/00000-stylegan2-waterbirds_512-gpus1-batch16-gamma8/network-snapshot-002200.pkl"
land_bias_aligned = "dataset_aligned_conflicting/landbird/bias_aligned"
land_bias_conflicting = "dataset_aligned_conflicting/landbird/bias_conflicting"
water_bias_aligned = "dataset_aligned_conflicting/waterbird/bias_aligned"
water_bias_conflicting = "dataset_aligned_conflicting/waterbird/bias_conflicting"

# -------------------------------------
with open(network_pkl, "rb") as f:
    data = legacy.load_network_pkl(f)
D = data["D"].to(device)
D.eval()

# print(D.img_resolution)  # 512
# print(D.c_dim) # 2 classi (tutto corretto)


transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
])

def compute_real_prob(folder_path, label):
    count_real = 0
    total = 0

    c = torch.zeros(1, D.c_dim, device=device)
    c[0, label] = 1.0
    
    for fname in os.listdir(folder_path):
        img_path = os.path.join(folder_path, fname)
                
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = D(img, c)
            prob = torch.sigmoid(logit)
            # probs.append(prob.item())
            
            if prob.item() > 0.5:
                count_real += 1
            total += 1
    
    return count_real, total, count_real / total if total > 0 else 0


land_aligned = compute_real_prob(land_bias_aligned, 0)
land_conflicting = compute_real_prob(land_bias_conflicting, 0)

water_aligned = compute_real_prob(water_bias_aligned, 1)
water_conflicting   = compute_real_prob(water_bias_conflicting, 1)


print("land bias aligned:")
print(f"{land_aligned[0]} / {land_aligned[1]}  ({land_aligned[2]*100:.2f}%)")

print("\nland bias conflicting:")
print(f"{land_conflicting[0]} / {land_conflicting[1]}  ({land_conflicting[2]*100:.2f}%)")

print("\nwater bias aligned:")
print(f"{water_aligned[0]} / {water_aligned[1]}  ({water_aligned[2]*100:.2f}%)")

print("\nwater bias conflicting:")
print(f"{water_conflicting[0]} / {water_conflicting[1]}  ({water_conflicting[2]*100:.2f}%)")