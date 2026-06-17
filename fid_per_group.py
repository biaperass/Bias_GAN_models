import pandas as pd
from pathlib import Path
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device).eval()

# (y, place) -> category code. y: 0=landbird, 1=waterbird. place: 0=land, 1=water.
CATEGORIES = {
    '00': (0, 0),  # bias-aligned
    '11': (1, 1),  # bias-aligned
    '01': (0, 1),  # bias-conflicting
    '10': (1, 0),  # bias-conflicting
}

MIN_SAMPLES = 2048  # below this, sigma is rank-deficient; FID becomes unreliable

def filtered_paths(df, base_dir, y_val, place_val, split_val=None):
    sub = df[(df['y'] == y_val) & (df['place'] == place_val)]
    if split_val is not None:
        sub = sub[sub['split'] == split_val]
    return [str(Path(base_dir) / fn) for fn in sub['img_filename']]

def fid_per_category(real_csv, real_base, gen_csv, gen_base, real_split=0, batch_size=50):
    real_df = pd.read_csv(real_csv)
    gen_df = pd.read_csv(gen_csv)
    results = {}

    for code, (y_val, place_val) in CATEGORIES.items():
        real_files = filtered_paths(real_df, real_base, y_val, place_val, split_val=real_split)
        gen_files = filtered_paths(gen_df, gen_base, y_val, place_val, split_val=None)

        n_real, n_gen = len(real_files), len(gen_files)
        flag = n_real < MIN_SAMPLES or n_gen < MIN_SAMPLES

        if n_real == 0 or n_gen == 0:
            results[code] = {'fid': None, 'n_real': n_real, 'n_gen': n_gen, 'note': 'empty subset'}
            continue

        bs_real = min(batch_size, n_real)
        bs_gen = min(batch_size, n_gen)

        mu_r, sigma_r = calculate_activation_statistics(real_files, model, bs_real, dims, device)
        mu_g, sigma_g = calculate_activation_statistics(gen_files, model, bs_gen, dims, device)

        fid = calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        results[code] = {
            'fid': fid, 'n_real': n_real, 'n_gen': n_gen,
            'rank_deficient': flag
        }

    return results

if __name__ == '__main__':
    res = fid_per_category(
        real_csv='Bias_GAN_models/projected-gan/data/waterbirds/metadata.csv',
        real_base='Bias_GAN_models/projected-gan/data/waterbirds/images',
        #gen_csv='generated_metadata_psi_0.5.csv',
        #gen_base='Bias_GAN_models/projected-gan/data/waterbirds/waterbirds_generated_images',
    )
    for code, r in res.items():
        print(code, r)