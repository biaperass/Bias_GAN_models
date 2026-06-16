import pandas as pd
original_df = pd.read_csv('data/waterbirds/waterbird_complete95_forest2water2/metadata.csv')
train_df = original_df.query('split == 0').copy()

# downsample logic from _resample_pool
n_minority_lb = len(train_df[(train_df['y']==0) & (train_df['place']==1)])
n_minority_wb = len(train_df[(train_df['y']==1) & (train_df['place']==0)])
print(f'n_minority_lb (y=0,p=1): {n_minority_lb}')
print(f'n_minority_wb (y=1,p=0): {n_minority_wb}')

for rho in [0.70, 0.80, 0.95]:
    counts = {
        (0,0): int(n_minority_lb * rho / (1.0 - rho)),
        (0,1): n_minority_lb,
        (1,0): n_minority_wb,
        (1,1): int(n_minority_wb * rho / (1.0 - rho)),
    }
    total = sum(counts.values())
    print(f'rho={rho:.2f}: counts={counts}, total={total}')