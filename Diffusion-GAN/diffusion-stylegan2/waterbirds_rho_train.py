import pandas as pd
import os
import shutil

def images_from_csv(input_csv, input_image_root, output_image_root, dry_run=False):
    
    df = pd.read_csv(input_csv)
    df = df.sort_values(by="img_id")
    print(f"Numero totale immagini nel CSV originale: {len(df)}")

    # creo nuova cartella per le immagini filtrate
    os.makedirs(output_image_root, exist_ok=True)

    # eliminare immagini non presenti nel nuovo csv 
    valid_paths = set(df["img_filename"].values)
    print(f"Numero immagini nel CSV: {len(valid_paths)}")

    count = 0

    for rel_path in valid_paths:
        src_path = os.path.join(input_image_root, rel_path)
        dst_path = os.path.join(output_image_root, rel_path)

        if not os.path.exists(src_path):
            print(f"[WARNING] File non trovato: {src_path}")
            continue

        if dry_run:
            count += 1
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            count += 1

    print(f"Totale immagini copiate: {count}")


if __name__ == "__main__":
    input_image_root = "./data/waterbirds/waterbird_complete95_forest2water2"

    input_csv_70 = "./data/waterbirds_rho_csv/waterbirds_0.70_fixed.csv"
    output_image_root_70 = "./data/waterbirds_rho_images/waterbirds_70"
    images_from_csv(input_csv_70, input_image_root, output_image_root_70, dry_run=False)

    input_csv_80 = "./data/waterbirds_rho_csv/waterbirds_0.80_fixed.csv"
    output_image_root_80 = "./data/waterbirds_rho_images/waterbirds_80"
    images_from_csv(input_csv_80, input_image_root, output_image_root_80, dry_run=False)

    input_csv_95 = "./data/waterbirds_rho_csv/waterbirds_0.95_fixed.csv"
    output_image_root_95 = "./data/waterbirds_rho_images/waterbirds_95"
    images_from_csv(input_csv_95, input_image_root, output_image_root_95, dry_run=False)