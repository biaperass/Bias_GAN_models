import pandas as pd
import json

"""
This script reads the metadata.csv file for the waterbirds dataset and creates a JSON file with the image filenames and their corresponding labels (0 or 1). The output JSON file will have the following structure:

{
  "labels": [
    ["img_filename1.jpg", label1],
    ["img_filename2.jpg", label2],
    ...
  ]
}
"""


def convert_csv_to_json(csv_path, output_json):

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="img_id") # waterbirds 
    #df = df.sort_values(by="path") # bffhq

    labels_list = []
    for _, row in df.iterrows():
        label_int = int(row["y"]) # waterbirds
        #label_int = int(row["class_label"]) # bffhq
        labels_list.append([row["img_filename"], label_int]) # waterbirds
        #labels_list.append([row["path"], label_int]) # bffhq

    with open(output_json, "w") as f:
        json.dump({"labels": labels_list}, f, indent=2)
        
        
if __name__ == "__main__":
    csv_path_70 = "slib/datasets/data/waterbirds_rho_csv_CORRECT/waterbirds_0.70_fixed.csv"
    output_json_70 = "slib/datasets/data/waterbirds_rho_json/waterbirds_0.70_fixed.json"
    
    csv_path_80 = "slib/datasets/data/waterbirds_rho_csv_CORRECT/waterbirds_0.80_fixed.csv"
    output_json_80 = "slib/datasets/data/waterbirds_rho_json/waterbirds_0.80_fixed.json"
    
    csv_path_95 = "slib/datasets/data/waterbirds_rho_csv_CORRECT/waterbirds_0.95_fixed.csv"
    output_json_95 = "slib/datasets/data/waterbirds_rho_json/waterbirds_0.95_fixed.json"

    convert_csv_to_json(csv_path_70, output_json_70)
    convert_csv_to_json(csv_path_80, output_json_80)
    convert_csv_to_json(csv_path_95, output_json_95)
    
    