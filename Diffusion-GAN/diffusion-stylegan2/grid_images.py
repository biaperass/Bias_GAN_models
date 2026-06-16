import os
from PIL import Image
import matplotlib.pyplot as plt

def plot_grid_from_folder(folder_path, file_name):
    grid_size=10
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:grid_size * grid_size] # grid 10x10

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20,20))

    for i, ax in enumerate(axes.flat):
        if i < len(image_files):
            img_path = os.path.join(folder_path, image_files[i])
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(file_name)


plot_grid_from_folder("generated_balance/bffhq_0_95", "generated_balance/bffhq_grid/bffhq_grid_0_95.png")
plot_grid_from_folder("generated_balance/bffhq_1_95", "generated_balance/bffhq_grid/bffhq_grid_1_95.png")

plot_grid_from_folder("generated_balance/bffhq_0_95_75", "generated_balance/bffhq_grid/bffhq_grid_0_95_75.png")
plot_grid_from_folder("generated_balance/bffhq_1_95_75", "generated_balance/bffhq_grid/bffhq_grid_1_95_75.png")

plot_grid_from_folder("generated_balance/bffhq_0_95_50", "generated_balance/bffhq_grid/bffhq_grid_0_95_50.png")
plot_grid_from_folder("generated_balance/bffhq_1_95_50", "generated_balance/bffhq_grid/bffhq_grid_1_95_50.png")
