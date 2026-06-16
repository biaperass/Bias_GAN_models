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


plot_grid_from_folder("generated_balance/bffhq_256_0_70", "generated_balance/bffhq_grid/bffhq_256_0_70.png")
plot_grid_from_folder("generated_balance/bffhq_256_1_70", "generated_balance/bffhq_grid/bffhq_256_1_70.png")

plot_grid_from_folder("generated_balance/bffhq_256_0_70_75", "generated_balance/bffhq_grid/bffhq_256_0_70_75.png")
plot_grid_from_folder("generated_balance/bffhq_256_1_70_75", "generated_balance/bffhq_grid/bffhq_256_1_70_75.png")

plot_grid_from_folder("generated_balance/bffhq_256_0_70_50", "generated_balance/bffhq_grid/bffhq_256_0_70_50.png")
plot_grid_from_folder("generated_balance/bffhq_256_1_70_50", "generated_balance/bffhq_grid/bffhq_256_1_70_50.png")

