Suppose you have a main folder, say 'data', and inside this folder are three different folders, e.g. 'folder1', 'folder2', and 'folder3'. Each folder contains images.

Here’s a Python script to display one image from each folder using matplotlib and PIL:

import os
from PIL import Image
import matplotlib.pyplot as plt

base_path = 'data'  # The main folder containing three folders
folders = ['folder1', 'folder2', 'folder3']  # list your folder names here

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, folder in enumerate(folders):
    folder_path = os.path.join(base_path, folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_path = os.path.join(folder_path, image_files[0])  # Take the first image from each folder
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(folder)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

Instructions:
- Change 'data', 'folder1', etc. to your actual folder names.
- You can modify the script to show more images or iterate differently if you like.
- Install required libraries if missing: pip install matplotlib pillow