import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

# Configuration
BASE_DIR = "cartoonset100k/cartoonset100k"
OUTLINE_BASE_DIR = "cartoonset100k_outlines"
TENSOR_BASE_DIR = "cartoonset100k_tensors"
NUM_FOLDERS = 10

# Image transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Outline generation function using Canny edge detection
def generate_outline(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges).convert("L")

# Process each folder (0–9)
for folder in range(NUM_FOLDERS):
    img_dir = os.path.join(BASE_DIR, str(folder))
    outline_dir = os.path.join(OUTLINE_BASE_DIR, str(folder))
    tensor_dir = os.path.join(TENSOR_BASE_DIR, str(folder))
    os.makedirs(outline_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)
    
    print(f"Processing folder {folder}...")
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        
        # Generate and save outline
        outline = generate_outline(img)
        outline_file = img_file.replace(".png", "_outline.png")
        outline.save(os.path.join(outline_dir, outline_file))
        
        # Apply transforms
        img_tensor = TRANSFORM(img)
        outline_tensor = TRANSFORM(outline)
        
        # Load attributes
        csv_file = img_file.replace(".png", ".csv")
        csv_path = os.path.join(img_dir, csv_file)
        if os.path.exists(csv_path):
            attrs = torch.tensor(pd.read_csv(csv_path, header=None).iloc[:, 1].values.astype("int64"))
        else:
            print(f"Warning: {csv_path} not found, skipping {img_file}")
            continue
        
        # Save tensor
        tensor_file = img_file.replace(".png", ".pt")
        torch.save(
            {"img": img_tensor, "outline": outline_tensor, "attrs": attrs},
            os.path.join(tensor_dir, tensor_file)
        )
    
    print(f"Folder {folder} completed: {len(img_files)} files processed.")

print("Conversion complete!")