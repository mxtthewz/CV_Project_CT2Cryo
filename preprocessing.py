# This file takes the CT and Cryo tiff files, and converts them into numpy arrays,
# which are then passed on to preprocessingB.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Project folders
CT_FOLDER = r"c:/Users/micha/Downloads/MSAI/Computer vision/Project files 2/CT"
CRYO_FOLDER = r"c:/Users/micha/Downloads/MSAI/Computer vision/Project files 2/Cryo"

OUTPUT_ROOT = r"c:/Users/micha/Downloads/MSAI/Computer vision/Project files 2/processed_data"
IMAGE_SIZE = 256   # CycleGAN standard


# Helper functions
def load_tiff(path):
    """Opens TIFF images safely, handles grayscale or RGB."""
    img = Image.open(path)
    return img.convert("RGB")  # force 3-channel (CycleGAN requirement)


def resize(img):
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)


def normalize_to_minus1_1(img_np):
    """Map image uint8 [0,255] -> float [-1,1]."""
    img_np = img_np.astype(np.float32)
    return img_np / 127.5 - 1.0


def preprocess_folder(input_folder, output_folder):
    """
    Loads all TIFFs in a folder, resizes, normalizes, and saves .npy files.
    Skips files that are already processed.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".tiff")])
    print(f"\nProcessing {len(files)} TIFFs in: {input_folder}")

    for fname in tqdm(files):
        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(output_folder, out_name)

        # Skip if already processed
        if os.path.exists(out_path):
            continue

        try:
            # Load image
            img = load_tiff(os.path.join(input_folder, fname))

            # Resize
            img = resize(img)

            # Convert to numpy
            img_np = np.array(img)

            # Normalize to [-1,1]
            img_np = normalize_to_minus1_1(img_np)

            # Save as .npy
            np.save(out_path, img_np)

        except Exception as e:
            print(f"❌ Skipping {fname}: {e}")


# Run for both folders
if __name__ == "__main__":
    preprocess_folder(CT_FOLDER, os.path.join(OUTPUT_ROOT, "CT"))
    preprocess_folder(CRYO_FOLDER, os.path.join(OUTPUT_ROOT, "Cryo"))

    print("\n✅ Preprocessing complete!")
    print(f"Processed files saved in:\n{OUTPUT_ROOT}")
