# Takes numpy arrays from preprocessing.py and converts them into pngs to be used
# by the GAN.
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Settings
PROCESSED_ROOT = r"c:/Users/micha/Downloads/MSAI/Computer vision/Project files 2/processed_data"
CT_FOLDER = os.path.join(PROCESSED_ROOT, "CT")
CRYO_FOLDER = os.path.join(PROCESSED_ROOT, "Cryo")

OUTPUT_ROOT = r"c:/Users/micha/Downloads/MSAI/Computer vision/Project files 2/datasets/visible"
IMAGE_SIZE = 256
TRAIN_RATIO = 0.8

SUBFOLDERS = ["trainA", "trainB", "testA", "testB"]

# Helper functions
def ensure_folders(root, subfolders):
    for f in subfolders:
        os.makedirs(os.path.join(root, f), exist_ok=True)

def sorted_npy_files(folder, reverse=False):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".npy")]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0), reverse=reverse)
    return files

def save_image(arr, path):
    # Convert from [-1,1] to [0,255] for PNG
    arr_img = ((arr + 1.0) * 127.5).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr_img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    img.save(path)

# Main
def prepare_cycle_gan(ct_folder, cryo_folder, output_root, train_ratio=0.8):
    ensure_folders(output_root, SUBFOLDERS)

    # CT order is reversed
    ct_files = sorted_npy_files(ct_folder, reverse=True)
    ct_arrays = [np.load(os.path.join(ct_folder, f)) for f in tqdm(ct_files, desc="Loading CT")]

    # Cryo order is normal
    cryo_files = sorted_npy_files(cryo_folder, reverse=False)
    cryo_arrays = [np.load(os.path.join(cryo_folder, f)) for f in tqdm(cryo_files, desc="Loading Cryo")]

    # Train/Test split
    ct_cutoff = int(len(ct_arrays) * train_ratio)
    cryo_cutoff = int(len(cryo_arrays) * train_ratio)

    # Save CT
    for i, arr in enumerate(tqdm(ct_arrays, desc="Saving CT")):
        subset = "trainA" if i < ct_cutoff else "testA"
        save_path = os.path.join(output_root, subset, f"{i:04d}.png")
        if os.path.exists(save_path):
            continue  # skip already saved PNG
        save_image(arr, save_path)

    # Save Cryo
    for i, arr in enumerate(tqdm(cryo_arrays, desc="Saving Cryo")):
        subset = "trainB" if i < cryo_cutoff else "testB"
        save_path = os.path.join(output_root, subset, f"{i:04d}.png")
        if os.path.exists(save_path):
            continue  # skip already saved PNG
        save_image(arr, save_path)

    print("\n✅ CycleGAN dataset ready!")
    print(f"TrainA: {ct_cutoff}, TestA: {len(ct_arrays)-ct_cutoff}")
    print(f"TrainB: {cryo_cutoff}, TestB: {len(cryo_arrays)-cryo_cutoff}")


# Run
if __name__ == "__main__":
    prepare_cycle_gan(CT_FOLDER, CRYO_FOLDER, OUTPUT_ROOT, TRAIN_RATIO)

