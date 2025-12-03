#!/usr/bin/env python3

# ----------------- config -----------------
folderUser = 'c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/Cryo_Recon/Cryo_Recon'
# folderUser = 'c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/Cryo/'
MaskCutOffVal = 180
meshName = "mesh_colored"+str(MaskCutOffVal)+".ply"

# ----------------- imports -----------------
import os
import sys
import time
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import psutil
import nibabel as nib
import glob
from pathlib import Path
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label, marching_cubes
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from scipy.ndimage import map_coordinates
import pyvista as pv


# general fxns
def prompt_choice(prompt, options, default=None):
    print(prompt)
    for k, v in options.items():
        print(f"{k}: {v}")
    choice = input(f"Enter option [{default}]: ").strip()
    if choice == "" and default is not None:
        return default
    return choice

def load_images(folder, pattern = "*.tif*"):
    if pattern == "*.tif*": #aka, if we are inputting the original cryo files
        files = sorted(glob.glob(str(Path(folder) / pattern)))
        ii = -1 #so we can skip every other image (too much memory otherwise)
    else:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(pattern)])
        ii= 1
    if not files:
        raise RuntimeError("No relevant files found in folder: " + folder)
    print(f"Found {len(files)} images. Loading now")

    sample = imageio.imread(os.path.join(folder, files[0]))
    H, W = sample.shape[:2]

    stack_color = []
    jj = 1
    for f in tqdm(files, desc="Loading slices"):
        if jj == 1:
            path = os.path.join(folder, f)
            img = imageio.imread(path)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
            if img.shape[0] != H or img.shape[1] != W: #if any images are not correct size (some had an extra row/col)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            stack_color.append(img)
        jj = jj * ii #load every other image if it is original cryo
    stack = np.stack(stack_color, axis=0)
    return stack, files

# removing background
def bg_adaptive_keep_edge(slice_bgr, cutoff=220, min_area=500):

    gray = cv2.cvtColor(slice_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV) #apply binary threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8) #Find "regions" in image

    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, num_labels): #Go through "regions" and sets 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255 #turn regions larger than value given to fxn to white

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #closes holes present
    mask = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (11,11), 0) #blur
    mask = (mask*255).astype(np.uint8)
    mask3 = cv2.merge([mask, mask, mask])

    out = (slice_bgr.astype(np.float32) * (mask3.astype(np.float32)/255.0)).astype(np.uint8)
    return out

# registration
def rigid_register(stack_gray): #Only uses translation to align the slices (unlike affine)
                                                #Sufficient since the images are pretty well aligned
    N = stack_gray.shape[0] #num images in stack
    out = np.zeros_like(stack_gray, dtype=stack_gray.dtype)
    out[0] = stack_gray[0]
    shifts = [(0.0, 0.0)]
    ref = stack_gray[0].astype(np.float32)
    for i in tqdm(range(1, N), desc="Rigid registration"): #progress bar while going through (took a while and I thought it froze at first)
        moving = stack_gray[i].astype(np.float32)
        shift, error, phasediff = phase_cross_correlation(ref, moving, upsample_factor=10)
        moved = np.fft.ifftn(fourier_shift(np.fft.fftn(moving), shift)).real
        out[i] = np.clip(moved, 0, 255).astype(stack_gray.dtype)
        ref = 0.9*ref + 0.1*out[i].astype(np.float32)
        shifts.append(tuple(shift))
    return out, shifts


# More fxns (for volume)
def build_gray_stack_from_color(stack_color):
    N, H, W, _ = stack_color.shape
    gray = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(N):
        g = cv2.cvtColor(stack_color[i], cv2.COLOR_BGR2GRAY)
        gray[i] = g
    return gray

def faces_to_pyvista_faces(faces): #convert from skimage to pyvista formatting
    fv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces.astype(np.int32)])
    return fv.ravel()

# MAIN CODE
def main():
    print("=== CryoGAN Reconstruction Integrated (Memory Optimized) ===")
    folder = folderUser

    pattern_choice = prompt_choice("\nPng or Tif(f):", {
        "1":"png", 
        "2":"tif(f)"
        }, default="2")
    
    ds_choice = prompt_choice("\nDownsample BEFORE marching cubes (int factor):", {
        "1":"No downsample (1×)",
        "2":"2×",
        "4":"4×"
        }, default="2")
    ds_factor = int(ds_choice)

    mc_step_choice = prompt_choice("\nMarching cubes step size (1 best detail, higher = faster/simpler):", {
        "1":"High detail", 
        "2":"Medium", 
        "4":"Low (fast)"
        }, default="2")
    mc_step = int(mc_step_choice)

    pix = float(input("Pixel spacing (mm) [default 0.489]: ") or 0.489)
    slice_thick = float(input("Slice thickness (mm) [default 0.5]: ") or 0.5)


    if pattern_choice == "1":
        stack_color, files = load_images(folder,"*.png")
    else:
        stack_color, files = load_images(folder,"*.tif*")
    N, H, W, C = stack_color.shape
    print("Loaded stack shape:", stack_color.shape)

    # Background removal
    print("\nApplying background removal to each slice...")
    processed_color = []
    for i in tqdm(range(N), desc="Background removal"):
        s = stack_color[i]
        out = bg_adaptive_keep_edge(s, cutoff=220, min_area=500)
        processed_color.append(out)
    processed_color = np.stack(processed_color, axis=0)
    print("Background removal complete. Shape:", processed_color.shape)

    gray_stack = build_gray_stack_from_color(processed_color)

    # Registration
    print("\nRunning rigid registration")
    gray_reg, shifts = rigid_register(gray_stack)


    # Apply shifts to color. I could never get it to work sadly
    color_reg = np.zeros_like(processed_color)
    ref = gray_reg[0].astype(np.float32)
    color_reg[0] = processed_color[0]
    for i in tqdm(range(1, N), desc="Apply shifts to color"):
        shift, err, pd = phase_cross_correlation(ref, gray_reg[i].astype(np.float32), upsample_factor=10)
        M = np.array([[1,0,shift[1]],[0,1,shift[0]]], dtype=np.float32)
        moved = cv2.warpAffine(processed_color[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        color_reg[i] = moved
        ref = 0.9*ref + 0.1*gray_reg[i].astype(np.float32)


    #Down sampling
    if ds_factor > 1:
        new_shape = (max(1, gray_reg.shape[0]//ds_factor), 
                     max(1, gray_reg.shape[1]//ds_factor), 
                     max(1, gray_reg.shape[2]//ds_factor))
        
        gray_downsampled = np.zeros(new_shape, dtype=np.uint8)
        color_downsampled = np.zeros((new_shape[0], new_shape[1], new_shape[2], 3), dtype=np.uint8)
        
        for i in tqdm(range(gray_reg.shape[0]), desc="Downsampling slices"):
            idx = i // ds_factor
            if idx < new_shape[0]:
                gray_downsampled[idx] = cv2.resize(gray_reg[i], 
                                                   (new_shape[2], new_shape[1]), 
                                                   interpolation=cv2.INTER_AREA)
                color_downsampled[idx] = cv2.resize(color_reg[i], 
                                                    (new_shape[2], new_shape[1]), 
                                                    interpolation=cv2.INTER_AREA)
        del gray_reg  # running low on memory
        del color_reg
        del processed_color
        del gray_stack
        vol = gray_downsampled.astype(np.float32)
        vol_color = color_downsampled
    else:
        vol = gray_reg.astype(np.float32)
        vol_color = color_reg

    vol_min = vol.min()
    vol_max = vol.max()
    vol -= vol_min
    vol /= (vol_max - vol_min + 1e-9)

    # rescaling as needed
    if abs(slice_thick - pix) > 1e-6:
        scale_z = slice_thick / pix
        new_Z = int(round(vol.shape[0] * scale_z))
        if new_Z <= 0: new_Z = vol.shape[0]
        vol = resize(vol, (new_Z, vol.shape[1], vol.shape[2]), preserve_range=True)


    # keep big parts
    thr = threshold_otsu(vol)
    mask = vol > thr
    lbl = label(mask)
    sizes = np.bincount(lbl.ravel())
    min_size = int(0.005 * lbl.size)
    mask_clean = np.zeros_like(mask, dtype=bool)
    for i, s in enumerate(sizes):
        if i == 0:
            continue
        if s >= min_size:
            mask_clean[lbl == i] = True
    mask = mask_clean

    # convert bgr to rgb (not really used due to color failure)
    vol_color = vol_color[..., ::-1].astype(np.float32) / 255.0  # convert BGR->RGB

    # Marching cubes to extract surface from the mask
    verts, faces, normals, values = marching_cubes(mask.astype(np.float32), level=0.5, step_size=mc_step)
    faces_pv = faces_to_pyvista_faces(faces) #

    colors = [] #more color stuff that I could not get to work
    for c in range(3):
        col = map_coordinates(vol_color[..., c], [verts[:,0], verts[:,1], verts[:,2]], order=1, mode='nearest')
        colors.append(col)
    colors = np.stack(colors, axis=1)  # shape (num_verts,3)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    mesh_pv = pv.PolyData(verts, faces_pv)
    mesh_pv.point_data["red"] = colors_uint8[:, 0]
    mesh_pv.point_data["green"] = colors_uint8[:, 1]
    mesh_pv.point_data["blue"] = colors_uint8[:, 2]
    
    mesh_pv.clean(inplace=True)
    # meshName = "mesh_colored"+str(MaskCutOffVal)+".ply"
    mesh_pv.save(meshName)  # PLY supports vertex colors

    print("Saved mesh")
    if faces.shape[0] <= 1_000_000: #Could automatically run lower quality ones, but not final ones
        print("\nLaunching 3D preview...")
        plotter = pv.Plotter()
        # Combine R,G,B channels for display
        rgb_display = np.stack([mesh_pv.point_data["red"], 
                                mesh_pv.point_data["green"], 
                                mesh_pv.point_data["blue"]], axis=1)
        mesh_pv.point_data["RGB"] = rgb_display
        plotter.add_mesh(mesh_pv, show_edges=False, scalars="RGB", rgb=True)
        plotter.add_axes()
        plotter.show()


    print("\nDone")

if __name__ == "__main__":
    main()