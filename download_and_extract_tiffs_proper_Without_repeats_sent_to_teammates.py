#Change main section based on if ct vs cryo

import os
import gzip
import shutil
import requests
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Fxns

def download_single_file(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(save_path), leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except requests.HTTPError:
        print(f"{os.path.basename(save_path)} not found on server.")
        return False
    except Exception as e:
        print(f"Error downloading {os.path.basename(save_path)}: {e}")
        return False

def extract_gz_file(gz_path):
    tiff_path = gz_path[:-3]
    if os.path.exists(tiff_path):
        if os.path.exists(gz_path):
            os.remove(gz_path)
        return True
    with gzip.open(gz_path, "rb") as f_in:
        with open(tiff_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    return True


def get_existing_numbers(output_folder):
    files = [f for f in os.listdir(output_folder) if re.match(r'\d+\.tiff(\.gz)?$', f)]
    numbers = set()
    for f in files:
        match = re.match(r'0*(\d+)\.tiff', f)
        if match:
            numbers.add(int(match.group(1)))
    return numbers

def download_tiffs_incremental(page_url, output_folder, batch_size=50, max_workers=5):
    os.makedirs(output_folder, exist_ok=True)
    existing_numbers = get_existing_numbers(output_folder)

    # Fetch page
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"Failed to access page")
        return

    # Find all .tiff.gz links
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    all_links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        match = re.match(r'0*(\d+)\.tiff\.gz$', href)#####################################################################################################
        # match = re.match(r'0*(\d+)\.tif\.gz$', href)
        if match:
            num = int(match.group(1))
            if num not in existing_numbers:
                full_url = urljoin(page_url, href)
                all_links.append((num, full_url))

    if not all_links:
        print("Nothing new to download.")
        return

    # Sort and select batch
    all_links.sort(key=lambda x: x[0])
    batch_links = all_links[:batch_size]
    print(f"➡ Downloading {len(batch_links)} new files: {batch_links[0][0]:04} - {batch_links[-1][0]:04}")

    # Download batch
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for num, link in batch_links:
            save_name = f"{num:04}.tiff.gz"
            save_path = os.path.join(output_folder, save_name)
            tasks.append(executor.submit(download_single_file, link, save_path))
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading batch"):
            future.result()

    # Extract .gz files
    gz_files = [f for f in os.listdir(output_folder) if f.lower().endswith(".gz")]
    if gz_files:
        print("\n Extracting downloaded .gz files...")
        for f in tqdm(gz_files, desc="Extracting", unit="file"):
            extract_gz_file(os.path.join(output_folder, f))

    print(f"\n Files saved in:\n   {output_folder}")

# -------------------------------
# Run Script
# -------------------------------
if __name__ == "__main__":
    download_tiffs_incremental(
        page_url="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/cryo/tiff/index.html",
        output_folder=r"c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/Cryo",
        batch_size=500,
        max_workers=5
    )

# if __name__ == "__main__":
#     download_tiffs_incremental(
#         page_url="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_tiffs/CAT-tiff/index.html",
#         output_folder=r"c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/CT",
#         batch_size=200,
#         max_workers=5
#     )



# if __name__ == "__main__":
#     print("start")
#     download_tiffs_incremental(
#         # "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_tiffs/CAT-tiff/index.html",
#         # r"c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/CT",
#         "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/cryo/tiff/index.html",
#         r"c:/Users/Sid/Downloads/MSAI/Computer vision/Project files 2/Cryo",
#         batch_size=50,
#         max_workers=5
#     )
#     print("stop")
