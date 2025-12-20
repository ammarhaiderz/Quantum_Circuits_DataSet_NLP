import fitz  # PyMuPDF
import os
from pathlib import Path
from PIL import Image
import ghostscript
from tqdm import tqdm


RAW_DIR = "images_test_50_preproc_cached_"        # Folder containing mixed files
OUT_DIR = "clean_images_50_preproc_cached"      # Folder where PNGs will be saved
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Remove transparency and replace with WHITE background
# ----------------------------------------------------------------------
def remove_transparency(im, bg_color=(255, 255, 255)):
    """
    Converts images with transparency into RGB images with a white background.
    """
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert("RGBA").split()[-1]
        bg = Image.new("RGB", im.size, bg_color)
        bg.paste(im, mask=alpha)
        return bg
    return im.convert("RGB")


# ----------------------------------------------------------------------
# PDF -> PNG (PyMuPDF)
# ----------------------------------------------------------------------
def convert_pdf(path, out_dir):
    doc = fitz.open(path)
    base = Path(path).stem

    for page_num in range(len(doc)):
        png_path = os.path.join(out_dir, f"{base}_page_{page_num+1}.png")

        pix = doc.load_page(page_num).get_pixmap(dpi=200)
        pix.save(png_path)

        # Remove transparency
        img = Image.open(png_path)
        img = remove_transparency(img)
        img.save(png_path, "PNG")


# ----------------------------------------------------------------------
# EPS -> PNG (Ghostscript)
# ----------------------------------------------------------------------
def convert_eps(path, out_dir):
    base = Path(path).stem
    temp_path = os.path.join(out_dir, base + "_tmp.png")
    final_path = os.path.join(out_dir, base + ".png")

    args = [
        "gs",
        "-dNOPAUSE",
        "-dBATCH",
        "-dEPSCrop",
        "-sDEVICE=pngalpha",
        "-r200",
        f"-sOutputFile={temp_path}",
        str(path),
    ]

    ghostscript.Ghostscript(*args)

    # Remove transparency
    img = Image.open(temp_path)
    img = remove_transparency(img)
    img.save(final_path, "PNG")

    os.remove(temp_path)


# ----------------------------------------------------------------------
# Normal Images (JPG/JPEG/PNG) -> PNG (Pillow)
# ----------------------------------------------------------------------
def convert_image(path, out_dir):
    base = Path(path).stem
    out_path = os.path.join(out_dir, base + ".png")

    img = Image.open(path)
    img = remove_transparency(img)
    img.save(out_path, "PNG")


# ----------------------------------------------------------------------
# Supported extensions
# ----------------------------------------------------------------------
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_EPS = {".eps"}


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
files = list(Path(RAW_DIR).glob("*"))

for file in tqdm(files, desc="Processing files"):
    ext = file.suffix.lower()

    try:
        if ext in SUPPORTED_PDF:
            convert_pdf(file, OUT_DIR)

        elif ext in SUPPORTED_EPS:
            convert_eps(file, OUT_DIR)

        elif ext in SUPPORTED_IMAGES:
            convert_image(file, OUT_DIR)

        else:
            print(f"Skipping unsupported file: {file.name}")

    except Exception as e:
        print(f"Error processing {file.name}: {e}")


print("\n[OK] Conversion finished — all images are in:", OUT_DIR)
