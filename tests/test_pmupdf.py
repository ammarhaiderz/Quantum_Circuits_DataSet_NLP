import fitz  # PyMuPDF
from pathlib import Path
import sys
import os


# =========================
# CONFIG
# =========================

PDF_PATH = "2509.13247.pdf"   # input PDF
OUT_DIR = "extracted_media"

MIN_IMAGE_SIZE = 100  # px (skip tiny icons)
DRAWING_DPI = 300     # resolution for drawings


# =========================
# UTILS
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# IMAGE EXTRACTION
# =========================

def extract_all_images(doc, out_dir):
    print("\n[+] Extracting embedded images")

    seen_xrefs = set()
    count = 0

    for page_number, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]

            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            pix = fitz.Pixmap(doc, xref)

            # Convert CMYK / others -> RGB
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
                continue

            out_path = os.path.join(
                out_dir,
                f"image_p{page_number}_xref{xref}.png"
            )

            pix.save(out_path)
            count += 1

    print(f"    [OK] {count} images extracted")


# =========================
# DRAWING EXTRACTION
# =========================

def extract_all_drawings(doc, out_dir):
    print("\n[+] Extracting vector drawings (rendered)")

    count = 0

    for page_number, page in enumerate(doc, start=1):
        drawings = page.get_drawings()

        if not drawings:
            continue

        # Render FULL PAGE because drawings are not separable
        mat = fitz.Matrix(DRAWING_DPI / 72, DRAWING_DPI / 72)
        pix = page.get_pixmap(matrix=mat)

        out_path = os.path.join(
            out_dir,
            f"drawing_page_{page_number}.png"
        )

        pix.save(out_path)
        count += 1

    print(f"    [OK] {count} drawing pages rendered")


# =========================
# MAIN
# =========================

def main():
    if not Path(PDF_PATH).exists():
        print(f"[ERROR] PDF not found: {PDF_PATH}")
        sys.exit(1)

    ensure_dir(OUT_DIR)

    doc = fitz.open(PDF_PATH)

    extract_all_images(doc, OUT_DIR)
    extract_all_drawings(doc, OUT_DIR)

    print("\n[OK] Done")
    print(f"[INFO] Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
