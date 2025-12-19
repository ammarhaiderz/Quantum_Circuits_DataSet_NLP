import fitz
import sys
from pathlib import Path

PID='2510.04993'
PDF_CACHE='arxiv_pdf_cache'
pdf_path = Path(PDF_CACHE) / f"{PID}.pdf"
if not pdf_path.exists():
    print(f"PDF not found at {pdf_path}")
    sys.exit(2)

print(f"Opening PDF: {pdf_path}\n")
try:
    doc = fitz.open(str(pdf_path))
except Exception as e:
    print(f"Failed to open PDF: {e}")
    sys.exit(3)

# Print number of pages
print(f"Pages: {doc.page_count}\n")

# Print extracted text for first N pages (showing lines around potential anchors)
N = doc.page_count
for pnum in range(N):
    page = doc.load_page(pnum)
    text = page.get_text("text")
    print(f"\n--- PAGE {pnum+1} ---\n")
    # print first 800 chars and also show lines containing 'Figure' or 'Fig.'
    snippet = text[:800]
    print(snippet)
    print('\nLines with Figure/Fig:')
    for i, line in enumerate(text.splitlines(), 1):
        if 'Figure' in line or 'Fig.' in line or 'Fig ' in line or 'Fig:' in line:
            print(f" {i:4d}: {line}")
    print('\n' + ('='*60) + '\n')

try:
    doc.close()
except Exception:
    pass
