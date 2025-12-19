#!/usr/bin/env python3
import sys
from pathlib import Path
try:
    import fitz
except Exception:
    fitz = None

if fitz is None:
    print('PyMuPDF not available')
    sys.exit(1)

def search(pdf_path, terms):
    doc = fitz.open(str(pdf_path))
    found = False
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text('text')
        low = text.lower()
        for term in terms:
            if term in low:
                found = True
                print(f'FOUND "{term}" on page {i+1}:')
                # print small context around each occurrence
                idx = 0
                while True:
                    idx = low.find(term, idx)
                    if idx == -1:
                        break
                    start = max(0, idx-80)
                    end = min(len(low), idx+len(term)+80)
                    ctx = low[start:end].replace('\n',' ')
                    print('...', ctx, '...')
                    idx += len(term)
    if not found:
        print('No terms found')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: search_pdf_terms.py ARXIV_ID')
        sys.exit(1)
    arxiv = sys.argv[1]
    pdf = Path('arxiv_pdf_cache') / f"{arxiv}.pdf"
    if not pdf.exists():
        print('PDF not found:', pdf)
        sys.exit(1)
    terms = ['decomposition', 'decompose', 'cnot']
    search(pdf, terms)
