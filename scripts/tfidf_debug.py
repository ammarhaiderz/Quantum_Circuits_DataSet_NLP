#!/usr/bin/env python3
"""TF-IDF debug runner for one PDF.

Usage:
  python scripts/tfidf_debug.py --arxiv-id 2001.01234
  python scripts/tfidf_debug.py --arxiv-id 2001.01234 --caption "Decomposition of CNOT"

The script will try to read a caption from `data/circuits.jsonl` for the given
arXiv id if `--caption` is not provided.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import textwrap

try:
    import fitz
except Exception:
    fitz = None

from config.settings import PDF_CACHE_DIR

# Import helpers from the project module
from core import circuit_store as cs
from collections import Counter


def find_caption_from_jsonl(arxiv_id: str) -> str | None:
    p = Path('data') / 'circuits.jsonl'
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('arxiv_id') == arxiv_id:
                descs = rec.get('descriptions') or []
                if descs:
                    return descs[0]
    return None


def build_candidates_from_doc(doc) -> list[tuple[int, str]]:
    """Mimic `find_caption_page_in_pdf` candidate windows generation."""
    anchor_re = cs.re.compile(r"(?:Fig\.?|Figure)\s*\d+\b", cs.re.IGNORECASE)
    stop_re = cs.re.compile(r"\b(?:Fig\.?|Figure|Table)\s*\d+\b", cs.re.IGNORECASE)

    page_candidates = []
    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
        except Exception:
            continue

        text_norm = cs._normalize_text(page_text)
        anchors = list(anchor_re.finditer(text_norm))
        if not anchors:
            page_candidates.append((page_num, text_norm))
            continue

        for m in anchors:
            start = m.end()
            window = text_norm[start:start + 500]
            s = stop_re.search(window)
            if s:
                window = window[:s.start()]
            page_candidates.append((page_num, window))

    return page_candidates


def run_debug(arxiv_id: str, caption: str | None, top_k: int = 5):
    if fitz is None:
        print("PyMuPDF (fitz) not available. Install it to run this debug.")
        return 1

    pdf_path = Path(PDF_CACHE_DIR) / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        print(f"PDF not found at: {pdf_path}")
        return 1

    if caption is None:
        caption = find_caption_from_jsonl(arxiv_id)
        if caption:
            print("Found caption in data/circuits.jsonl:")
        else:
            print("No caption provided and none found in data/circuits.jsonl. Provide --caption.")
            return 1

    print("\n=== INPUT CAPTION ===")
    print(caption)

    clean_caption = cs._clean_caption_for_search(caption)
    print("\n=== CLEANED CAPTION ===")
    print(clean_caption)

    qtokens = cs._tokenize_for_comparison(clean_caption, is_latex=True, min_len=1)
    print("\n=== CAPTION TOKENS (post-tokenize_for_comparison) ===")
    print(sorted(list(qtokens))[:200])

    # open pdf and build candidates
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"Failed to open PDF: {e}")
        return 1

    candidates = build_candidates_from_doc(doc)
    print(f"\nBuilt {len(candidates)} candidate windows from PDF pages.")

    # show a few page previews and tokenizations
    for pnum, win in candidates[:10]:
        toks = cs._tokenize_for_comparison(win, is_latex=False, min_len=1)
        preview = win[:200].replace('\n', ' ')
        print(f"\n--- Page {pnum+1} preview ---")
        print(preview)
        print("Tokens (first 40):", sorted(list(toks))[:40])

    # prepare counters for TF-IDF
    docs = [Counter(cs._tokenize_for_comparison(win, is_latex=False, min_len=1)) for _, win in candidates]
    pages = [p for p, _ in candidates]
    qcounter = Counter(cs._tokenize_for_comparison(clean_caption, is_latex=True, min_len=1))

    if not docs:
        print("No document candidates to score.")
        return 1

    scores = cs._compute_tfidf_similarity(qcounter, docs)

    # pair scores with pages and previews
    scored = list(zip(scores, pages, [w for _, w in candidates]))
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

    print("\n=== TOP SCORES ===")
    for score, page, window in scored_sorted[:top_k]:
        print(f"score={score:.4f} page={page+1}")
        print(textwrap.fill(window[:500].replace('\n', ' '), width=100))
        print('-' * 40)

    doc.close()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arxiv-id', required=True)
    p.add_argument('--caption', required=False)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    exit(run_debug(args.arxiv_id, args.caption, args.top_k))


if __name__ == '__main__':
    main()
