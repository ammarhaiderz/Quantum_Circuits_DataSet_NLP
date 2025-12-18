import json
from pathlib import Path
import re
from difflib import SequenceMatcher

try:
    import fitz
except Exception:
    fitz = None

from config.settings import PDF_CACHE_DIR

DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = DATA_DIR / 'circuits.jsonl'


def emit_record(record: dict):
    """Append a circuit record (dict) to JSONL storage."""
    try:
        with open(JSONL_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Non-fatal: print for debugging
        print(f"⚠️ Failed to write circuit record: {e}")


def _normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def _clean_caption_for_search(caption: str) -> str:
    """Clean caption text for more robust PDF searching.
    
    Removes LaTeX artifacts and extracts meaningful search phrases.
    """
    if not caption:
        return ""
    
    # Remove common LaTeX artifacts that won't appear in PDF
    cleaned = caption
    cleaned = re.sub(r"<ref>|<cit\.?>", "", cleaned)  # Remove reference markers
    cleaned = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", cleaned)  # Remove LaTeX commands like \cite{...}
    cleaned = re.sub(r"[{}_\\]", "", cleaned)  # Remove LaTeX special chars
    cleaned = re.sub(r"\s+", " ", cleaned.strip())  # Normalize whitespace
    
    return cleaned


def find_caption_page_in_pdf(arxiv_id: str, caption: str, threshold: float = 0.5) -> int:
    """Find page number (1-based) in the cached PDF matching the caption.

    Uses multiple robust strategies with PyMuPDF:
    1. Clean caption text and use PyMuPDF's search_for() for fuzzy matching
    2. Extract distinctive phrases (5-10 words) and search for those
    3. Search for circuit-related keywords near caption words
    4. Fallback to word overlap with reasonable threshold
    
    Returns page number or None when not found.
    """
    if fitz is None:
        return None

    pdf_path = Path(PDF_CACHE_DIR) / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    # Clean the caption
    clean_caption = _clean_caption_for_search(caption)
    if not clean_caption or len(clean_caption) < 10:
        try:
            doc.close()
        except:
            pass
        return None

    # Strategy 1: Use PyMuPDF's search_for with distinctive phrases
    # Extract first meaningful 8-15 words (skip very short captions)
    words = clean_caption.split()
    if len(words) >= 5:
        # Try different phrase lengths for robustness
        for phrase_len in [10, 8, 6, 5]:
            if len(words) >= phrase_len:
                search_phrase = " ".join(words[:phrase_len])
                for page_num in range(doc.page_count):
                    try:
                        page = doc.load_page(page_num)
                        # search_for returns list of Rect objects if found
                        hits = page.search_for(search_phrase, quads=False)
                        if hits:
                            doc.close()
                            return page_num + 1
                    except:
                        continue

    # Strategy 2: Search for distinctive terms (longer words that are more unique)
    distinctive_words = [w for w in words if len(w) > 6][:5]
    if len(distinctive_words) >= 2:
        # Try combinations of distinctive words
        for i in range(len(distinctive_words) - 1):
            phrase = f"{distinctive_words[i]} {distinctive_words[i+1]}"
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    hits = page.search_for(phrase, quads=False)
                    if hits:
                        doc.close()
                        return page_num + 1
                except:
                    continue

    # Strategy 3: Regex search for circuit/quantum keywords + caption fragments
    circuit_keywords = r"(circuit|quantum|gate|qubit|implementation|algorithm)"
    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            page_text = page.get_text("text").lower()
            
            # Check if page has circuit keywords and caption words nearby
            if re.search(circuit_keywords, page_text):
                # Count how many caption words appear on this page
                caption_words = [w for w in words if len(w) > 3]
                matches = sum(1 for w in caption_words[:15] if w.lower() in page_text)
                if matches >= min(5, len(caption_words) * 0.6):
                    doc.close()
                    return page_num + 1
        except:
            continue

    # Strategy 4: Improved word overlap with better threshold
    caption_words = [w for w in words if len(w) > 2][:20]  # Limit to first 20 meaningful words
    best_match = None
    best_ratio = 0
    
    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            page_text = _normalize_text(page.get_text("text"))
            page_words = set(re.findall(r"\w{3,}", page_text))  # Words with 3+ chars
            
            if page_words:
                match_count = sum(1 for w in caption_words if w in page_words)
                ratio = match_count / max(1, len(caption_words))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = page_num + 1
        except:
            continue

    doc.close()
    
    # Return best match if it meets threshold
    if best_ratio >= threshold:
        return best_match
    
    return None


def update_pages_in_jsonl(arxiv_id: str = None):
    """Update records in `data/circuits.jsonl` filling `page` where missing.

    If `arxiv_id` is provided, only update records for that paper; otherwise update all.
    """
    if not JSONL_PATH.exists():
        return 0

    updated = 0
    tmp = JSONL_PATH.with_suffix('.tmp')
    try:
        with open(JSONL_PATH, 'r', encoding='utf-8') as rf, open(tmp, 'w', encoding='utf-8') as wf:
            for line in rf:
                try:
                    rec = json.loads(line)
                except Exception:
                    wf.write(line)
                    continue

                if rec.get('page') is None and rec.get('descriptions'):
                    if arxiv_id and rec.get('arxiv_id') != arxiv_id:
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue

                    caption = rec.get('descriptions')[0] if rec.get('descriptions') else ''
                    page = find_caption_page_in_pdf(rec.get('arxiv_id', ''), caption)
                    if page:
                        rec['page'] = page
                        updated += 1

                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # replace original
        tmp.replace(JSONL_PATH)
    except Exception as e:
        print(f"⚠️ Failed to update pages in JSONL: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return 0

    return updated
