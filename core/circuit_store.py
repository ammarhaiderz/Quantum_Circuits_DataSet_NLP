import json
from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict
import math

try:
    import fitz
except Exception:
    fitz = None

from config.settings import PDF_CACHE_DIR, USE_STOPWORDS, NORMALIZE_HYPHENS

DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = DATA_DIR / 'circuits.jsonl'
JSON_PATH = DATA_DIR / 'circuits.json'
# metadata path
META_PATH = DATA_DIR / 'circuits_meta.json'
LATEX_META_PATH = DATA_DIR / 'latex_checkpoint.json'
# Ensure files exist so other modules can rely on their presence
try:
    if not JSONL_PATH.exists():
        JSONL_PATH.write_text('', encoding='utf-8')
except Exception:
    pass
try:
    if not JSON_PATH.exists():
        JSON_PATH.write_text('[]', encoding='utf-8')
except Exception:
    pass


def emit_record(record: dict):
    """Append a circuit record (dict) to JSONL storage."""
    try:
        with open(JSONL_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # After appending to JSONL, regenerate a proper JSON array file.
        try:
            _regenerate_json()
        except Exception:
            # Non-fatal; JSONL still contains the record.
            pass
    except Exception as e:
        # Non-fatal: print for debugging
        print(f"⚠️ Failed to write circuit record: {e}")


def _regenerate_json():
    """Read `circuits.jsonl` and write `circuits.json` as a JSON array (atomic)."""
    if not JSONL_PATH.exists():
        # ensure empty JSON file
        JSON_PATH.write_text('[]', encoding='utf-8')
        return

    tmp = JSON_PATH.with_suffix('.json.tmp')
    records_dict = {}
    common_png_dir = Path('circuit_images') / 'rendered_pdflatex' / 'png'
    with open(JSONL_PATH, 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)

                # derive stem from raw_block_file
                rb = rec.get('raw_block_file', '')
                if not rb:
                    continue
                try:
                    tex_name = Path(rb).name
                    stem = Path(tex_name).stem
                except Exception:
                    continue

                # Only include records that have a matching PNG in the common rendered_pdflatex/png folder
                if common_png_dir.exists():
                    matches = list(common_png_dir.glob(f"*{stem}*.png"))
                else:
                    matches = []

                if not matches:
                    # skip records without a final PNG in the canonical folder
                    continue

                # choose most recently modified match
                try:
                    chosen = max(matches, key=lambda p: p.stat().st_mtime)
                    img_name = chosen.name
                except Exception:
                    img_name = matches[0].name

                rec['image_filename'] = img_name
                # Use the PNG filename as the main key in the JSON output
                records_dict[img_name] = rec
            except Exception:
                # skip malformed lines
                continue
    # write atomically a JSON object mapping image_filename -> record
    with open(tmp, 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(records_dict, ensure_ascii=False, indent=2))

    try:
        tmp.replace(JSON_PATH)
    except Exception:
        # fallback: try rename
        try:
            tmp.rename(JSON_PATH)
        except Exception:
            raise


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

def _tokenize_for_comparison(text: str, is_latex: bool = False, min_len: int = 3):
    """Normalize and tokenize text for caption comparison.

    - If `is_latex` and `pylatexenc` is available, convert LaTeX to plain text.
    - Normalize unicode (NFKC), remove soft-hyphens, strip combining marks.
    - Lowercase and extract words with length >= `min_len`.
    - Return a set of word tokens.
    """
    if not text:
        return set()

    s = text
    if is_latex:
        try:
            from pylatexenc.latex2text import LatexNodes2Text
            s = LatexNodes2Text().latex_to_text(s)
        except Exception:
            # fallback: remove simple \command{...} constructs
            s = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", s)

    # remove soft hyphen and non-printing spaces
    s = s.replace('\u00AD', '')
    s = s.replace('\xa0', ' ')
    # optional: normalize hyphenation across line breaks (e.g. "multi-\nqubit" -> "multiqubit")
    if NORMALIZE_HYPHENS:
        s = re.sub(r"-\s+", "", s)

    # Unicode normalize and remove combining marks
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.category(ch).startswith('M'))
    s = unicodedata.normalize('NFKC', s)

    s = s.lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Domain whitelist: keep these even if short
    DOMAIN_WHITELIST = {
        'cx', 'cnot', 'ccx', 'rccx', 'toffoli', 'mcx', 'measure', 'measurements',
        'h', 'x', 'y', 'z', 'u', 'mct', 'mcx'
    }

    # small stopword list (keeps domain tokens)
    STOPWORDS = {
        'a', 'an', 'and', 'the', 'of', 'in', 'on', 'for', 'to', 'is', 'are', 'be', 'by',
        'with', 'that', 'this', 'it', 'as', 'from', 'at', 'which', 'or', 'we', 'will',
        'can', 'our', 'such', 'these', 'those', 'their', 'has', 'have', 'was', 'were',
        'but', 'not', 'also', 'than', 'then', 'itself'
    }

    # extract all word-like tokens then filter
    raw_tokens = re.findall(r"\w+", s)
    tokens_final = []
    for t in raw_tokens:
        if not t:
            continue
        # drop long numeric sequences (arXiv ids, page numbers, years)
        if re.search(r"\d{3,}", t):
            continue
        # drop purely numeric tokens
        if t.isdigit():
            continue

        # apply min length unless whitelisted domain token
        if len(t) < min_len and t not in DOMAIN_WHITELIST:
            continue

        # stopword removal (configurable)
        if USE_STOPWORDS and t in STOPWORDS and t not in DOMAIN_WHITELIST:
            continue

        tokens_final.append(t)

    return set(tokens_final)


def _compute_tfidf_similarity(query_tokens: Counter, doc_tokens_list: list[Counter]) -> list[float]:
    """Compute TF-IDF cosine similarity between a query token Counter and
    a list of document token Counters. Returns list of similarity scores.

    Lightweight TF-IDF implementation (no external deps):
    - tf = term_count / doc_length
    - idf = log((N + 1) / (df + 1)) + 1
    - tf-idf vector built from vocabulary across docs+query
    - cosine similarity between query vector and each doc vector
    """
    # Build document frequencies
    N = len(doc_tokens_list)
    df = defaultdict(int)
    for d in doc_tokens_list:
        for term in d.keys():
            df[term] += 1
    # include query terms in df if absent
    for t in query_tokens.keys():
        if t not in df:
            df[t] += 0

    # compute idf
    idf = {t: math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0 for t in set(df) | set(query_tokens.keys())}

    # helper to build tf-idf vector
    def tfidf_vec(counter: Counter) -> dict:
        l = sum(counter.values())
        if l == 0:
            return {}
        vec = {}
        for t, c in counter.items():
            tf = c / l
            vec[t] = tf * idf.get(t, 1.0)
        return vec

    qvec = tfidf_vec(query_tokens)
    # precompute norm of qvec
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    scores = []
    for d in doc_tokens_list:
        dvec = tfidf_vec(d)
        dnorm = math.sqrt(sum(v * v for v in dvec.values()))
        if qnorm == 0 or dnorm == 0:
            scores.append(0.0)
            continue
        # dot product
        dot = 0.0
        for t, qv in qvec.items():
            dv = dvec.get(t, 0.0)
            dot += qv * dv
        scores.append(dot / (qnorm * dnorm))
    return scores


def find_caption_page_in_pdf(arxiv_id: str, caption: str, threshold: float = 0.02) -> int | None:
    """Find page number (1-based) in the cached PDF matching the caption.

    TF-IDF based approach:
    - Preprocess LaTeX caption into normalized tokens (no minimum length)
    - For each PDF page build candidate caption windows (anchor windows if anchors
      present, otherwise whole page)
    - Compute TF-IDF cosine similarity between the caption and each window
    - Return the page of the best-scoring window if score >= threshold
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

    clean_caption = _clean_caption_for_search(caption)
    if not clean_caption:
        try:
            doc.close()
        except:
            pass
        return None

    # tokenize caption (allow single-char tokens)
    qtokens = _tokenize_for_comparison(clean_caption, is_latex=True, min_len=1)
    if not qtokens:
        try:
            doc.close()
        except:
            pass
        return None

    # Build candidate windows per page
    anchor_re = re.compile(r"(?:Fig\.?|Figure)\s*\d+\b", re.IGNORECASE)
    stop_re = re.compile(r"\b(?:Fig\.?|Figure|Table)\s*\d+\b", re.IGNORECASE)

    page_candidates = []  # list of (page_num, window_text)
    try:
        for page_num in range(doc.page_count):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
            except Exception:
                continue

            text_norm = _normalize_text(page_text)

            # find anchors; if none, use whole page as single candidate
            anchors = list(anchor_re.finditer(text_norm))
            if not anchors:
                page_candidates.append((page_num, text_norm))
                continue

            for m in anchors:
                start = m.end()
                window = text_norm[start:start + 200]
                # stop at next anchor/table reference
                s = stop_re.search(window)
                if s:
                    window = window[:s.start()]
                page_candidates.append((page_num, window))

        # prepare token counters for each candidate
        docs = []
        pages = []
        for pnum, win in page_candidates:
            tokens = _tokenize_for_comparison(win, is_latex=False, min_len=1)
            docs.append(Counter(tokens))
            pages.append(pnum)

        qcounter = Counter(_tokenize_for_comparison(clean_caption, is_latex=True, min_len=1))
        if not docs:
            try:
                doc.close()
            except:
                pass
            return None

        scores = _compute_tfidf_similarity(qcounter, docs)
        # find best score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        best_page = pages[best_idx]

        try:
            doc.close()
        except:
            pass

        if best_score >= threshold:
            return best_page + 1
        return None
    finally:
        try:
            doc.close()
        except Exception:
            pass



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
