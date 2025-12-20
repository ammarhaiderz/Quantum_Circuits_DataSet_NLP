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

from config.settings import CACHE_DIR, PDF_CACHE_DIR, USE_STOPWORDS, NORMALIZE_HYPHENS
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = set()
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from utils.latex_text_utils import (
    extract_context_snippet,
    load_latex_source,
)


def _find_figure_mentions_pdf(raw_page_text: str, figure_number: int | None) -> list[tuple[int, int]]:
    """Find numeric figure references in PDF text (e.g., "Fig. 3")."""
    if figure_number is None:
        return []
    try:
        pattern = re.compile(rf"\b(?:fig\.|figure)\s*{re.escape(str(figure_number))}\b", re.IGNORECASE)
        return [(m.start(), m.end()) for m in pattern.finditer(raw_page_text)]
    except Exception:
        return []

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
        # Skip records with no meaningful descriptions (captions)
        try:
            descs = record.get('descriptions') if isinstance(record, dict) else None
            if not descs or all((not d or not str(d).strip()) for d in descs):
                # nothing useful to index/emit
                return
        except Exception:
            # if access fails, fall through and attempt to write
            pass
        # Normalize descriptions (preserve domain tokens like TOF/CX and numeric groups)
        if isinstance(record, dict) and record.get('descriptions'):
            try:
                record['descriptions'] = [normalize_caption_text(d) if isinstance(d, str) else d for d in record.get('descriptions', [])]
            except Exception:
                # best-effort: leave descriptions as-is on failure
                pass

        # Best-effort: compute page at emit time to avoid race where update_pages_in_jsonl
        # was called before this record existed. Use caption -> PDF matcher if possible.
        try:
            if isinstance(record, dict) and record.get('page') is None:
                aid = record.get('arxiv_id')
                if aid and record.get('descriptions'):
                    try:
                        res = find_caption_page_in_pdf(aid, record['descriptions'][0])
                        if res:
                            page_val, fig_val = res
                            if page_val:
                                record['page'] = page_val
                            # Use LaTeX only to enrich descriptions; positions strictly from PDF text.
                            try:
                                latex_text = load_latex_source(aid, CACHE_DIR)
                            except Exception:
                                latex_text = None

                            if latex_text:
                                try:
                                    span_tex = locate_description_span_raw(latex_text, record['descriptions'][0])
                                    if span_tex:
                                        snippet = extract_context_snippet(latex_text, span_tex, strip_latex=True)
                                        snippet = _sanitize_description_snippet(snippet, strip_latex=True)
                                        if snippet and 'descriptions' in record and isinstance(record['descriptions'], list):
                                            if snippet not in record['descriptions']:
                                                record['descriptions'].append(snippet)
                                except Exception:
                                    pass

                            span_positions = None
                            page_text_pdf = None
                            try:
                                if page_val and fitz is not None:
                                    pdf_path2 = Path(PDF_CACHE_DIR) / f"{aid}.pdf"
                                    if pdf_path2.exists():
                                        doc2 = fitz.open(str(pdf_path2))
                                        try:
                                            pg = doc2.load_page(page_val - 1)
                                            page_text_pdf = pg.get_text("text")
                                            span_pdf = locate_description_span_raw(page_text_pdf, record['descriptions'][0])
                                            if span_pdf:
                                                span_positions = [list(span_pdf)]
                                        finally:
                                            try:
                                                doc2.close()
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            if span_positions:
                                record['text_positions'] = span_positions

                            # Mention spans: only from PDF text when available
                            try:
                                if page_text_pdf and record.get('figure') is not None:
                                    mention_spans = _find_figure_mentions_pdf(page_text_pdf, record.get('figure'))
                                else:
                                    mention_spans = []

                                if mention_spans:
                                    if not record.get('text_positions'):
                                        record['text_positions'] = []
                                    for sp in mention_spans:
                                        sp_list = [sp[0], sp[1]]
                                        if sp_list not in record['text_positions']:
                                            record['text_positions'].append(sp_list)

                                    if 'descriptions' in record and isinstance(record['descriptions'], list):
                                        for sp in mention_spans:
                                            msnip = extract_context_snippet(page_text_pdf, sp, strip_latex=False) if page_text_pdf else ""
                                            msnip = _sanitize_description_snippet(msnip, strip_latex=False)
                                            if msnip and msnip not in record['descriptions']:
                                                record['descriptions'].append(msnip)
                            except Exception:
                                pass
                            if fig_val is not None:
                                record['figure'] = fig_val
                    except Exception:
                        # non-fatal: leave page as-is
                        pass
        except Exception:
            pass
        # (figure numbers removed) No further figure-number extraction performed

        with open(JSONL_PATH, 'a', encoding='utf-8') as f:
            # Ensure `figure` key exists in JSONL records (use None when unknown)
            if 'figure' not in record:
                record['figure'] = None
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

                # If the record already contains an `image_filename` assigned at emit time,
                # prefer that (newer flow assigns this before emitting). Otherwise fall back
                # to deriving a stem from legacy fields (`block_name` or `raw_block_file`).
                img_name = rec.get('image_filename')
                if img_name:
                    # verify the PNG actually exists in the common folder
                    if common_png_dir.exists() and (common_png_dir / img_name).exists():
                        # do not store auxiliary fields inside the value dict
                        clean = dict(rec)
                        for _f in ('raw_block_file', 'block_id', 'block_name', 'image_filename'):
                            clean.pop(_f, None)
                        # Ensure `figure` is the second key after `arxiv_id` when present
                        def _order_value_dict(d: dict) -> dict:
                            out = {}
                            # keep arxiv_id first if present
                            if 'arxiv_id' in d:
                                out['arxiv_id'] = d.pop('arxiv_id')
                            # place figure second; if missing set to None
                            if 'figure' in d:
                                out['figure'] = d.pop('figure')
                            else:
                                out['figure'] = None
                            # then dump remaining keys in original order
                            for k, v in d.items():
                                out[k] = v
                            return out

                        records_dict[img_name] = _order_value_dict(clean)
                    else:
                        # missing file; skip
                        continue
                else:
                    # legacy fallback
                    rb = rec.get('raw_block_file', '') or rec.get('block_name', '')
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
                        img_name2 = chosen.name
                    except Exception:
                        img_name2 = matches[0].name

                    # do not inject `image_filename` into the stored value; keep it only as the key
                    clean2 = dict(rec)
                    for _f in ('raw_block_file', 'block_id', 'block_name', 'image_filename'):
                        clean2.pop(_f, None)
                    # Preserve ordering with `figure` as second field when available
                    def _order_value_dict2(d: dict) -> dict:
                        out = {}
                        if 'arxiv_id' in d:
                            out['arxiv_id'] = d.pop('arxiv_id')
                        # ensure figure key exists (null when absent)
                        if 'figure' in d:
                            out['figure'] = d.pop('figure')
                        else:
                            out['figure'] = None
                        for k, v in d.items():
                            out[k] = v
                        return out

                    records_dict[img_name2] = _order_value_dict2(clean2)
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
    # remove explicit reference placeholders and simple <cit> tokens
    cleaned = re.sub(r"<ref>|<cit\.?>", "", cleaned)
    # remove common LaTeX ref/cite commands like \ref{..}, \cref{..}, \cite{..}
    cleaned = re.sub(r"\\(?:ref|cref|Cref|cite|autoref)\{[^}]*\}", "", cleaned)
    # remove occurrences like 'Figure <ref>' or 'Figure \ref{...}' which refer to other figures
    cleaned = re.sub(r"(?:Figure|Fig\.?)(?:\s*<ref>|\s*\\ref\{[^}]*\})", "", cleaned, flags=re.IGNORECASE)
    # remove generic standalone 'Figure' words
    cleaned = re.sub(r"\b(?:Figure|Fig\.)\b", "", cleaned, flags=re.IGNORECASE)
    # remove other LaTeX commands
    cleaned = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", cleaned)
    # remove remaining special chars
    cleaned = re.sub(r"[{}_\\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.strip())
    
    return cleaned


def normalize_caption_text(s: str) -> str:
    """Normalize caption/description text while preserving domain tokens.

    - Unwrap common LaTeX subscript forms like _{1,2} or _1,2_
    - Replace adjacent underscore-groups with spaces
    - Insert spaces between alpha and numeric runs (TOF1 -> TOF 1)
    - Collapse extra whitespace
    """
    if not s:
        return s
    try:
        # Work on a copy
        t = s
        # common LaTeX escaping
        t = t.replace('\\_', '_')
        # unwrap \text{...}
        t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
        # _{1,2,3} -> 1,2,3
        t = re.sub(r"_\{([^}]*)\}", r"\1", t)
        # _1,2,3_ -> 1,2,3
        t = re.sub(r"_([0-9,]+)_", r"\1", t)
        # join adjacent numeric groups separated by underscores (or multiple underscores)
        t = re.sub(r"([0-9,])_+([0-9,])", r"\1 \2", t)
        # if numeric groups ended up adjacent after removals (e.g. "1,2,41,3,4"), insert space
        t = re.sub(r"([0-9,])(?=[0-9])", r"\1 ", t)
        # Insert space between letters and digits: TOF1 -> TOF 1
        t = re.sub(r"([A-Za-z])(?=\d)", r"\1 ", t)
        # Insert space between digits and letters: 1TOF -> 1 TOF
        t = re.sub(r"(?<=\d)(?=[A-Za-z])", ' ', t)
        # replace remaining underscores with space
        t = t.replace('_', ' ')
        # collapse whitespace
        # remove spaces after commas when both sides are digits: '1, 2' -> '1,2'
        t = re.sub(r'(?<=\d),\s+(?=\d)', ',', t)
        t = re.sub(r"\s+", ' ', t).strip()
        return t
    except Exception:
        return s



def _sanitize_description_snippet(text: str, *, strip_latex: bool = True, max_len: int = 600) -> str:
    """Clean non-caption description text for embedding/transformer use.

    - Optionally remove simple LaTeX commands/math markers
    - Drop control chars and collapse whitespace
    - Truncate overly long snippets to avoid noisy tails
    """
    if not text:
        return ""
    try:
        t = text
        if strip_latex:
            # remove LaTeX comments and commands
            t = re.sub(r"%.*", " ", t)
            t = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", t)
            # strip math regions
            t = re.sub(r"\$[^$]*\$", " ", t)
            # drop braces/underscores commonly left behind
            t = re.sub(r"[{}_^~]", " ", t)

        # remove control chars
        t = re.sub(r"[\u0000-\u001f]", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()

        if max_len and len(t) > max_len:
            t = t[:max_len].rsplit(" ", 1)[0]

        return t
    except Exception:
        return text



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
    if TfidfVectorizer is None or cosine_similarity is None:
        return [0.0] * len(doc_tokens_list)

    # Build texts as UNIQUE tokens only (presence-based)
    query_text = " ".join(query_tokens.keys())
    doc_texts = [" ".join(d.keys()) for d in doc_tokens_list]

    # Caption-constrained vocabulary
    vocab = set(query_tokens.keys())
    for d in doc_tokens_list:
        vocab.update(d.keys())

    if USE_STOPWORDS:
        stopset = set(ENGLISH_STOP_WORDS)
        vocab = {t for t in vocab if (t not in stopset) or (t in query_tokens)}

    if not vocab:
        return [0.0] * len(doc_texts)

    vec = TfidfVectorizer(
        vocabulary=sorted(vocab),
        tokenizer=str.split,
        lowercase=False,
        ngram_range=(1, 1),
        norm="l2"
    )

    mat = vec.fit_transform([query_text] + doc_texts)
    sims = cosine_similarity(mat[0], mat[1:])[0]

    return sims.tolist()


def find_caption_page_in_pdf(arxiv_id: str, caption: str, threshold: float = 0.08) -> tuple[int, int | None] | None:
    """
    Find the page number (1-based) in the PDF where the figure caption appears.

    Strategy:
    - Normalize LaTeX caption once
    - Tokenize caption
    - For each page:
        - Look for caption anchors at start of line (Fig./Figure N)
        - If anchors exist: extract short windows after anchors
        - Else: fall back to fixed-size text windows
    - Rank all windows using TF-IDF similarity
    - Return page of best match if above threshold
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

    # Normalize and tokenize LaTeX caption
    clean_caption = _normalize_text(_clean_caption_for_search(caption))
    qtokens = _tokenize_for_comparison(clean_caption, is_latex=True, min_len=1)
    if not qtokens:
        doc.close()
        return None

    # Caption anchors: require start-of-sentence to be a valid anchor.
    # Capture the figure number so we can return it alongside the page.
    anchor_re = re.compile(r"(?:^|[\.!?]\s+)(?:Fig\.?|Figure)\s*(\d+)\b", re.IGNORECASE)
    stop_re = re.compile(r"(?:^|[\.!?]\s+)(?:Fig\.?|Figure|Table)\s*\d+\b", re.IGNORECASE)

    page_candidates: list[tuple[int, str]] = []

    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            page_raw = page.get_text("text")
        except Exception:
            continue

        # Work line-by-line to avoid mixing paragraph text when newlines were
        # collapsed. This prevents mid-sentence "Fig. 1" references from being
        # treated as caption anchors.
        lines = page_raw.splitlines()
        found_anchor = False
        for li, line_raw in enumerate(lines):
            try:
                line_norm = _normalize_text(line_raw)
            except Exception:
                line_norm = line_raw.lower()

            for m in anchor_re.finditer(line_norm):
                found_anchor = True
                # Prefer text after the anchor on the same line; otherwise take
                # the next non-empty line as the caption candidate.
                after = line_norm[m.end():].strip()
                candidate = ''
                if after:
                    candidate = after
                else:
                    # look ahead up to 3 following lines for a likely caption line
                    for k in range(li + 1, min(len(lines), li + 4)):
                        nxt = _normalize_text(lines[k])
                        if nxt.strip():
                            candidate = nxt.strip()
                            break

                # stop at next anchor/table reference within the candidate
                s = stop_re.search(candidate)
                if s:
                    candidate = candidate[:s.start()]
                candidate = re.sub(r"^[\s:\-\–]+", "", candidate)
                # store candidate along with the captured figure number (int)
                try:
                    fignum = int(m.group(1))
                except Exception:
                    fignum = None
                page_candidates.append((page_num, candidate, fignum))

        if not found_anchor:
            # Fallback: use normalized whole page broken into windows
            page_text_norm = _normalize_text(page_raw)
            for i in range(0, len(page_text_norm), 300):
                page_candidates.append((page_num, page_text_norm[i:i + 300], None))

    docs = []
    pages = []

    fig_nums = []
    for pnum, text, fnum in page_candidates:
        tokens = _tokenize_for_comparison(text, is_latex=False, min_len=2)
        if tokens:
            docs.append(Counter(tokens))
            pages.append(pnum)
            fig_nums.append(fnum)

    if not docs:
        doc.close()
        return None

    qcounter = Counter(qtokens)
    scores = _compute_tfidf_similarity(qcounter, docs)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_page = pages[best_idx]
    best_fig = fig_nums[best_idx] if fig_nums else None

    doc.close()

    # Return (page, figure_number) where figure_number may be None.
    return (best_page + 1, best_fig)

def locate_description_span_raw(
    raw_page_text: str,
    caption: str
) -> tuple[int, int] | None:
    """
    Locate the caption span in RAW page text and return (start, end) character offsets.

    Strategy:
    - Lightly normalize BOTH raw page text and caption in the SAME way
    - Try full-caption substring match
    - Fallback: match a short anchor (first few words)
    - Map normalized offsets back to raw text offsets
    """

    if not raw_page_text or not caption:
        return None

    def _normalize_with_mapping(s: str) -> tuple[str, list[int]]:
        """Lightly normalize and keep index mapping to raw chars."""
        out_chars: list[str] = []
        mapping: list[int] = []
        prev_space = False
        for idx, ch in enumerate(s):
            # drop soft hyphen
            if ch == '\u00ad':
                continue
            # normalize nbsp to space
            if ch == '\xa0':
                ch = ' '
            ch_low = ch.lower()
            if ch_low.isspace():
                if prev_space:
                    # collapse multiple spaces
                    continue
                prev_space = True
                out_chars.append(' ')
                mapping.append(idx)
                continue
            prev_space = False
            out_chars.append(ch_low)
            mapping.append(idx)
        norm = ''.join(out_chars).strip()
        return norm, mapping

    try:
        cleaned_caption = _clean_caption_for_search(caption)
    except Exception:
        cleaned_caption = caption

    raw_norm, map_norm_to_raw = _normalize_with_mapping(raw_page_text)
    cap_norm, _ = _normalize_with_mapping(cleaned_caption)

    if not cap_norm:
        return None

    # Direct substring match on normalized strings (fast path)
    idx = raw_norm.find(cap_norm)
    if idx != -1:
        if idx + len(cap_norm) - 1 >= len(map_norm_to_raw):
            return None
        raw_start = map_norm_to_raw[idx]
        raw_end = map_norm_to_raw[idx + len(cap_norm) - 1] + 1
        return (raw_start, raw_end)

    # Fallback 1: flexible whitespace regex on all tokens
    try:
        tokens = [t for t in re.split(r"\s+", cap_norm) if t]
        if tokens:
            pattern = r"\\s+".join([re.escape(t) for t in tokens])
            m = re.search(pattern, raw_norm, flags=re.IGNORECASE)
            if m:
                start_norm = m.start()
                end_norm = m.end()
                if end_norm - 1 < len(map_norm_to_raw):
                    raw_start = map_norm_to_raw[start_norm]
                    raw_end = map_norm_to_raw[end_norm - 1] + 1
                    return (raw_start, raw_end)
    except Exception:
        pass

    # Fallback 2: shorter anchor (first N tokens) to catch truncated captions
    try:
        short_tokens = tokens[:8] if tokens else []
        if short_tokens:
            pattern_short = r"\\s+".join([re.escape(t) for t in short_tokens])
            m = re.search(pattern_short, raw_norm, flags=re.IGNORECASE)
            if m:
                start_norm = m.start()
                end_norm = m.end()
                if end_norm - 1 < len(map_norm_to_raw):
                    raw_start = map_norm_to_raw[start_norm]
                    raw_end = map_norm_to_raw[end_norm - 1] + 1
                    return (raw_start, raw_end)
    except Exception:
        pass

    # Fallback 3: token subsequence (>=70% of tokens in order, gaps allowed)
    try:
        if tokens:
            idxes = []
            start = 0
            for tok in tokens:
                pos = raw_norm.find(tok, start)
                if pos == -1:
                    continue
                idxes.append((pos, pos + len(tok)))
                start = pos + len(tok)
            if idxes and len(idxes) / len(tokens) >= 0.7:
                raw_start = map_norm_to_raw[idxes[0][0]]
                raw_end = map_norm_to_raw[idxes[-1][1] - 1] + 1
                return (raw_start, raw_end)
    except Exception:
        pass

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
                    try:
                        res = find_caption_page_in_pdf(rec.get('arxiv_id', ''), caption)
                        if res:
                            page_val, fig_val = res
                            if page_val:
                                rec['page'] = page_val
                                updated += 1
                                # Use LaTeX only for description enrichment; positions strictly from PDF text.
                                try:
                                    latex_text = load_latex_source(rec.get('arxiv_id', ''), CACHE_DIR)
                                except Exception:
                                    latex_text = None

                                if latex_text:
                                    try:
                                        span_tex = locate_description_span_raw(latex_text, caption)
                                        if span_tex:
                                            try:
                                                snippet = extract_context_snippet(latex_text, span_tex, strip_latex=True)
                                                snippet = _sanitize_description_snippet(snippet, strip_latex=True)
                                                if snippet and 'descriptions' in rec and isinstance(rec['descriptions'], list):
                                                    if snippet not in rec['descriptions']:
                                                        rec['descriptions'].append(snippet)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                span_positions = rec.get('text_positions') if rec.get('text_positions') else None
                                page_text_pdf = None
                                try:
                                    if page_val and fitz is not None:
                                        pdf_path3 = Path(PDF_CACHE_DIR) / f"{rec.get('arxiv_id', '')}.pdf"
                                        if pdf_path3.exists():
                                            doc3 = fitz.open(str(pdf_path3))
                                            try:
                                                pg3 = doc3.load_page(page_val - 1)
                                                page_text_pdf = pg3.get_text("text")
                                                span_pdf = locate_description_span_raw(page_text_pdf, caption)
                                                if span_pdf:
                                                    span_positions = [list(span_pdf)]
                                            finally:
                                                try:
                                                    doc3.close()
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                                if span_positions:
                                    rec['text_positions'] = span_positions

                                try:
                                    mention_spans = []
                                    if page_text_pdf and rec.get('figure') is not None:
                                        mention_spans = _find_figure_mentions_pdf(page_text_pdf, rec.get('figure'))

                                    if mention_spans:
                                        if not rec.get('text_positions'):
                                            rec['text_positions'] = []
                                        for sp in mention_spans:
                                            sp_list = [sp[0], sp[1]]
                                            if sp_list not in rec['text_positions']:
                                                rec['text_positions'].append(sp_list)
                                        if 'descriptions' in rec and isinstance(rec['descriptions'], list):
                                            for sp in mention_spans:
                                                msnip = extract_context_snippet(page_text_pdf, sp, strip_latex=False) if page_text_pdf else ""
                                                msnip = _sanitize_description_snippet(msnip, strip_latex=False)
                                                if msnip and msnip not in rec['descriptions']:
                                                    rec['descriptions'].append(msnip)
                                except Exception:
                                    pass
                            if fig_val is not None:
                                rec['figure'] = fig_val
                    except Exception:
                        # leave as-is on failure
                        pass

                # Ensure `figure` key is always present in JSONL (null when unknown)
                if 'figure' not in rec:
                    rec['figure'] = None
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
