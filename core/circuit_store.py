"""Utilities for persisting extracted circuit records and related metadata.

This module handles writing circuit records to disk, normalizing captions,
inferring page numbers/positions from PDFs and LaTeX sources, and regenerating
the consolidated ``circuits.json`` map. It exposes a function-based API and
keeps behavior compatible with the existing pipeline while improving
readability and reuse.
"""

import json
from pathlib import Path
import re
import unicodedata
from collections import Counter
from typing import Optional

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
from core.quantum_problem_classifier import (
    classify_quantum_problem,
    prepare_label_embeddings,
)


def _find_figure_mentions_pdf(raw_page_text: str, figure_number: int | None) -> list[tuple[int, int]]:
    """Locate numeric figure references in PDF text.

    Parameters
    ----------
    raw_page_text : str
        Full text of a PDF page.
    figure_number : int or None
        Target figure number to search for.

    Returns
    -------
    list[tuple[int, int]]
        List of character spans matching patterns like ``Fig. 3`` or ``Figure 3``.
    """
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

# Optional quantum-problem classifier cache (set via `set_quantum_problem_model`).
_QP_MODEL = None
_QP_LABEL_KEYS = None
_QP_LABEL_EMBEDS = None


def _ensure_data_files() -> None:
    """Create empty data files if missing.

    Ensures ``circuits.jsonl`` and ``circuits.json`` exist so downstream
    consumers can open them without extra guards.

    Returns
    -------
    None
    """

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


# Ensure files exist so other modules can rely on their presence
_ensure_data_files()


def _normalize_legacy_fields(record: dict) -> None:
    """Normalize legacy keys in-place (remove label, migrate figure).

    Parameters
    ----------
    record : dict
        Record to normalize. Mutated in place.

    Returns
    -------
    None
    """

    try:
        record.pop('label', None)
    except Exception:
        pass

    try:
        if 'figure_number' not in record and 'figure' in record:
            record['figure_number'] = record.pop('figure')
        else:
            record.pop('figure', None)
    except Exception:
        pass

    if 'figure_number' not in record:
        record['figure_number'] = None
def normalize_caption_text(s: str) -> str:
    """Normalize caption/description text while preserving domain tokens.

    Parameters
    ----------
    s : str
        Raw caption or description text.

    Returns
    -------
    str
        Cleaned caption with LaTeX/table artifacts removed and spacing normalized.

    Notes
    -----
    Operations include unwrapping common LaTeX subscript forms, spacing between
    alphanumeric runs, and collapsing whitespace.
    """
    if not s:
        return s
    try:
        # Work on a copy
        t = s
        # drop placeholder tags like <cit.> or <ref>
        t = re.sub(r"<\s*(?:cit|ref)\.?\s*>", " ", t, flags=re.IGNORECASE)
        # strip common table/spacing artifacts (e.g., "-3pt ccccc" column specs)
        t = re.sub(r"-?\d+\s*pt", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"[|]*[clrp](?:[|]*[clrp]){2,}", " ", t, flags=re.IGNORECASE)
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

        # If nothing alphabetic remains, drop the fragment
        if not re.search(r"[A-Za-z]", t):
            return ""
        return t
    except Exception:
        return s



def _sanitize_description_snippet(text: str, *, strip_latex: bool = True, max_len: int = 600) -> str:
    """Clean non-caption description text for embedding/transformer use.

    Parameters
    ----------
    text : str
        Input snippet.
    strip_latex : bool, optional
        Whether to remove LaTeX commands/math markers. Default is ``True``.
    max_len : int, optional
        Maximum length (in characters) after cleaning. Default is 600; use 0 to disable.

    Returns
    -------
    str
        Sanitized snippet, or empty string if no alphabetic characters remain.
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

        # drop common LaTeX/table artifacts that leak into snippets
        t = re.sub(r"<ref>|<cit>\.?>?", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\\\\", " ", t)  # TeX row separators
        t = re.sub(r"[&]{2,}", " ", t)  # table column artifacts
        t = re.sub(r"\b[clrp]{3,}\b", " ", t, flags=re.IGNORECASE)  # column spec like cccc
        t = re.sub(r"(?:\s*&\s*){3,}", " ", t)  # bare ampersand rows
        t = re.sub(r"@C=\d+\.\d+em|@R=\d+\.\d+em|@!R", " ", t)
        t = re.sub(r"\[!ht\]", " ", t)

        # remove control chars
        t = re.sub(r"[\u0000-\u001f]", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()

        # Drop snippets that have no alphabetic characters after cleaning
        if not re.search(r"[A-Za-z]", t):
            return ""

        if max_len and len(t) > max_len:
            t = t[:max_len].rsplit(" ", 1)[0]

        return t
    except Exception:
        return text

def _normalize_descriptions(record: dict) -> None:
    """Normalize and filter description strings in-place.

    Applies ``normalize_caption_text`` to each string description and removes
    empty fragments that result from cleaning table/placeholder artifacts.

    Parameters
    ----------
    record : dict
        Record whose ``descriptions`` will be normalized in-place.

    Returns
    -------
    None
    """

    if not isinstance(record, dict) or not record.get('descriptions'):
        return

    cleaned: list[str] = []
    for desc in record.get('descriptions', []):
        if isinstance(desc, str):
            norm = normalize_caption_text(desc)
            if norm.strip():
                cleaned.append(norm)
        else:
            cleaned.append(desc)
    record['descriptions'] = cleaned


def _has_meaningful_descriptions(record: dict) -> bool:
    """Check whether the record has a non-empty description.

    Parameters
    ----------
    record : dict
        Record to inspect.

    Returns
    -------
    bool
        ``True`` if at least one description string is non-empty.
    """

    try:
        descs = record.get('descriptions') if isinstance(record, dict) else None
        return bool(descs and any(str(d).strip() for d in descs))
    except Exception:
        return False


def _maybe_classify_record(record: dict) -> None:
    """Attach ``quantum_problem`` using the optional classifier cache.

    Parameters
    ----------
    record : dict
        Record to update. Mutated in place when classification succeeds.

    Returns
    -------
    None
    """

    if _QP_MODEL is None or _QP_LABEL_KEYS is None or _QP_LABEL_EMBEDS is None:
        return
    if not isinstance(record, dict) or not record.get('descriptions'):
        return

    try:
        label = classify_quantum_problem(
            _QP_MODEL,
            record.get('descriptions', []),
            _QP_LABEL_KEYS,
            _QP_LABEL_EMBEDS,
        )
        record['quantum_problem'] = label
    except Exception:
        pass


def _append_description(record: dict, snippet: str) -> None:
    """Append a description snippet if it is new and non-empty.

    Parameters
    ----------
    record : dict
        Target record whose ``descriptions`` will be updated.
    snippet : str
        Text snippet to append when non-empty and not already present.

    Returns
    -------
    None
    """

    if not snippet or not isinstance(record, dict):
        return
    descs = record.get('descriptions')
    if descs is None:
        record['descriptions'] = [snippet]
        return
    if snippet not in descs:
        descs.append(snippet)


def _append_text_positions(record: dict, spans: list[tuple[int, int]] | list[list[int]] | None) -> None:
    """Merge text span positions, avoiding duplicates.

    Parameters
    ----------
    record : dict
        Target record whose ``text_positions`` will be updated.
    spans : list[tuple[int, int]] or list[list[int]] or None
        Character spans to merge into the record.

    Returns
    -------
    None
    """

    if not spans:
        return
    existing = record.get('text_positions') if isinstance(record, dict) else None
    if existing is None:
        record['text_positions'] = [list(sp) for sp in spans]
        return
    for sp in spans:
        sp_list = list(sp)
        if sp_list not in existing:
            existing.append(sp_list)


def _load_pdf_page_text(arxiv_id: str, page_number: int) -> Optional[str]:
    """Load raw text for a specific PDF page if available.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier for the PDF.
    page_number : int
        One-based page index.

    Returns
    -------
    str or None
        Page text if available; otherwise ``None``.
    """

    if fitz is None or not page_number:
        return None
    pdf_path = Path(PDF_CACHE_DIR) / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_number - 1)
            return page.get_text("text")
        finally:
            try:
                doc.close()
            except Exception:
                pass
    except Exception:
        return None


def _enrich_record_from_sources(record: dict) -> None:
    """Fill page/figure info and add nearby snippets from LaTeX/PDF sources.

    Parameters
    ----------
    record : dict
        Record to enrich. Mutated in place.

    Returns
    -------
    None
    """

    if not isinstance(record, dict):
        return

    arxiv_id = record.get('arxiv_id')
    caption = record.get('descriptions', [None])[0] if record.get('descriptions') else ''
    if not arxiv_id or not caption:
        return

    page_val = record.get('page')
    fig_val = None

    if page_val is None:
        try:
            res = find_caption_page_in_pdf(arxiv_id, caption)
        except Exception:
            res = None
        if res:
            page_val, fig_val = res
            record['page'] = page_val
            if fig_val is not None:
                record['figure_number'] = fig_val

    # LaTeX context snippet (best-effort)
    try:
        latex_text = load_latex_source(arxiv_id, CACHE_DIR)
    except Exception:
        latex_text = None

    if latex_text:
        try:
            span_tex = locate_description_span_raw(latex_text, caption)
            if span_tex:
                snippet = extract_context_snippet(latex_text, span_tex, strip_latex=True)
                snippet = _sanitize_description_snippet(snippet, strip_latex=True)
                _append_description(record, snippet)
        except Exception:
            pass

    # PDF span and mention snippets
    page_text_pdf = _load_pdf_page_text(arxiv_id, page_val) if page_val else None
    span_positions = None
    if page_text_pdf:
        try:
            span_pdf = locate_description_span_raw(page_text_pdf, caption)
            if span_pdf:
                span_positions = [list(span_pdf)]
                _append_text_positions(record, span_positions)
        except Exception:
            pass

        try:
            if record.get('figure_number') is not None:
                mention_spans = _find_figure_mentions_pdf(page_text_pdf, record.get('figure_number'))
            else:
                mention_spans = []
            if mention_spans:
                _append_text_positions(record, mention_spans)
                for sp in mention_spans:
                    msnip = extract_context_snippet(page_text_pdf, sp, strip_latex=False)
                    msnip = _sanitize_description_snippet(msnip, strip_latex=False)
                    _append_description(record, msnip)
        except Exception:
            pass


def _write_jsonl_record(record: dict) -> None:
    """Append a record to ``circuits.jsonl`` and regenerate ``circuits.json``.

    Parameters
    ----------
    record : dict
        Record to persist.

    Returns
    -------
    None
    """

    with open(JSONL_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        _regenerate_json()
    except Exception:
        pass


def set_quantum_problem_model(model):
    """Register a pre-loaded SBERT model for quantum problem classification.

    Parameters
    ----------
    model : Any
        SentenceTransformer-like model with ``encode`` support.

    Returns
    -------
    bool
        ``True`` if the model was registered successfully, else ``False``.
    """
    global _QP_MODEL, _QP_LABEL_KEYS, _QP_LABEL_EMBEDS
    _QP_MODEL = model
    try:
        _QP_LABEL_KEYS, _QP_LABEL_EMBEDS = prepare_label_embeddings(model)
    except Exception:
        _QP_LABEL_KEYS, _QP_LABEL_EMBEDS = None, None
        _QP_MODEL = None
    return _QP_MODEL is not None


def emit_record(record: dict):
    """Append a circuit record to JSONL storage.

    Parameters
    ----------
    record : dict
        Circuit record containing at least ``descriptions`` and ``arxiv_id``.

    Returns
    -------
    None
    """

    try:
        if not isinstance(record, dict):
            return

        rec = dict(record)
        _normalize_legacy_fields(rec)
        _normalize_descriptions(rec)
        if not _has_meaningful_descriptions(rec):
            return

        _maybe_classify_record(rec)

        if rec.get('page') is None:
            _enrich_record_from_sources(rec)

        rec.pop('figure', None)
        _write_jsonl_record(rec)

    except Exception as exc:
        print(f"[WARN] Failed to write circuit record: {exc}")


def _regenerate_json():
    """Regenerate ``circuits.json`` from ``circuits.jsonl`` atomically.

    Returns
    -------
    None
    """
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

                # Backward compatibility: migrate legacy `figure` key
                if 'figure_number' not in rec and 'figure' in rec:
                    rec['figure_number'] = rec.pop('figure')
                else:
                    rec.pop('figure', None)

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
                        # Ensure `figure_number` is the second key after `arxiv_id` when present
                        def _order_value_dict(d: dict) -> dict:
                            out = {}
                            # keep arxiv_id first if present
                            if 'arxiv_id' in d:
                                out['arxiv_id'] = d.pop('arxiv_id')
                            # place figure_number second; if missing set to None
                            if 'figure_number' in d:
                                out['figure_number'] = d.pop('figure_number')
                            elif 'figure' in d:
                                out['figure_number'] = d.pop('figure')
                            else:
                                out['figure_number'] = None
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
                    # Preserve ordering with `figure_number` as second field when available
                    def _order_value_dict2(d: dict) -> dict:
                        out = {}
                        if 'arxiv_id' in d:
                            out['arxiv_id'] = d.pop('arxiv_id')
                        # ensure figure_number key exists (null when absent)
                        if 'figure_number' in d:
                            out['figure_number'] = d.pop('figure_number')
                        elif 'figure' in d:
                            out['figure_number'] = d.pop('figure')
                        else:
                            out['figure_number'] = None
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
    """Normalize text for comparison.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Lowercased string with collapsed whitespace.
    """
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def _clean_caption_for_search(caption: str) -> str:
    """Clean caption text for more robust PDF searching.

    Removes LaTeX artifacts and extracts meaningful search phrases.

    Parameters
    ----------
    caption : str
        Caption text, possibly containing LaTeX markers.

    Returns
    -------
    str
        Normalized caption suitable for matching in PDF text.
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







def _tokenize_for_comparison(text: str, is_latex: bool = False, min_len: int = 3):
    """Normalize and tokenize text for caption comparison.

    Parameters
    ----------
    text : str
        Input text.
    is_latex : bool, optional
        Whether to treat the input as LaTeX and attempt conversion. Default is ``False``.
    min_len : int, optional
        Minimum token length to retain, unless whitelisted. Default is 3.

    Returns
    -------
    set[str]
        Set of normalized word tokens.
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
    """Compute TF-IDF cosine similarity for a query against documents.

    Parameters
    ----------
    query_tokens : Counter
        Token counts for the query.
    doc_tokens_list : list[Counter]
        Token counts for each candidate document.

    Returns
    -------
    list[float]
        Similarity scores aligned with ``doc_tokens_list``. Returns zeros when
        scikit-learn is unavailable.
    """

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
    """Find the PDF page where a caption appears and optionally its figure number.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier of the paper.
    caption : str
        Caption text to search for.
    threshold : float, optional
        Minimum similarity threshold (unused in current implementation but kept
        for API compatibility). Default is 0.08.

    Returns
    -------
    tuple[int, int or None] or None
        ``(page_number, figure_number)`` when found; otherwise ``None``.

    Notes
    -----
    Searches line-anchored "Fig./Figure N" patterns first, then falls back to
    windowed TF-IDF matching over page text.
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
    """Locate the caption span in raw page text.

    Parameters
    ----------
    raw_page_text : str
        Raw page text (PDF or LaTeX).
    caption : str
        Caption text to locate.

    Returns
    -------
    tuple[int, int] or None
        Character offsets ``(start, end)`` in the raw text, or ``None`` if not found.

    Notes
    -----
    Attempts normalized substring match, whitespace-flexible regex, short-token
    anchors, and a token-subsequence heuristic with order preservation.
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
    """Fill missing page numbers in ``data/circuits.jsonl``.

    Parameters
    ----------
    arxiv_id : str, optional
        If provided, only records for this paper are updated.

    Returns
    -------
    int
        Count of records whose page number was newly set.
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

                _normalize_legacy_fields(rec)
                _normalize_descriptions(rec)

                if arxiv_id and rec.get('arxiv_id') != arxiv_id:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                missing_page = rec.get('page') is None
                if missing_page and rec.get('descriptions'):
                    _enrich_record_from_sources(rec)
                    if missing_page and rec.get('page') is not None:
                        updated += 1

                _maybe_classify_record(rec)
                rec.pop('figure', None)
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tmp.replace(JSONL_PATH)
    except Exception as exc:
        print(f"[WARN] Failed to update pages in JSONL: {exc}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return 0

    return updated
