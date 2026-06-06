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
import difflib
import unicodedata
from collections import Counter
from typing import Optional

try:
    import fitz
except Exception:
    fitz = None

from config.settings import (
    IMAGE_PIPELINE_CACHE_DIR,
    IMAGE_PIPELINE_PDF_CACHE_DIR,
    LATEX_RENDER_DIR,
    USE_STOPWORDS,
    NORMALIZE_HYPHENS,
)
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
        num = re.escape(str(figure_number))
        # Allow supplemental anchors such as "Figure S10" in addition to numeric references.
        pattern = re.compile(rf"\b(?:fig\.|figure)\s*(?:s\s*)?{num}\b", re.IGNORECASE)
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

def strip_latex_environments(text: str) -> str:
    r"""
    Remove all LaTeX environments completely:
    \begin{...} ... \end{...}

    This is generic and does NOT rely on environment names.
    """
    pattern = re.compile(
        r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}",
        flags=re.DOTALL
    )
    return re.sub(pattern, " ", text)
def is_natural_language_paragraph(p: str) -> bool:
    """
    Heuristic to decide whether a paragraph is real prose
    suitable for text-to-image supervision.
    """

    p = p.strip()

    # Reject empty or very short
    if len(p) < 40:
        return False

    # Reject LaTeX commands at start
    if p.startswith(("\\", "{", "[", "@")):
        return False

    # Reject pure labels
    if re.fullmatch(r"\\label\{[^}]+\}", p):
        return False

    # Reject section headers
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", p):
        return False

    # Reject symbol-dominated text
    symbol_ratio = sum(
        c in "•≡⊕⊗⟂|⟩⟨"
        for c in p
    ) / max(len(p), 1)
    if symbol_ratio > 0.05:
        return False

    # Must contain alphabetic words
    if not re.search(r"[A-Za-z]{4,}", p):
        return False

    return True


def _sanitize_description_snippet(
    text: str,
    *,
    strip_latex: bool = True,
    max_len: int = 300
) -> str:
    """
    Clean descriptive text for text-to-image fine-tuning labels.

    The goal is to produce a fluent, natural-language sentence that
    describes the visual content of a circuit figure.

    This function:
    - removes LaTeX commands and references
    - removes figure/label scaffolding
    - removes citations
    - normalizes punctuation and whitespace
    - preserves descriptive semantics
    """

    if not text:
        return ""

    t = text
        # --- Hard reject figure placement / layout junk ---
    if re.search(r"\[\s*!?\s*ht\s*\]", t):
        return ""
    if re.search(r"@C\s*=\s*[\d\.]+\s*em|@R\s*=\s*[\d\.]+\s*em|@!R", t):
        return ""
    if re.search(r"(?:&\s*){3,}", t):
        return ""

    # --- Remove LaTeX control and math (softly) ---
    if strip_latex:
        # Remove comments
        t = re.sub(r"%.*", " ", t)

        # Remove citations \cite{...}
        t = re.sub(r"\\cite\{[^}]*\}", " ", t)

        # Remove references \ref{...}, \cref{...}, etc.
        t = re.sub(r"\\(?:ref|cref|Cref|autoref)\{[^}]*\}", " ", t)

        # Replace inline math with readable placeholders
        t = re.sub(r"\$\s*([A-Za-z0-9_,\\ ]+)\s*\$", r"\1", t)

        t = re.sub(
            r"\$\s*([^$]+?)\s*\$",
            lambda m: m.group(1).replace(";", ", "),
            t,
        )

    # --- Remove figure scaffolding ---
    # (Figure 3), Figure 3, Fig. 3, etc.
    t = re.sub(
        r"\(?\b(?:Figure|Fig\.?)\s+[A-Za-z0-9\-]+\)?",
        " ",
        t,
        flags=re.IGNORECASE,
    )

    # --- Normalize punctuation ---
    t = t.replace("–", "-")
    t = t.replace("—", "-")

    # Remove leftover braces and underscores
    t = re.sub(r"[{}_]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # --- Sentence sanity ---
    # Ensure it ends like a sentence
    if not re.search(r"[.!?]$", t):
        t = t.rstrip(" ,;:") + "."

    # --- Length sanity ---
    if len(t.split()) < 5:
        return ""

    if max_len and len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "."

    return t


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


def _append_text_positions(record: dict, spans):
    if not spans or not isinstance(record, dict):
        return

    clean_spans = []
    for sp in spans:
        try:
            s, e = int(sp[0]), int(sp[1])
            # HARD RULES
            if s < 0 or e <= s:
                continue
            if e - s < 10:   # minimum meaningful span
                continue
            clean_spans.append([s, e])
        except Exception:
            continue

    if not clean_spans:
        return

    existing = record.get("text_positions")
    if existing is None:
        record["text_positions"] = clean_spans
    else:
        for sp in clean_spans:
            if sp not in existing:
                existing.append(sp)


def _normalize_text_positions(record: dict) -> None:
    """Normalize and filter text span positions in-place.

    Removes placeholder spans like ``[0, 0]`` and malformed spans.
    """

    if not isinstance(record, dict):
        return

    spans = record.get("text_positions")
    if not isinstance(spans, list) or not spans:
        return

    # Legacy format only: list of [start, end].
    # We KEEP [0,0] placeholders (caller may use them for unmatched sentences).
    cleaned: list[list[int]] = []
    for sp in spans:
        try:
            s, e = int(sp[0]), int(sp[1])
        except Exception:
            continue
        if s < 0 or e < 0:
            continue
        # Allow placeholders [0,0]
        if s == 0 and e == 0:
            cand = [0, 0]
        else:
            if e <= s:
                continue
            cand = [s, e]
        # IMPORTANT: do not de-duplicate.
        # We want one span entry per description sentence.
        cleaned.append(cand)

    record["text_positions"] = cleaned


def _has_meaningful_text_positions(record: dict) -> bool:
    """Return True if record has at least one non-placeholder span."""

    if not isinstance(record, dict):
        return False
    spans = record.get("text_positions")
    if not isinstance(spans, list):
        return False
    for sp in spans:
        try:
            s, e = int(sp[0]), int(sp[1])
        except Exception:
            continue
        if s == 0 and e == 0:
            continue
        if s >= 0 and e > s:
            return True
    return False


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _protect_abbreviations(s: str) -> tuple[str, dict[str, str]]:
    """Protect common abbreviations to avoid naive sentence splitting errors."""
    repl = {}
    if not s:
        return s, repl

    # Minimal set tuned for paper captions
    abbrs = [
        "Fig.", "fig.", "Eq.", "eq.", "Eqs.", "No.", "Dr.", "Prof.",
        "e.g.", "i.e.", "et al.", "vs.", "cf.",
    ]
    out = s
    for i, a in enumerate(abbrs):
        key = f"__ABBR{i}__"
        if a in out:
            out = out.replace(a, key)
            repl[key] = a
    return out, repl


def _restore_abbreviations(s: str, repl: dict[str, str]) -> str:
    out = s
    for k, v in (repl or {}).items():
        out = out.replace(k, v)
    return out


def _split_sentences_simple(text: str) -> list[str]:
    """Simple sentence splitter (no external models).

    Good enough for captions/prose snippets; avoids NLTK punkt dependency.
    """
    if not text:
        return []

    s = _collapse_ws(text)
    s, repl = _protect_abbreviations(s)

    # Split on end punctuation + whitespace + next token start.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])", s)
    parts = [_restore_abbreviations(p.strip(), repl) for p in parts if p and p.strip()]

    # Extra fallback: if we still have huge chunks with no punctuation, split on semicolons.
    out: list[str] = []
    for p in parts:
        if len(p) > 300 and ";" in p:
            out.extend([q.strip() for q in p.split(";") if q.strip()])
        else:
            out.append(p)

    return out


def _normalize_for_match(s: str) -> tuple[str, set[str]]:
    """Normalize a sentence for matching against PDF text."""
    if not s:
        return "", set()

    t = s
    # Strip common LaTeX-ish markup that won't appear in PDF text
    # Examples: \ket{0}, \labelO, \mathrm{...}, \alpha, { } and escaped newlines.
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^\}]*\})?", " ", t)
    t = t.replace("{", " ").replace("}", " ")
    t = t.replace("\\", " ")
    t = t.replace("\u00AD", "")  # soft hyphen
    t = t.replace("\xa0", " ")
    t = unicodedata.normalize("NFKC", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()

    tokens = set(re.findall(r"[a-z0-9]+", t))
    tokens = {tok for tok in tokens if len(tok) >= 3 and not tok.isdigit()}
    return t, tokens


def _extract_pdf_sentence_spans(page_text: str) -> list[dict]:
    """Extract PDF sentences with original character spans."""
    if not page_text:
        return []

    text = page_text
    spans: list[dict] = []

    start = 0
    # End a sentence at .?! followed by whitespace/newline.
    for m in re.finditer(r"[.!?]+\s+", text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            norm, toks = _normalize_for_match(seg)
            spans.append({
                "text": seg,
                "start": start,
                "end": end,
                "norm": norm,
                "tokens": toks,
            })
        start = end

    tail = text[start:].strip()
    if tail:
        norm, toks = _normalize_for_match(tail)
        spans.append({
            "text": tail,
            "start": start,
            "end": len(text),
            "norm": norm,
            "tokens": toks,
        })

    return spans


def _align_one_sentence_to_pdf_span(
    sentence: str,
    pdf_sents: list[dict],
    *,
    min_ratio: float,
    min_token_recall: float,
    min_score: float,
) -> list[int]:
    """Return best [start,end] span for a sentence, or [0,0] if not found."""

    norm_q, toks_q = _normalize_for_match(sentence)
    if not norm_q or len(norm_q) < 10:
        return [0, 0]

    best = None
    for ps in pdf_sents:
        if not ps.get("norm"):
            continue

        if toks_q:
            tok_recall = len(toks_q & ps["tokens"]) / max(len(toks_q), 1)
        else:
            tok_recall = 0.0

        if toks_q and tok_recall < (min_token_recall * 0.5):
            continue

        ratio = difflib.SequenceMatcher(None, norm_q, ps["norm"]).ratio()
        score = 0.7 * ratio + 0.3 * tok_recall

        if best is None or score > best["score"]:
            best = {
                "start": ps["start"],
                "end": ps["end"],
                "ratio": ratio,
                "tok_recall": tok_recall,
                "score": score,
            }

    if (
        best
        and best["ratio"] >= min_ratio
        and best["score"] >= min_score
        and best["tok_recall"] >= min_token_recall
    ):
        return [int(best["start"]), int(best["end"])]

    return [0, 0]


def _align_description_items_to_pdf_spans(
    page_text: str | None,
    descriptions: list[str],
    figure_block_span: list[int] | None = None,
    *,
    min_ratio: float = 0.55,
    min_token_recall: float = 0.25,
    min_score: float = 0.60,
) -> list[list[int]]:
    """Align each *description item* to a PDF span.

    For each description string, we split into sentences and align each sentence.
    The final span for the description item is computed as:
    - [min(start), max(end)] across all matched sentence spans
    - [0,0] if none of its sentences could be aligned

    Output is legacy-only: list of [start, end] spans with length == len(descriptions).
    """

    if not descriptions:
        return []

    if not page_text:
        return [[0, 0] for _ in descriptions]

    pdf_sents = _extract_pdf_sentence_spans(page_text)
    if not pdf_sents:
        return [[0, 0] for _ in descriptions]

    item_spans: list[list[int]] = []
    for idx, desc in enumerate(descriptions):
        # Candidate set:
        # - For caption (idx==0), search the FULL page text to maximize match chance.
        # - For later items (extra context), prefer figure-block window when available.
        candidates = pdf_sents
        if idx > 0 and (
            isinstance(figure_block_span, list)
            and len(figure_block_span) == 2
            and isinstance(figure_block_span[0], int)
            and isinstance(figure_block_span[1], int)
            and figure_block_span[1] > figure_block_span[0]
        ):
            b0, b1 = figure_block_span
            subset = [s for s in pdf_sents if not (s["end"] <= b0 or s["start"] >= b1)]
            if subset:
                candidates = subset

        if not isinstance(desc, str) or not desc.strip():
            item_spans.append([0, 0])
            continue

        sent_spans = []
        for sent in _split_sentences_simple(desc):
            sp = _align_one_sentence_to_pdf_span(
                sent,
                candidates,
                min_ratio=min_ratio,
                min_token_recall=min_token_recall,
                min_score=min_score,
            )
            if sp != [0, 0]:
                sent_spans.append(sp)

        if not sent_spans:
            # Fallback behavior:
            # - For the first description item (usually the caption), keep [0,0] if not matched.
            # - For subsequent items (extra context), use the broader figure block span when available.
            if idx > 0 and (
                isinstance(figure_block_span, list)
                and len(figure_block_span) == 2
                and all(isinstance(x, int) for x in figure_block_span)
                and figure_block_span[1] > figure_block_span[0]
            ):
                item_spans.append([int(figure_block_span[0]), int(figure_block_span[1])])
            else:
                item_spans.append([0, 0])
        else:
            starts = [sp[0] for sp in sent_spans]
            ends = [sp[1] for sp in sent_spans]
            item_spans.append([int(min(starts)), int(max(ends))])

    return item_spans


def _strip_internal_fields(record: dict) -> None:
    """Remove internal-only metadata fields before persisting."""
    if not isinstance(record, dict):
        return
    record.pop('latex_text_positions', None)
    record.pop('figure_block_span', None)



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
    pdf_path = Path(IMAGE_PIPELINE_PDF_CACHE_DIR) / f"{arxiv_id}.pdf"
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


def locate_figure_block(
    page_text: str,
    figure_number: int | None = None,
    *,
    window: int = 350
) -> tuple[int, int] | None:
    """
    Locate an approximate figure block in PDF text using figure anchors only.

    This function is POSITION-ONLY:
    - No caption matching
    - No semantic extraction
    - No fuzzy logic

    Parameters
    ----------
    page_text : str
        Raw text extracted from a PDF page (PyMuPDF).
    figure_number : int or None
        Figure number to anchor on (preferred).
    window : int, optional
        Character window size around the anchor.

    Returns
    -------
    tuple[int, int] or None
        (start, end) character span approximating the figure region.
    """

    if not page_text:
        return None

    text = page_text

    # -----------------------------
    # 1. Try exact figure number anchor
    # -----------------------------
    if figure_number is not None:
        try:
            num = re.escape(str(figure_number))
            pattern = re.compile(
                rf"\b(?:fig\.|figure)\s*(?:s\s*)?{num}\b",
                flags=re.IGNORECASE
            )
            m = pattern.search(text)
            if m:
                center = m.start()
                start = max(0, center - window)
                end = min(len(text), center + window)
                return (start, end)
        except Exception:
            pass

    # -----------------------------
    # 2. Fallback: any Figure / Fig anchor
    # -----------------------------
    try:
        pattern = re.compile(r"\b(?:fig\.|figure)\b", re.IGNORECASE)
        m = pattern.search(text)
        if m:
            center = m.start()
            start = max(0, center - window)
            end = min(len(text), center + window)
            return (start, end)
    except Exception:
        pass

    # -----------------------------
    # 3. No anchor found
    # -----------------------------
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
        latex_text = load_latex_source(arxiv_id, IMAGE_PIPELINE_CACHE_DIR)
    except Exception:
        latex_text = None

    if latex_text:
        try:
            # Adjacent paragraph after figure (primary)
            para = _extract_paragraph_after_figure(latex_text, caption)
            if para:
                _append_description(record, para)
            else:
                # Fallback: localized snippet
                span_tex = locate_description_span_raw(latex_text, caption)
                if span_tex:
                    snippet = extract_context_snippet(latex_text, span_tex, strip_latex=True)
                    snippet = _sanitize_description_snippet(snippet, strip_latex=True)
                    if snippet:
                        _append_description(record, snippet)
        except Exception:
            pass

     # PDF span and mention snippets (POSITIONS ONLY — NO TEXT)
    page_text_pdf = _load_pdf_page_text(arxiv_id, page_val) if page_val else None
    if page_text_pdf:
        try:
            # Caption span (for alignment only)
            # Anchor to figure block, not caption string
            fig_block = locate_figure_block(
                page_text_pdf,
                figure_number=record.get("figure_number")
            )

            if fig_block:
                record["figure_block_span"] = [int(fig_block[0]), int(fig_block[1])]
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


def _get_common_png_dir() -> Path:
    """Return canonical PNG output directory.

    Uses ``LATEX_RENDER_DIR`` from settings. Supports configurations where
    ``LATEX_RENDER_DIR`` points at either the rendered root directory (e.g.
    ``circuit_images/rendered_pdflatex``) or directly at the PNG folder
    (e.g. ``circuit_images/rendered_pdflatex/png``).
    """

    try:
        base = Path(LATEX_RENDER_DIR)
    except Exception:
        base = Path('circuit_images') / 'rendered_pdflatex'

    # If the configured path already points to a png folder, keep it.
    try:
        if base.name.lower() == 'png':
            return base
    except Exception:
        pass

    return base / 'png'


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
    """Append a circuit record to JSONL storage."""

    try:
        if not isinstance(record, dict):
            return

        rec = dict(record)

        # 1. Normalize legacy keys
        _normalize_legacy_fields(rec)

        # 1b. Clean placeholder spans early
        _normalize_text_positions(rec)

        # 2. Enrich from LaTeX / PDF (adds new descriptions)
        # Also enrich when we have no meaningful text span alignment yet.
        if rec.get('page') is None or not _has_meaningful_text_positions(rec):
            _enrich_record_from_sources(rec)

        # 2b. Re-normalize spans after enrichment
        _normalize_text_positions(rec)

        # 3. Normalize descriptions ONCE
        _normalize_descriptions(rec)

        # 🔹 4. Unicode normalization (THIS IS WHERE YOUR LINE GOES)
        if isinstance(rec.get("descriptions"), list):
            for i, d in enumerate(rec["descriptions"]):
                if isinstance(d, str):
                    rec["descriptions"][i] = unicodedata.normalize("NFKC", d)

        if not _has_meaningful_descriptions(rec):
            return

        # 3b. Compute per-sentence PDF spans (best-effort)
        try:
            if rec.get("arxiv_id") and rec.get("page") and isinstance(rec.get("descriptions"), list) and rec["descriptions"]:
                page_text_pdf = _load_pdf_page_text(str(rec.get("arxiv_id")), int(rec.get("page")))
                spans = _align_description_items_to_pdf_spans(
                    page_text_pdf,
                    rec["descriptions"],
                    rec.get("figure_block_span"),
                )
                if spans:
                    rec["text_positions"] = spans
        except Exception:
            pass

        # Normalize positions again after alignment
        _normalize_text_positions(rec)

        # Optional classifier (now sees final clean text)
        _maybe_classify_record(rec)

        # Do not persist internal-only metadata
        _strip_internal_fields(rec)

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
    common_png_dir = _get_common_png_dir()
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
                        # do not store auxiliary/internal fields inside the value dict
                        clean = dict(rec)
                        for _f in (
                            'raw_block_file', 'block_id', 'block_name', 'image_filename',
                            'latex_text_positions', 'figure_block_span'
                        ):
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

                    # do not inject `image_filename` or internal fields into the stored value; keep it only as the key
                    clean2 = dict(rec)
                    for _f in (
                        'raw_block_file', 'block_id', 'block_name', 'image_filename',
                        'latex_text_positions', 'figure_block_span'
                    ):
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

    pdf_path = Path(IMAGE_PIPELINE_PDF_CACHE_DIR) / f"{arxiv_id}.pdf"
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
    anchor_re = re.compile(
    r"(?:^|[\.!?]\s+)(?:Fig\.?|Figure)\s*(S?\d+)\b",
    re.IGNORECASE
    )
    stop_re = re.compile(r"(?:^|[\.!?]\s+)(?:Fig\.?|Figure|Table)\s*\d+\b", re.IGNORECASE)

    page_candidates: list[tuple[int, str, int | None]] = []

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

    return None

def _extract_paragraph_after_figure(latex: str, caption: str) -> str | None:
    """
    Extract the first natural-language paragraph AFTER a figure,
    skipping all LaTeX environments and layout junk.
    """

    # Try exact match first
    idx = latex.find(caption)

    # Fallback: find the figure environment itself
    if idx == -1:
        m = re.search(r"\\begin\{figure\}", latex)
        if not m:
            return None
        idx = m.start()

    tail = latex[idx:]
    end_fig = tail.find(r"\end{figure}")
    if end_fig == -1:
        return None

    after = tail[end_fig + len(r"\end{figure}"):]

    # Remove ALL LaTeX environments generically
    after = strip_latex_environments(after)

    # Split into candidate paragraphs
    paragraphs = [p.strip() for p in after.split("\n\n") if p.strip()]

    for p in paragraphs:
        # Stop at structural boundaries
        if p.startswith("\\section"):
            break

        if p.startswith("\\subsection"):
            continue  # skip header, but keep scanning

        if p.startswith(("\\begin", "\\definition")):
            break

        cleaned = _sanitize_description_snippet(p, strip_latex=True)

        if cleaned and is_natural_language_paragraph(cleaned):
            return cleaned

    return None

def _extract_sentence_around(text: str, start: int, end: int) -> str:
    """Extract the full sentence containing a span."""
    left = max(
        text.rfind(".", 0, start),
        text.rfind("?", 0, start),
        text.rfind("!", 0, start),
    )
    right_candidates = [
        text.find(".", end),
        text.find("?", end),
        text.find("!", end),
    ]
    right_candidates = [r for r in right_candidates if r != -1]
    right = min(right_candidates) if right_candidates else len(text)

    sentence = text[left + 1 : right + 1].strip()

    # Reject diagram-like sentences
    if sum(not c.isalnum() and not c.isspace() for c in sentence) / max(len(sentence), 1) > 0.35:
        return ""

    return sentence




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
                _normalize_text_positions(rec)

                if arxiv_id and rec.get('arxiv_id') != arxiv_id:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                missing_page = rec.get('page') is None
                missing_positions = rec.get('descriptions') and not _has_meaningful_text_positions(rec)

                if (missing_page or missing_positions) and rec.get('descriptions'):
                    _enrich_record_from_sources(rec)
                    if missing_page and rec.get('page') is not None:
                        updated += 1

                # Re-normalize after enrichment
                _normalize_descriptions(rec)
                _normalize_text_positions(rec)

                # Fill per-sentence PDF spans when possible
                try:
                    if rec.get("arxiv_id") and rec.get("page") and isinstance(rec.get("descriptions"), list) and rec["descriptions"]:
                        page_text_pdf = _load_pdf_page_text(str(rec.get("arxiv_id")), int(rec.get("page")))
                        spans = _align_description_items_to_pdf_spans(
                            page_text_pdf,
                            rec["descriptions"],
                            rec.get("figure_block_span"),
                        )
                        if spans:
                            rec["text_positions"] = spans
                except Exception:
                    pass

                _normalize_text_positions(rec)

                _maybe_classify_record(rec)
                _strip_internal_fields(rec)
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
