import re
import tarfile
from pathlib import Path
from typing import Optional

# In-memory cache of concatenated LaTeX sources keyed by arXiv id
_LATEX_SOURCE_CACHE: dict[str, str] = {}


def load_latex_source(arxiv_id: str, cache_dir: str | Path) -> Optional[str]:
    """Return concatenated LaTeX text from a cached tar.gz source for the paper.

    Reads all `.tex` members under `cache_dir/{arxiv_id}.tar.gz` and joins them.
    Keeps an in-memory cache per arXiv id to avoid repeated tar extraction.
    Returns None on any failure or when no `.tex` files are found.
    """
    if not arxiv_id:
        return None

    if arxiv_id in _LATEX_SOURCE_CACHE:
        return _LATEX_SOURCE_CACHE[arxiv_id]

    tar_path = Path(cache_dir) / f"{arxiv_id}.tar.gz"
    if not tar_path.exists():
        return None

    try:
        texts: list[str] = []
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".tex"):
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        texts.append(f.read().decode("utf-8", "ignore"))
                    finally:
                        f.close()
                except Exception:
                    continue
        if not texts:
            return None
        full_text = "\n\n".join(texts)
        _LATEX_SOURCE_CACHE[arxiv_id] = full_text
        return full_text
    except Exception:
        return None


def extract_context_snippet(raw_text: str, span: tuple[int, int], margin: int = 200, strip_latex: bool = True) -> str:
    """Return up to `margin` chars immediately after the caption span (skip the span itself).

    If `strip_latex` is True, drops comments and simple LaTeX commands/environments
    (non-nested) to make the snippet more readable. No backward context is included.
    """
    try:
        start, end = span
        # begin right after the caption span; do not include backward context
        start_ctx = min(len(raw_text), end)
        end_ctx = min(len(raw_text), end + margin)
        snippet = raw_text[start_ctx:end_ctx]

        if strip_latex:
            snippet = re.sub(r"%.*", "", snippet)
            snippet = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", snippet)

        snippet = snippet.strip()
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet
    except Exception:
        return ""


def find_figure_mentions(raw_text: str, figure_label: Optional[str], figure_number: Optional[int] = None) -> list[tuple[int, int]]:
    """Find LaTeX references to a figure.

    Prefers label-based references (\ref/\cref/\autoref/\pageref). If none
    are found and a figure number is provided, falls back to numeric "Fig./Figure N".
    Returns spans as (start, end) offsets in the raw LaTeX text.
    """
    spans: list[tuple[int, int]] = []
    try:
        if figure_label:
            lbl = re.escape(figure_label.strip())
            ref_pattern = re.compile(rf"(?:~|\\?\s)?\\(?:[cC]?[Rr]ef|[aA]utoref|[pP]ageref)\{{{lbl}\}}")
            spans.extend((m.start(), m.end()) for m in ref_pattern.finditer(raw_text))

        if not spans and figure_number is not None:
            num_pattern = re.compile(rf"\b(?:fig\.|figure)\s*{re.escape(str(figure_number))}\b", re.IGNORECASE)
            spans.extend((m.start(), m.end()) for m in num_pattern.finditer(raw_text))
    except Exception:
        return []

    return spans
