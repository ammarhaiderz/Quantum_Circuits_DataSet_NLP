"""
One-shot quantum problem classification using SBERT similarity.

Public API:
- prepare_label_embeddings(model) -> (label_keys, label_embeddings)
- classify_quantum_problem(model, descriptions, label_keys, label_embeddings, threshold=0.45) -> str
- apply_classification(records, model, label_keys, label_embeddings, threshold=0.45) -> None

Notes:
- Does not mutate anything except `quantum_problem` when using apply_classification.
- Does not alter descriptions/gates/text_positions/page/figure.
- Idempotent: safe to re-run.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer, util

from config.quantum_problem_labels import QUANTUM_PROBLEM_LABELS

DEFAULT_THRESHOLD = 0.45
UNSPECIFIED = "Unspecified quantum circuit"


def prepare_label_embeddings(
    model: SentenceTransformer,
) -> Tuple[List[str], np.ndarray]:
    """Encode label descriptions once and return ``(keys, embeddings)``.

    Embeddings are L2-normalized for cosine similarity.

    Parameters
    ----------
    model : SentenceTransformer
        SBERT model used to encode label descriptions.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        Ordered label keys and their normalized embeddings.
    """
    label_keys = list(QUANTUM_PROBLEM_LABELS.keys())
    label_texts = list(QUANTUM_PROBLEM_LABELS.values())

    # model.encode can directly return normalized vectors
    label_embeddings = model.encode(
        label_texts,
        normalize_embeddings=True,
    )
    # Ensure numpy array for util.cos_sim compatibility
    if not isinstance(label_embeddings, np.ndarray):
        label_embeddings = np.asarray(label_embeddings)
    return label_keys, label_embeddings


def _concat_descriptions(descriptions: Iterable[str]) -> str:
    """Concatenate non-empty description strings into a single query.

    Parameters
    ----------
    descriptions : Iterable[str]
        Candidate description strings.

    Returns
    -------
    str
        Space-joined text of valid descriptions (may be empty).
    """
    return " ".join([d for d in descriptions if isinstance(d, str) and d.strip()]).strip()


def classify_quantum_problem(
    model: SentenceTransformer,
    descriptions: Sequence[str],
    label_keys: Sequence[str],
    label_embeddings: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Return best label or ``UNSPECIFIED`` using SBERT cosine similarity.

    Parameters
    ----------
    model : SentenceTransformer
        SBERT model for encoding the query text.
    descriptions : Sequence[str]
        Descriptions to concatenate into the query.
    label_keys : Sequence[str]
        Ordered label keys aligned with ``label_embeddings``.
    label_embeddings : ArrayLike
        Precomputed, normalized embeddings of label texts.
    threshold : float, optional
        Minimum cosine similarity required to accept a label (default ``0.45``).

    Returns
    -------
    str
        Best matching label or ``UNSPECIFIED`` when below threshold or missing text.

    Notes
    -----
    Pure function: does not mutate inputs.
    """
    if not descriptions:
        return UNSPECIFIED

    query_text = _concat_descriptions(descriptions)
    if not query_text:
        return UNSPECIFIED

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(query_embedding, label_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return label_keys[best_idx]
    return UNSPECIFIED


def apply_classification(
    records: Sequence[dict],
    model: SentenceTransformer,
    label_keys: Sequence[str],
    label_embeddings: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Assign ``quantum_problem`` in-place for each record.

    Parameters
    ----------
    records : Sequence[dict]
        Mutable records that may contain ``descriptions``.
    model : SentenceTransformer
        SBERT model for encoding descriptions.
    label_keys : Sequence[str]
        Ordered label keys aligned with embeddings.
    label_embeddings : ArrayLike
        Precomputed, normalized embeddings of label texts.
    threshold : float, optional
        Minimum cosine similarity to accept a label (default ``0.45``).

    Returns
    -------
    None

    Notes
    -----
    Only ``quantum_problem`` is written; other fields remain unchanged.
    """
    for rec in records:
        try:
            descs = rec.get("descriptions") if isinstance(rec, dict) else None
            label = classify_quantum_problem(model, descs or [], label_keys, label_embeddings, threshold)
            rec["quantum_problem"] = label
        except Exception:
            # Fail-safe: leave as unspecified on error
            try:
                rec["quantum_problem"] = UNSPECIFIED
            except Exception:
                pass
