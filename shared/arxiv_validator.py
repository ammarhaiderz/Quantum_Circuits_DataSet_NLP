"""arXiv category filter utilities with simple JSON caching."""

from __future__ import annotations

import json
import os
import time
from typing import Mapping, MutableMapping, Optional

import requests

from config.settings import REQUEST_DELAY, CACHE_DIR

_DEFAULT_CACHE_FILE = os.path.join(CACHE_DIR, "arxiv_category_cache.json")
_GLOBAL_CACHE: dict[str, bool] = {}


def load_cache(cache_file: str = _DEFAULT_CACHE_FILE) -> dict[str, bool]:
    """Load the arXiv category cache from disk.

    Parameters
    ----------
    cache_file : str, optional
        Path to the JSON cache file. Defaults to ``arxiv_category_cache.json``.

    Returns
    -------
    dict[str, bool]
        Mapping from arXiv ID (without version) to a boolean flag indicating
        whether the paper is in the ``quant-ph`` category.
    """

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            if isinstance(cache_data, dict):
                return {str(k): bool(v) for k, v in cache_data.items()}
        except Exception:
            return {}
    return {}


def save_cache(cache: Mapping[str, bool], cache_file: str = _DEFAULT_CACHE_FILE) -> None:
    """Persist the arXiv category cache to disk.

    Parameters
    ----------
    cache : Mapping[str, bool]
        Current cache mapping.
    cache_file : str, optional
        Destination JSON file. Defaults to ``arxiv_category_cache.json``.
    """

    try:
        cache_dir = os.path.dirname(cache_file) or "."
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        print(f"[WARN] Failed to save cache: {exc}")


def _get_global_cache(cache_file: str = _DEFAULT_CACHE_FILE) -> MutableMapping[str, bool]:
    """Return the module-level cache, loading it on first access."""

    if not _GLOBAL_CACHE:
        _GLOBAL_CACHE.update(load_cache(cache_file))
    return _GLOBAL_CACHE


def is_quantum_paper(
    arxiv_id: str,
    cache: Optional[MutableMapping[str, bool]] = None,
    cache_file: str = _DEFAULT_CACHE_FILE,
    session: Optional[requests.sessions.Session] = None,
    request_delay: float = REQUEST_DELAY / 2,
    timeout: float = 10.0,
) -> bool:
    """Check whether an arXiv paper belongs to the ``quant-ph`` category.

    The function consults a JSON-backed cache first. When a cache miss occurs,
    it queries the arXiv export API, updates the cache, and persists it.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier (version suffix allowed).
    cache : MutableMapping[str, bool], optional
        Cache mapping to use. If omitted, a module-level cache is used.
    cache_file : str, optional
        Path to the cache file for persistence. Defaults to
        ``arxiv_category_cache.json``.
    session : requests.sessions.Session, optional
        Optional requests session to reuse connections.
    request_delay : float, optional
        Delay in seconds inserted before the API call. Defaults to half of
        ``REQUEST_DELAY``.
    timeout : float, optional
        HTTP timeout in seconds. Defaults to 10.0.

    Returns
    -------
    bool
        ``True`` if the paper is categorized as ``quant-ph`` or if the API
        check fails (conservative). ``False`` otherwise.
    """

    cache_ref = cache if cache is not None else _get_global_cache(cache_file)

    if arxiv_id in cache_ref:
        return cache_ref[arxiv_id]

    result = check_arxiv_api(
        arxiv_id,
        session=session,
        request_delay=request_delay,
        timeout=timeout,
    )

    cache_ref[arxiv_id] = result
    save_cache(cache_ref, cache_file)
    return result


def check_arxiv_api(
    arxiv_id: str,
    *,
    session: Optional[requests.sessions.Session] = None,
    request_delay: float = REQUEST_DELAY / 2,
    timeout: float = 10.0,
) -> bool:
    """Query the arXiv export API for a paper's category membership.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier (version suffix allowed).
    session : requests.sessions.Session, optional
        Optional requests session to reuse connections.
    request_delay : float, optional
        Delay in seconds inserted before the API call. Defaults to half of
        ``REQUEST_DELAY``.
    timeout : float, optional
        HTTP timeout in seconds. Defaults to 10.0.

    Returns
    -------
    bool
        ``True`` if the paper appears to be in ``quant-ph``. If the request
        fails or returns a non-200 status code, the function returns ``True``
        (conservative default).
    """

    clean_id = arxiv_id.split("v")[0]

    try:
        time.sleep(request_delay)
        url = (
            "http://export.arxiv.org/api/query?id_list="
            f"{clean_id}&max_results=1"
        )
        req = session.get(url, timeout=timeout) if session else requests.get(url, timeout=timeout)

        if req.status_code == 200:
            return "quant-ph" in req.text

        print(f"[WARN] API returned {req.status_code} for {arxiv_id}")
        return True

    except Exception as exc:
        print(f"[WARN] API check failed for {arxiv_id}: {exc}")
        return True


def get_cache_stats(cache: Optional[Mapping[str, bool]] = None) -> dict[str, float]:
    """Summarize cache contents.

    Parameters
    ----------
    cache : Mapping[str, bool], optional
        Cache to summarize. If omitted, the module-level cache is used.

    Returns
    -------
    dict[str, float]
        ``total`` (int), ``quantum`` (int), ``non_quantum`` (int), and
        ``quantum_percentage`` (float, percent of quantum entries).
    """

    cache_ref = cache if cache is not None else _GLOBAL_CACHE or _get_global_cache()
    total = len(cache_ref)
    quantum = sum(1 for v in cache_ref.values() if v)
    non_quantum = total - quantum
    return {
        "total": total,
        "quantum": quantum,
        "non_quantum": non_quantum,
        "quantum_percentage": (quantum / total * 100) if total else 0.0,
    }
