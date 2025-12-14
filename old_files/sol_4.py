import io
import tarfile
import re
import os
import time
import requests
import pandas as pd

import logging
from logging.handlers import RotatingFileHandler

from pylatexenc.latex2text import LatexNodes2Text

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer

# =================================================
# SETTINGS (TUNABLE)
# =================================================

ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_test_50_preproc_cached"
CACHE_DIR = "arxiv_cache"
MAX_IMAGES = 50

REQUEST_DELAY = 15  # seconds
SIMILARITY_THRESHOLD = 0.28
USE_NEGATIVE_PENALTY = True
NEGATIVE_PENALTY_ALPHA = 1.0
# If True, count negatives only in caption (not context)
PENALIZE_CAPTION_ONLY = True
# Cap the penalty relative to raw similarity (e.g., 0.8 => penalty <= 80% of sim)
USE_PENALTY_CAP = True
PENALTY_CAP_RATIO = 0.8
# Small positive boost per protected token occurrence
USE_PROTECTED_TOKEN_BOOST = True
PROTECTED_TOKEN_BOOST = 0.10
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# ---------- Context controls (TUNABLE) ----------
USE_CONTEXT = True
CONTEXT_WINDOW_WORDS = 7                 # 7 words on both sides
CONTEXT_WEIGHT = 0.1                    # how much context influences similarity
USE_FIG_MENTION_CONTEXT = False           # try extracting context around "Fig." mentions
USE_FIG_BLOCK_CONTEXT = True             # window around the figure environment in TeX

QUERY_SETS = {
    "gate": """
        quantum circuit
        gate-level circuit
        cnot hadamard pauli rx ry rz
        control qubit target qubit ancilla
    """,

    "algorithm": """
        grover shor qft
        oracle circuit
        quantum algorithm
    """,

    "variational": """
        ansatz
        variational circuit
        vqe qaoa vqc
        parameterized circuit
    """,

    "decomposition": """
        gate decomposition
        circuit decomposition
        transpiled circuit
        synthesis
    """
}

# ---------- Text preprocessing controls ----------
USE_STEMMING = True
USE_STOPWORDS = True
NORMALIZE_HYPHENS = True

STEMMER = PorterStemmer()

PROTECTED_TOKENS = {
    "cnot", "cx", "cz",
    "rx", "ry", "rz",
    "qft", "qaoa", "vqe", "vqc",
    "iswap"
}

# ---------- Penalty controls ----------
NEGATIVE_RAW_TOKENS = {
    # plots & charts
    "plot", "graph", "chart", "histogram",
    "scatter", "bar", "boxplot", "violin",
    "heatmap", "contour", "surface",
    "curve", "trend", "profil", "exampl",
    "implement",
    "demonstr",
    "code", "kernel", "notebook", "script", "function",
    "cuda", "cpu", "gpu", "illustration", "pulse", "duration", "scatter",
    "energy", "level", "spectrum", "eigenvalu", "matrix", "spectrum", "variational", "numerics",
    "fidelity", "overlap", "correlation", "concurrence", "log", "coefficient", "covariance",
    "dataset", "benchmark", "simulation", "simul", "iqm", "qpu", "hardware", "outlier",
    "training", "test", "validation", "fold", "cross-valid", "bloch", "sphere",
    "data", "dyson", "fit", "regress", "classif", "clust", "latice", "lattice",
    "geometry", "graph", "network",

    # statistics
    "distribut", "probabl", "expect",
    "varianc", "mean", "averag",
    "standard", "deviat", "confid",
    "interval", "percent", "ratio",

    # evaluation
    "result", "perform", "accuraci",
    "error", "loss", "benchmark",
    "metric", "score", "evaluat",
    "compar", "improv", "gain",

    # physics quantities
    "energi", "fidel", "overlap",
    "spectrum", "spectra",
    "eigenvalu", "eigenstat",
    "amplitud", "phase",
    "frequenc", "reson",

    # simulation
    "simul", "numer", "comput",
    "trial", "sampl",
    "iteration", "epoch",
    "converg", "optim",

    # experimental
    "measur", "readout",
    "nois", "decoher",
    "calibr", "volt",
    "current", "signal",

    # abstract visuals
    "node", "edge",
    "layout", "topolog",
    "network", "connect"
}

NEGATIVE_TOKENS = {
    STEMMER.stem(token)
    for token in NEGATIVE_RAW_TOKENS
}

FILENAME_NEGATIVE_RAW = {
    "plot", "graph", "chart", "hist",
    "loss", "acc", "accuracy",
    "result", "results",
    "benchmark", "energy",
    "spectrum", "spectra",
    "prob", "distribution",
    "heatmap", "surface",
    "curve", "spectrum", "distribution",
    "simu", "simulation", "3d",
}

FILENAME_NEGATIVE_TOKENS = {
    STEMMER.stem(t) for t in FILENAME_NEGATIVE_RAW
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =================================================
# LOGGING (STANDARD SOLUTION)
# =================================================

LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")

logger = logging.getLogger("quantum_figure_miner")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# =================================================
# REGEX
# =================================================

FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)

# For figure mention context: "Fig. 3", "Figure 2", "Fig 5"
FIG_MENTION_RE = re.compile(r"\b(Fig\.?|Figure)\s*(\d+)\b", re.IGNORECASE)

# =================================================
# TEXT PREPROCESSING
# =================================================

def normalize_text(text):
    text = text.lower()
    if NORMALIZE_HYPHENS:
        text = re.sub(r"[-_/]", " ", text)
    return text


def stem_token(token):
    if token in PROTECTED_TOKENS:
        return token
    return STEMMER.stem(token)


def tfidf_analyzer(text):
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)

    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    if USE_STEMMING:
        tokens = [stem_token(t) for t in tokens]

    return tokens


def preprocess_text_to_string(text):
    return " ".join(tfidf_analyzer(text))


def count_negative_tokens(preprocessed_text):
    tokens = preprocessed_text.split()
    return sum(t in NEGATIVE_TOKENS for t in tokens)

# =================================================
# UTILS
# =================================================

def custom_text_cleaner(text: str) -> str:
    """
    Aggressive but controlled cleaner for Word2Vec.
    """

    text = text.lower()

    # Remove LaTeX math
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove scientific notation
    text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
    text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

    # Remove figure references
    text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

    # Handle parentheses
    def handle_parentheses(match):
        inside = match.group(1).strip()
        if len(inside.split()) <= 2:
            return " "  # remove short (a), (b), (1)
        return " " + inside + " "  # keep long explanation

    text = re.sub(r"\(([^)]*)\)", handle_parentheses, text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Normalize punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clear_output_dir(directory, extensions=SUPPORTED_EXT):
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in extensions):
            os.remove(fpath)


def read_arxiv_ids(filename):
    with open(filename, "r") as f:
        return [l.strip().replace("arXiv:", "") for l in f if l.strip()]


def download_source(arxiv_id):
    cache_path = os.path.join(CACHE_DIR, f"{arxiv_id}.tar.gz")

    if os.path.exists(cache_path):
        logger.info(f"📦 Using cached source for {arxiv_id}")
        with open(cache_path, "rb") as f:
            return io.BytesIO(f.read())

    logger.info(f"📥 Downloading {arxiv_id} (sleep {REQUEST_DELAY}s)")
    time.sleep(REQUEST_DELAY)

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(cache_path, "wb") as f:
                f.write(r.content)
            return io.BytesIO(r.content)
        else:
            logger.warning(f"❌ HTTP {r.status_code} for {arxiv_id}")
    except Exception as e:
        logger.warning(f"❌ Download error for {arxiv_id}: {e}")

    return None


def preprocess_filename(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    tokens = name.split()
    return [STEMMER.stem(t) for t in tokens]


def filename_is_negative(img_path):
    fname = os.path.basename(img_path)
    tokens = preprocess_filename(fname)
    return any(t in FILENAME_NEGATIVE_TOKENS for t in tokens)

# =================================================
# CONTEXT EXTRACTION
# =================================================

def _window_around_index(tokens, idx, window):
    left = tokens[max(0, idx - window): idx]
    right = tokens[idx + 1: idx + 1 + window]
    return " ".join(left + right)

def extract_fig_block_context(tex, fig_start, fig_end, window_words=7):
    """
    Takes a window of N words before and after the figure environment in the raw TeX.
    This is figure-local context (often includes words like 'circuit', 'oracle', etc.).
    """
    before = tex[max(0, fig_start - 800):fig_start]
    after = tex[fig_end: min(len(tex), fig_end + 800)]

    # tokenize roughly by words
    before_tokens = re.findall(r"[A-Za-z0-9]+", before)
    after_tokens = re.findall(r"[A-Za-z0-9]+", after)

    left = " ".join(before_tokens[-window_words:])
    right = " ".join(after_tokens[:window_words])

    return (left + " " + right).strip()

def extract_fig_mention_context(tex, fig_number, window_words=7):
    """
    Finds occurrences of 'Fig. <n>' / 'Figure <n>' and takes N words on both sides.
    If the paper text says 'As shown in Fig. 3, the quantum circuit ...', this captures it.
    """
    tokens = re.findall(r"[A-Za-z0-9\.]+", tex)
    contexts = []

    # Walk through tokens to find "Fig."/"Figure" + number patterns
    for i in range(len(tokens) - 1):
        t0 = tokens[i]
        t1 = re.sub(r"\D", "", tokens[i + 1])  # strip non-digits
        if re.fullmatch(r"(?i)fig\.?|figure", t0) and t1 == str(fig_number):
            # Context window around the mention (exclude the mention tokens themselves)
            left = tokens[max(0, i - window_words): i]
            right = tokens[i + 2: i + 2 + window_words]
            contexts.append(" ".join(left + right))

    return " ".join(contexts).strip()

def try_extract_figure_number_from_caption(caption_text):
    """
    Tries to detect 'Fig. 3' / 'Figure 3' in the caption itself.
    If it exists, we can use it to get mention-based context.
    """
    m = FIG_MENTION_RE.search(caption_text)
    if not m:
        return None
    return int(m.group(2))

# =================================================
# FIGURE EXTRACTION
# =================================================

def extract_figures_from_tex(tex):
    figures = []
    for match in FIG_RE.finditer(tex):
        block = match.group()
        cap = CAP_RE.search(block)
        img = IMG_RE.search(block)
        if not (cap and img):
            continue

        try:
            caption = LatexNodes2Text().latex_to_text(cap.group(1))
        except Exception:
            caption = cap.group(1)

        fig_context = ""
        if USE_CONTEXT and USE_FIG_BLOCK_CONTEXT:
            fig_context = extract_fig_block_context(
                tex, match.start(), match.end(), window_words=CONTEXT_WINDOW_WORDS
            )

        figures.append({
            "caption": caption.strip(),
            "img_path": img.group(1).strip(),
            "fig_block_context": fig_context,
            # mention_context will be added later once we know fig number (if any)
            "fig_mention_context": ""
        })
    return figures

# =================================================
# TFIDF FILTER (Caption + Context)
# =================================================

def tfidf_filter(figures):
    # We will compute similarity on:
    # - caption alone
    # - caption+context (caption plus context snippets)
    # and combine them (caption dominates, context small boost)

    captions = [f["caption"] for f in figures]

    all_query_text = "\n".join(QUERY_SETS.values())
    ALLOWED_VOCAB = set(tfidf_analyzer(all_query_text))

    vectorizer = TfidfVectorizer(
        analyzer=tfidf_analyzer,
        vocabulary=ALLOWED_VOCAB,
        ngram_range=(1, 2)
    )

    # Build combined text per figure
    combined_texts = []
    raw_contexts = []

    for f in figures:
        ctx_parts = []

        # Mention-based context (best if we can link a number)
        mention_ctx = ""
        if USE_CONTEXT and USE_FIG_MENTION_CONTEXT:
            fig_num = try_extract_figure_number_from_caption(f["caption"])
            if fig_num is not None:
                # NOTE: mention context needs the full tex, but we don't keep it here.
                # So we only fill mention_ctx if caller provides it. In our pipeline,
                # we fill mention_ctx during extraction by using fig_block_context.
                # If you want full mention context, we add it during per-tex extraction.
                pass

        # Always include fig-block local context if enabled
        if USE_CONTEXT and f.get("fig_block_context"):
            ctx_parts.append(f["fig_block_context"])

        raw_ctx = " ".join(ctx_parts).strip()
        raw_contexts.append(raw_ctx)

        combined = (f["caption"] + " " + raw_ctx).strip()
        combined_texts.append(combined)

    # Fit on: all combined figure texts + query sets (multi-query)
    tfidf = vectorizer.fit_transform(combined_texts + list(QUERY_SETS.values()))
    caption_vecs = tfidf[:len(combined_texts)]
    query_vecs = tfidf[len(combined_texts):]

    sims = cosine_similarity(caption_vecs, query_vecs)

    for i, f in enumerate(figures):
        # Preprocess caption and context separately for logging/EDA
        preproc_caption = preprocess_text_to_string(f["caption"])
        preproc_context = preprocess_text_to_string(raw_contexts[i]) if raw_contexts[i] else ""
        preproc_combined = preprocess_text_to_string(combined_texts[i])

        per_query = {name: float(sims[i, j]) for j, name in enumerate(QUERY_SETS.keys())}
        best_sim = max(per_query.values())
        best_query = max(per_query, key=per_query.get)

        # Negative count: caption-only or caption+context
        if PENALIZE_CAPTION_ONLY:
            neg_count = count_negative_tokens(preproc_caption)
        else:
            neg_count = count_negative_tokens(preproc_combined)

        # Optional filename-based soft penalty
        filename_neg = filename_is_negative(f["img_path"])
        filename_penalty = 0.0
        if filename_neg and USE_NEGATIVE_PENALTY:
            # apply a small fixed penalty for negative-looking filenames
            filename_penalty = NEGATIVE_PENALTY_ALPHA * 0.5

        # Base penalty
        penalty = (NEGATIVE_PENALTY_ALPHA * neg_count) if USE_NEGATIVE_PENALTY else 0.0
        penalty += filename_penalty

        # Cap penalty relative to raw similarity if enabled
        if USE_PENALTY_CAP:
            penalty = min(penalty, best_sim * PENALTY_CAP_RATIO)

        # Optional positive boost for protected tokens occurrences
        boost = 0.0
        if USE_PROTECTED_TOKEN_BOOST:
            tokens_all = preproc_combined.split()
            prot_hits = sum(t in PROTECTED_TOKENS for t in tokens_all)
            boost = PROTECTED_TOKEN_BOOST * prot_hits

        adjusted_sim = max(0.0, min(1.0, best_sim + boost) - penalty)

        f["raw_context"] = raw_contexts[i]
        f["raw_caption_plus_context"] = combined_texts[i]

        f["preprocessed_caption"] = preproc_caption
        f["preprocessed_context"] = preproc_context
        f["preprocessed_text"] = preproc_combined

        f["similarities"] = per_query
        f["best_query"] = best_query

        f["similarity_raw"] = best_sim
        f["negative_tokens"] = neg_count
        f["penalty"] = penalty
        f["protected_boost"] = boost
        f["filename_negative"] = filename_neg
        f["similarity"] = adjusted_sim

    return figures

# =================================================
# IMAGE EXTRACTION
# =================================================

def extract_images(tar, figures, pid, saved_set):
    members = {m.name: m for m in tar.getmembers()}
    extracted = []

    safe_pid = pid.replace("/", "_").replace(".", "_")

    for idx, f in enumerate(figures):
        base = f["img_path"]
        if filename_is_negative(base):
            continue

        for ext in [""] + SUPPORTED_EXT:
            candidate = base + ext
            if candidate in members:
                key = f"{pid}:{candidate}"
                if key in saved_set:
                    continue

                data = tar.extractfile(members[candidate]).read()
                fname = f"{safe_pid}_{idx}_{os.path.basename(candidate)}"
                out = os.path.join(OUTPUT_DIR, fname)

                with open(out, "wb") as w:
                    w.write(data)

                saved_set.add(key)
                extracted.append({
                    "file": out,
                    "img_name": os.path.basename(candidate),
                    "caption": f["caption"],
                    "preprocessed_text": f["preprocessed_text"],
                    "similarity": f["similarity"]
                })
                break

    return extracted

# =================================================
# MAIN
# =================================================

if __name__ == "__main__":
    logger.info("🧹 Clearing previously saved images...")
    clear_output_dir(OUTPUT_DIR)

    arxiv_ids = read_arxiv_ids(ID_FILE)

    papers_checked = 0
    papers_with_figures = 0
    papers_with_candidates = 0
    papers_with_extracted = 0
    total_figures_seen = 0

    total_saved = 0
    saved_uniques = set()

    text_records = []

    for pid in arxiv_ids:
        if total_saved >= MAX_IMAGES:
            break

        papers_checked += 1

        src = download_source(pid)
        if not src:
            continue

        try:
            tar = tarfile.open(fileobj=src, mode="r:gz")
        except Exception:
            continue

        figures = []
        for m in tar.getmembers():
            if m.name.endswith(".tex"):
                try:
                    tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                    figures.extend(extract_figures_from_tex(tex))
                except Exception:
                    pass

        if not figures:
            continue

        papers_with_figures += 1
        total_figures_seen += len(figures)

        figures = tfidf_filter(figures)
        figures = sorted(figures, key=lambda x: x["similarity"], reverse=True)

        logger.info("🔍 Top captions by similarity:")
        for i, f in enumerate(figures[:PRINT_TOP_CAPTIONS], start=1):
            logger.info(
                f"[{i}] sim={f['similarity']:.4f} "
                f"(raw={f['similarity_raw']:.4f}, neg={f['negative_tokens']}, "
                f"pen={f['penalty']:.4f}, boost={f.get('protected_boost',0.0):.4f}, "
                f"fname_neg={f.get('filename_negative', False)}, best={f['best_query']})"
            )
            logger.info(f"RAW CAPTION: {f['caption']}")
            if USE_CONTEXT and f.get("raw_context"):
                logger.info(f"RAW CONTEXT: {f['raw_context']}")
            logger.info(f"PREP CAPTION: {f['preprocessed_caption']}")
            if USE_CONTEXT and f.get("preprocessed_context"):
                logger.info(f"PREP CONTEXT: {f['preprocessed_context']}")
            logger.info(f"PREP COMBINED: {f['preprocessed_text']}")

        accepted = [f for f in figures if f["similarity"] >= SIMILARITY_THRESHOLD][:TOP_K_PER_PAPER]
        if accepted:
            papers_with_candidates += 1

        extracted = extract_images(tar, accepted, pid, saved_uniques)
        if extracted:
            papers_with_extracted += 1

        total_saved += len(extracted)

        extracted_lookup = {e["img_name"]: e for e in extracted}

        for f in figures:
            img_name = os.path.basename(f["img_path"])
            e = extracted_lookup.get(img_name)

            rec = {
                "paper_id": pid,
                "img_name": img_name,
                "image_path": e["file"] if e else None,

                "raw_caption": f["caption"],
                "raw_context": f.get("raw_context", ""),
                "raw_caption_plus_context": f.get("raw_caption_plus_context", ""),

                "preprocessed_caption": f.get("preprocessed_caption", ""),
                "preprocessed_context": f.get("preprocessed_context", ""),
                "preprocessed_text": f.get("preprocessed_text", ""),

                "similarity": f["similarity"],
                "similarity_raw": f["similarity_raw"],
                "negative_tokens": f["negative_tokens"],
                "penalty": f["penalty"],
                "best_query": f["best_query"],

                "selected": f["similarity"] >= SIMILARITY_THRESHOLD,
                "extracted": e is not None
            }

            for k, v in f["similarities"].items():
                rec[f"sim_{k}"] = v

            text_records.append(rec)

        logger.info(f"📑 {pid}")
        for e in extracted:
            logger.info(f"   ✔ {e['img_name']}  sim={e['similarity']:.3f}")
        logger.info(f"📊 Total saved: {total_saved}/{MAX_IMAGES}")

    df = pd.DataFrame(text_records)
    df_path = os.path.join(OUTPUT_DIR, "caption_text_log.csv")
    df.to_csv(df_path, index=False)

    logger.info("================ SUMMARY ================")
    logger.info(f"Papers checked: {papers_checked}")
    logger.info(f"Papers with figures: {papers_with_figures}")
    logger.info(f"Papers with candidates: {papers_with_candidates}")
    logger.info(f"Papers with extracted images: {papers_with_extracted}")
    logger.info(f"Total figures seen: {total_figures_seen}")
    logger.info(f"Total images saved: {total_saved}")
    logger.info(f"Caption log saved to: {df_path}")
    logger.info("=========================================")
