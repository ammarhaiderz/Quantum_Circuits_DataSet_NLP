import io
import tarfile
import re
import os
import time
import requests
import pandas as pd

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
NEGATIVE_PENALTY_ALPHA = 1.5
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

QUERY_SETS = {
    "circuit_core": """
        quantum circuit diagram
        circuit diagram
        gate based quantum circuit
        circuit representation
        gate sequence
        controlled gate
        qubit register
        circuit depth
    """,

    "gate_level": """
        cnot hadamard pauli
        rx ry rz
        controlled not
        control target qubit
        multi qubit gate
        ancilla qubit
    """,

    "algorithmic_circuits": """
        quantum algorithm circuit
        oracle circuit diagram
        grover circuit
        shor circuit
        qft circuit
    """,

    "variational_circuits": """
        variational quantum circuit
        parameterized circuit
        ansatz circuit
        vqe circuit
        qaoa circuit
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

# ---------- Penalty controls (TUNABLE) ----------

# These should match your stemming (Porter) output
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
    "training", "test", "validation", "fold", "cross-valid", "bloch", "sphere", "spherical", "spheric",
    "data", "dyson", "fit", "regress", "classif", "clust", "latice", "lattice", 
    "geometry", "graph", "network", "geometric", "time", "population", "ms","frequency", "domain", "duration", "mod", "modulus",
    "rate", "decay", "decoher", "nois", "signal", "volt", "current", "microsecond", "nanosecond", "millisecond",
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
    "simu", "simulation", "3d", "sphere", "spheric", "spherical",
    "duration", "time",

}

FILENAME_NEGATIVE_TOKENS = {
    STEMMER.stem(t) for t in FILENAME_NEGATIVE_RAW
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =================================================
# REGEX
# =================================================

FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)

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
    text = clean_caption_text(text)
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

def clean_caption_text(text: str) -> str:
    """
    Cleans LaTeX-heavy captions without destroying circuit vocabulary.
    """
    text = text.lower()

    # Remove LaTeX math
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove figure references
    text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

    # Remove scientific notation
    text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
    text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Normalize punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
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

    # ✅ Use cached version if available
    if os.path.exists(cache_path):
        print(f"📦 Using cached source for {arxiv_id}")
        with open(cache_path, "rb") as f:
            return io.BytesIO(f.read())

    # ⏳ Respect arXiv delay (ONLY when downloading)
    print(f"\n📥 Downloading {arxiv_id}")
    time.sleep(REQUEST_DELAY)

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(cache_path, "wb") as f:
                f.write(r.content)
            return io.BytesIO(r.content)
    except Exception as e:
        print("❌", e)

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


def extract_figures_from_tex(tex):
    figures = []
    for block in FIG_RE.findall(tex):
        cap = CAP_RE.search(block)
        img = IMG_RE.search(block)
        if not (cap and img):
            continue

        try:
            caption = LatexNodes2Text().latex_to_text(cap.group(1))
        except Exception:
            caption = cap.group(1)

        figures.append({
            "caption": caption.strip(),
            "img_path": img.group(1).strip()
        })
    return figures


def tfidf_filter(figures):
    texts = [f["caption"] for f in figures]

    # Build ALLOWED_VOCAB from ALL query sets (fixes your commented-out variable)
    all_query_text = "\n".join(QUERY_SETS.values())
    ALLOWED_VOCAB = set(tfidf_analyzer(all_query_text))

    vectorizer = TfidfVectorizer(
        analyzer=tfidf_analyzer,
        vocabulary=ALLOWED_VOCAB,
        ngram_range=(1, 2),
        use_idf=True,            # ✅ CRITICAL: Must be True
        smooth_idf=True,         # ✅ CRITICAL: Avoids division by zero
        sublinear_tf=False,      # Keep raw term frequency
        norm='l2'

    )

    tfidf = vectorizer.fit_transform(texts + list(QUERY_SETS.values()))
    caption_vecs = tfidf[:len(texts)]
    query_vecs = tfidf[len(texts):]

    sims = cosine_similarity(caption_vecs, query_vecs)

    for i, f in enumerate(figures):
        preproc = preprocess_text_to_string(f["caption"])

        per_query = {
            name: float(sims[i, j])
            for j, name in enumerate(QUERY_SETS.keys())
        }
        best_sim = max(per_query.values())
        best_query = max(per_query, key=per_query.get)

        # ---- Negative-token penalty (safe & clamped) ----

        neg_count = count_negative_tokens(preproc)

        if USE_NEGATIVE_PENALTY:
            penalty = NEGATIVE_PENALTY_ALPHA * neg_count
        else:
            penalty = 0.0

        # Clamp to keep similarity in a valid range
        adjusted_sim = max(0.0, best_sim - penalty)

        f["preprocessed_text"] = preproc
        f["similarities"] = per_query
        f["best_query"] = best_query

        f["similarity_raw"] = best_sim
        f["negative_tokens"] = neg_count
        f["penalty"] = penalty
        f["similarity"] = adjusted_sim

    return figures


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
    print("🧹 Clearing previously saved images...")
    clear_output_dir(OUTPUT_DIR)

    arxiv_ids = read_arxiv_ids(ID_FILE)

    # ---- Counters ----
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

        # ---------- DEBUG PRINT ----------
        print("\n   🔍 Top captions by similarity:")
        for i, f in enumerate(figures[:PRINT_TOP_CAPTIONS], start=1):
            # Full debug: adjusted + raw + penalty + best_query
            print(
                f"\n   [{i}] sim={f['similarity']:.4f} "
                f"(raw={f['similarity_raw']:.4f}, neg={f['negative_tokens']}, pen={f['penalty']:.4f}, best={f['best_query']})"
            )
            print("   RAW:")
            print("   ", f["caption"])
            print("   PREPROCESSED:")
            print("   ", f["preprocessed_text"])
        # ----------------------------------

        accepted = [
            f for f in figures
            if f["similarity"] >= SIMILARITY_THRESHOLD
        ][:TOP_K_PER_PAPER]

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
                "preprocessed_text": f["preprocessed_text"],
                "similarity": f["similarity"],
                "similarity_raw": f["similarity_raw"],
                "negative_tokens": f["negative_tokens"],
                "penalty": f["penalty"],
                "best_query": f["best_query"],
                "selected": f["similarity"] >= SIMILARITY_THRESHOLD,
                "extracted": e is not None
            }

            # also store per-query similarities in separate columns
            for k, v in f["similarities"].items():
                rec[f"sim_{k}"] = v

            text_records.append(rec)

        print(f"📑 {pid}")
        for e in extracted:
            print(f"   ✔ {e['img_name']}  sim={e['similarity']:.3f}")

        print(f"📊 Total saved: {total_saved}/{MAX_IMAGES}")

    # =================================================
    # SAVE DATAFRAME
    # =================================================

    df = pd.DataFrame(text_records)
    df_path = os.path.join(OUTPUT_DIR, "caption_text_log.csv")
    df.to_csv(df_path, index=False)

    print("\n================ SUMMARY ================")
    print(f"Papers checked: {papers_checked}")
    print(f"Papers with figures: {papers_with_figures}")
    print(f"Papers with candidates: {papers_with_candidates}")
    print(f"Papers with extracted images: {papers_with_extracted}")
    print(f"Total figures seen: {total_figures_seen}")
    print(f"Total images saved: {total_saved}")
    print(f"Caption log saved to: {df_path}")
    print("=========================================\n")
