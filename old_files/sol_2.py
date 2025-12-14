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
SIMILARITY_THRESHOLD = 0.4
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# QUERY_TEXT = """
# quantum circuit
# qubit circuit
# gate array
# cnot
# hadamard
# controlled
# u-gate
# grover
# shor
# qaoa
# vqe
# vqc
# ansatz
# pauli-x
# pauli-y
# pauli-z
# x-gate
# y-gate
# z-gate
# rx
# ry
# rz
# s-gate
# t-gate
# toffoli
# fredkin
# """

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
        quantum algorithm circuit
    """,

    "variational": """
        ansatz
        variational circuit
        vqe qaoa vqc
        parameterized circuit
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

QUANTUM_POSITIVE_TOKENS = {
    "quantum", "qubit", "qbit", "gate", "cnot", "hadamard", "pauli",
    "superposition", "entanglement", "coherence", "decoherence",
    "algorithm", "grover", "shor", "qft", "vqe", "qaoa", "ansatz",
    "rx", "ry", "rz", "swap", "iswap", "toffoli", "fredkin",
    "measurement", "observable", "expectation", "fidelity"
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
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)

    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    if USE_STEMMING:
        tokens = [stem_token(t) for t in tokens]

    return tokens


def preprocess_text_to_string(text):
    return " ".join(tfidf_analyzer(text))

# =================================================
# UTILS
# =================================================

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

    ALLOWED_VOCAB = set(tfidf_analyzer(QUERY_TEXT))

    vectorizer = TfidfVectorizer(
    analyzer=tfidf_analyzer,
    vocabulary=ALLOWED_VOCAB,
    ngram_range=(1, 2))

    tfidf = vectorizer.fit_transform(texts + list(QUERY_SETS.values()))
    caption_vecs = tfidf[:len(texts)]
    query_vecs = tfidf[len(texts):]

    sims = cosine_similarity(caption_vecs, query_vecs)

    for i, f in enumerate(figures):
        f["similarities"] = {
            name: sims[i, j]
            for j, name in enumerate(QUERY_SETS.keys())
        }
        f["similarity"] = max(f["similarities"].values())

    return figures


def extract_images(tar, figures, pid, saved_set):
    members = {m.name: m for m in tar.getmembers()}
    extracted = []

    safe_pid = pid.replace("/", "_").replace(".", "_")

    for idx, f in enumerate(figures):
        base = f["img_path"]
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
            print(f"\n   [{i}] sim={f['similarity']:.4f}")
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

            text_records.append({
                "paper_id": pid,
                "img_name": img_name,
                "image_path": e["file"] if e else None,
                "raw_caption": f["caption"],
                "preprocessed_text": f["preprocessed_text"],
                "similarity": f["similarity"],
                "selected": f["similarity"] >= SIMILARITY_THRESHOLD,
                "extracted": e is not None
            })

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
