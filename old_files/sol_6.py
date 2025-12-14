import io
import tarfile
import re
import os
import time
import requests
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

from pylatexenc.latex2text import LatexNodes2Text
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import gensim.downloader as gensim_api

# =================================================
# SETTINGS
# =================================================

ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_tfidf_glove"
CACHE_DIR = "arxiv_cache"

MAX_IMAGES = 50
REQUEST_DELAY = 15

TFIDF_THRESHOLD = 0.12
GLOVE_THRESHOLD = 0.28
NEGATIVE_PENALTY_ALPHA = 1.5

TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

USE_CONTEXT = True
USE_NEGATIVE_PENALTY = True
DOWNLOAD_IMAGES = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =================================================
# LOGGING (UNICODE SAFE)
# =================================================

LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")

logger = logging.getLogger("tfidf_glove")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(message)s")

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =================================================
# QUERY SETS
# =================================================

QUERY_SETS = {
    "gate": "quantum circuit gate cnot hadamard rx ry rz ancilla",
    "algorithm": "grover shor qft oracle quantum algorithm",
    "variational": "ansatz vqe qaoa variational circuit",
    "decomposition": "gate decomposition transpiled synthesis"
}

# =================================================
# TEXT CLEANING
# =================================================

STEMMER = PorterStemmer()

PROTECTED_TOKENS = {
    "cnot","cx","cz","rx","ry","rz","qft","qaoa","vqe","vqc","iswap"
}

NEGATIVE_RAW = {
    "plot","graph","chart","heatmap","distribution","probability",
    "result","accuracy","loss","benchmark","energy","spectrum",
    "data","simulation","hardware","training","testing","bloch",
    "measurement","noise","fidelity"
}

NEGATIVE_TOKENS = {STEMMER.stem(t) for t in NEGATIVE_RAW}

FILENAME_NEGATIVE_RAW = {
    "plot","graph","hist","loss","acc","benchmark",
    "spectrum","distribution","simulation","heatmap"
}

FILENAME_NEGATIVE_TOKENS = {STEMMER.stem(t) for t in FILENAME_NEGATIVE_RAW}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
    text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)
    text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

    def paren(m):
        inside = m.group(1).strip()
        return inside if len(inside.split()) > 2 else " "

    text = re.sub(r"\(([^)]*)\)", paren, text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    tokens = clean_text(text).split()
    out = []
    for t in tokens:
        if t in PROTECTED_TOKENS:
            out.append(t)
        elif t not in ENGLISH_STOP_WORDS:
            out.append(STEMMER.stem(t))
    return out

# =================================================
# MODELS
# =================================================

def load_glove():
    logger.info("[MODEL] Loading GloVe (100d, inference-only)")
    return gensim_api.load("glove-wiki-gigaword-100")

def sent_embedding(tokens, model):
    vecs = [model[t] for t in tokens if t in model]
    return np.mean(vecs, axis=0) if vecs else None

# =================================================
# FIGURE EXTRACTION
# =================================================

FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)

def extract_figures(tex):
    figs = []
    for m in FIG_RE.finditer(tex):
        block = m.group()
        cap = CAP_RE.search(block)
        img = IMG_RE.search(block)
        if not (cap and img):
            continue

        caption = LatexNodes2Text().latex_to_text(cap.group(1))
        ctx = tex[max(0, m.start()-600):m.start()] if USE_CONTEXT else ""

        figs.append({
            "caption": caption.strip(),
            "context": ctx,
            "img_path": img.group(1).strip()
        })
    return figs

# =================================================
# IMAGE EXTRACTION
# =================================================

def filename_is_negative(path):
    name = os.path.basename(path).lower()
    tokens = [STEMMER.stem(t) for t in re.findall(r"[a-z]+", name)]
    return any(t in FILENAME_NEGATIVE_TOKENS for t in tokens)

def extract_image(tar, fig, pid, idx, saved_set):
    members = {m.name: m for m in tar.getmembers()}

    base = fig["img_path"]
    for ext in [""] + SUPPORTED_EXT:
        candidate = base + ext
        if candidate in members:
            key = f"{pid}:{candidate}"
            if key in saved_set:
                return None

            data = tar.extractfile(members[candidate]).read()
            safe_pid = pid.replace("/", "_").replace(".", "_")
            fname = f"{safe_pid}_{idx}_{os.path.basename(candidate)}"
            out_path = os.path.join(OUTPUT_DIR, fname)

            with open(out_path, "wb") as f:
                f.write(data)

            saved_set.add(key)
            return out_path
    return None

# =================================================
# MAIN PIPELINE
# =================================================

def main():
    glove = load_glove()

    query_tokens = {k: tokenize(v) for k, v in QUERY_SETS.items()}
    query_embs = {k: sent_embedding(v, glove) for k, v in query_tokens.items()}

    tfidf = TfidfVectorizer(
        analyzer=lambda x: tokenize(x),
        vocabulary=set(sum(query_tokens.values(), [])),
        encoding="utf-8"
    )

    with open(ID_FILE, encoding="utf-8") as f:
        arxiv_ids = [l.strip() for l in f if l.strip()]

    saved = 0
    saved_uniques = set()
    records = []

    for pid in arxiv_ids:
        if saved >= MAX_IMAGES:
            break

        cache = os.path.join(CACHE_DIR, f"{pid}.tar.gz")
        if not os.path.exists(cache):
            time.sleep(REQUEST_DELAY)
            r = requests.get(f"https://arxiv.org/e-print/{pid}", timeout=30)
            if r.status_code != 200:
                continue
            with open(cache, "wb") as f:
                f.write(r.content)

        tar = tarfile.open(cache, "r:gz")

        figures = []
        for m in tar.getmembers():
            if m.name.endswith(".tex"):
                tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                figures.extend(extract_figures(tex))

        if not figures:
            continue

        texts = [f["caption"] + " " + f["context"] for f in figures]
        tfidf_mat = tfidf.fit_transform(texts + list(QUERY_SETS.values()))
        sims = cosine_similarity(tfidf_mat[:-len(QUERY_SETS)], tfidf_mat[-len(QUERY_SETS):])

        for i, f in enumerate(figures):
            if saved >= MAX_IMAGES:
                break

            if sims[i].max() < TFIDF_THRESHOLD:
                continue

            tokens = tokenize(texts[i])
            emb = sent_embedding(tokens, glove)
            if emb is None:
                continue

            glove_sims = {
                k: cosine_similarity([emb], [query_embs[k]])[0][0]
                for k in query_embs if query_embs[k] is not None
            }

            best_query = max(glove_sims, key=glove_sims.get)
            raw = glove_sims[best_query]

            neg = sum(t in NEGATIVE_TOKENS for t in tokens)
            penalty = NEGATIVE_PENALTY_ALPHA * neg if USE_NEGATIVE_PENALTY else 0.0
            final = max(0.0, raw - penalty)

            if final < GLOVE_THRESHOLD:
                continue

            if filename_is_negative(f["img_path"]):
                continue

            image_path = None
            if DOWNLOAD_IMAGES:
                image_path = extract_image(
                    tar, f, pid, i, saved_uniques
                )

            saved += 1

            records.append({
                "paper_id": pid,
                "caption": f["caption"],
                "image_path": image_path,
                "similarity": final,
                "similarity_raw": raw,
                "best_query": best_query,
                "negative_tokens": neg
            })

            logger.info(
                "[OK] sim=%.3f raw=%.3f neg=%d | %s",
                final, raw, neg, f["caption"]
            )

    pd.DataFrame(records).to_csv(
        os.path.join(OUTPUT_DIR, "caption_text_log.csv"),
        index=False
    )

    logger.info("DONE")

if __name__ == "__main__":
    main()
