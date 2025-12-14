import io
import tarfile
import re
import os
import time
import requests
from pylatexenc.latex2text import LatexNodes2Text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# SETTINGS (TUNABLE)
# -------------------------------------------------

ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_test_10"
MAX_IMAGES = 250

REQUEST_DELAY = 3.5  # seconds
SIMILARITY_THRESHOLD = 0.085
TOP_K_PER_PAPER = 10

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# QUERY_TEXT = """
# quantum circuit
# qubit
# gate
# cnot
# hadamard
# controlled
# measurement
# quantum fourier transform
# grover
# shor
# qaoa
# vqe
# ansatz
# error correction
# """

QUERY_TEXT = """
quantum circuit
qubit circuit 
gate array
cnot
hadamard
controlled
u-gate
grover
shor
qaoa
vqe
vqc
ansatz
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# REGEX
# -------------------------------------------------

FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)

# -------------------------------------------------
# UTILS
# -------------------------------------------------

def read_arxiv_ids(filename):
    with open(filename, "r") as f:
        return [l.strip().replace("arXiv:", "") for l in f if l.strip()]


def download_source(arxiv_id):
    print(f"\n📥 Downloading {arxiv_id}")
    time.sleep(REQUEST_DELAY)
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
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
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf = vectorizer.fit_transform(texts + [QUERY_TEXT])
    sims = cosine_similarity(tfidf[:-1], tfidf[-1])

    for i, f in enumerate(figures):
        f["similarity"] = float(sims[i])

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
                    "caption": f["caption"],
                    "similarity": f["similarity"]
                })
                break

    return extracted

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":

    arxiv_ids = read_arxiv_ids(ID_FILE)

    # ---------------- COUNTERS ----------------
    papers_checked = 0
    papers_with_figures = 0
    papers_with_candidates = 0
    papers_with_extracted = 0
    total_figures_seen = 0
    # ------------------------------------------

    total_saved = 0
    saved_uniques = set()

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
            print("   ⚠ No figures found")
            continue

        papers_with_figures += 1
        total_figures_seen += len(figures)

        figures = tfidf_filter(figures)
        figures = sorted(figures, key=lambda x: x["similarity"], reverse=True)

        # ---------------- DEBUG ----------------
        print("   🔍 Top similarity scores:")
        for f in figures[:3]:
            print(f"      sim={f['similarity']:.3f} | {f['caption'][:80]}")
        # ---------------------------------------

        accepted = [
            f for f in figures
            if f["similarity"] >= SIMILARITY_THRESHOLD
        ][:TOP_K_PER_PAPER]

        if accepted:
            papers_with_candidates += 1
        else:
            print("   ❌ No figures passed similarity threshold")

        extracted = extract_images(tar, accepted, pid, saved_uniques)

        if extracted:
            papers_with_extracted += 1

        total_saved += len(extracted)

        print(f"📑 {pid}")
        for e in extracted:
            print(f"   ✔ {os.path.basename(e['file'])}  sim={e['similarity']:.3f}")

        print(f"📊 Total saved: {total_saved}/{MAX_IMAGES}")

    # ---------------- SUMMARY ----------------
    print("\n================ SUMMARY ================")
    print(f"Papers checked: {papers_checked}")
    print(f"Papers with figures: {papers_with_figures}")
    print(f"Papers with candidates (passed threshold): {papers_with_candidates}")
    print(f"Papers with extracted images: {papers_with_extracted}")
    print(f"Total figures seen: {total_figures_seen}")
    print(f"Total images saved: {total_saved}")
    print("=========================================\n")
