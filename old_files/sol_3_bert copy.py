import io
import tarfile
import re
import os
import time
import requests
import pandas as pd
import torch  # Added for memory management

from pylatexenc.latex2text import LatexNodes2Text

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer
# === Sentence-BERT integration ===
from sentence_transformers import SentenceTransformer, util
import numpy as np

# =================================================
# SETTINGS (TUNABLE)
# =================================================

ID_FILE = "paper_list_36_.txt"
OUTPUT_DIR = "images_test_50_preproc_cached_"
CACHE_DIR = "arxiv_cache"
MAX_IMAGES = 50

REQUEST_DELAY = 15  # seconds
SIMILARITY_THRESHOLD = 0.3
USE_NEGATIVE_PENALTY = True
NEGATIVE_PENALTY_ALPHA = 1.9
SBERT_MIN_SIM = 0.4  # Adjusted for SBERT (typically 0.5-0.8 range)
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

QUERY_SETS = {
    "circuit_core": """
        quantum circuit diagram
        qubit circuit
        gate based quantum circuit
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
        quantum algorithm 
        oracle
        grover 
        shor 
        qft
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

QUANTUM_POSITIVE_TOKENS = {
    "quantum", "qubit", "qbit", "gate", "cnot", "hadamard", "pauli",
    "superposition", "entanglement", "coherence", "decoherence",
    "algorithm", "grover", "shor", "qft", "vqe", "qaoa", "ansatz",
    "rx", "ry", "rz", "swap", "iswap", "toffoli", "fredkin",
    "measurement", "observable", "expectation", "fidelity"
}
# ---------- Penalty controls (TUNABLE) ----------

# These should match your stemming (Porter) output
NEGATIVE_RAW_TOKENS = {
    # plots & charts
    "plot", "graph", "chart", "histogram",
    "scatter", "bar", "boxplot", "violin",
    "heatmap", "contour", "surface",
    "curve", "trend", "profil", "exampl",
    "implement", "flowchart",
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
    "sparsity", "histogram", "bin", "sparse",
    

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

    # Electrical/electronic (will be stemmed: resistor -> resistor, etc.)
    "resistor", "capacitor", "inductor", "transistor", "diode", "amplifier",
    "voltage", "current", "frequency", "signal", "impedance", "resistance",
    "capacitance", "inductance", "transmission", "power", "supply",
    "battery", "switch", "relay", "motor", "generator", "transformer",
    "analog", "digital", "pulse", "waveform", "amplitude", "phase",
    "ac", "dc", "alternating", "direct", "oscillator", "filter",
    "opamp", "operational", "mosfet", "bjt", "thyristor", "sensor",
    
    # 3D terms
    "3d", "three", "dimensional", "isometric", "perspective",
    "render", "rendering", "visualization", "volume", "mesh",
    "wireframe", "solid", "shaded", "lit", "lighting", "camera",
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
    REMOVES 'circuit' ONLY if no quantum terms are present.
    """
    text = text.lower()

    # Check if we should remove "circuit"
    has_quantum_term = any(term in text for term in QUANTUM_POSITIVE_TOKENS)
    
    # Remove LaTeX math
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove figure references
    text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

    # Remove scientific notation
    text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
    text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # NORMALIZE "circuit" word based on quantum context
    if not has_quantum_term:
        # Remove "circuit" if no quantum terms
        text = re.sub(r"\bcircuit(s)?\b", " ", text)
    else:
        # Keep "circuit" but normalize it (optional)
        text = re.sub(r"\bcircuit(s)?\b", " quantum_circuit ", text)

    # Normalize punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_sbert_model():
    """
    Loads Sentence-BERT model (small, fast model suitable for your task).
    Model will be downloaded on first run, then cached locally.
    """
    print("📦 Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Test the model
        test_embed = model.encode("test", convert_to_tensor=True)
        print(f"✅ SBERT model loaded successfully (embedding dim: {test_embed.shape[0]})")
        return model
    except Exception as e:
        print(f"❌ Failed to load SBERT model: {e}")
        print("Try: pip install sentence-transformers")
        raise


def prepare_sbert_queries(sbert_model):
    """
    Prepare optimized query embeddings for SBERT.
    Each query set is encoded as the average of its individual lines.
    """
    query_embeds = {}
    
    for name, query_text in QUERY_SETS.items():
        # Split into individual query lines
        lines = [line.strip() for line in query_text.strip().split('\n') if line.strip()]
        
        if not lines:
            print(f"⚠️ Warning: Empty query set for {name}")
            continue
            
        try:
            # Encode each line separately
            line_embeds = sbert_model.encode(lines, convert_to_tensor=True)
            
            # Average embeddings (good for general similarity)
            avg_embed = torch.mean(line_embeds, dim=0)
            
            query_embeds[name] = avg_embed
            print(f"✅ Encoded query set '{name}' with {len(lines)} lines")
            
        except Exception as e:
            print(f"❌ Failed to encode query set '{name}': {e}")
            # Fallback: encode as single string
            combined_text = " ".join(lines)
            query_embeds[name] = sbert_model.encode(combined_text, convert_to_tensor=True)
    
    return query_embeds


def sbert_rerank(figures, sbert_model, query_embeds):
    """
    Re-rank candidates using Sentence-BERT similarity.
    IMPORTANT: Do NOT replace TF-IDF acceptance logic.
    Only produce sbert_sim for ordering / secondary filtering.
    """
    if not figures or not query_embeds:
        return figures
    
    # Initialize SBERT fields for all figures to avoid KeyError
    for f in figures:
        f["best_sbert_query"] = None
        f["sbert_sim"] = 0.0
    
    # Extract captions for batch processing
    captions = [f["caption"] for f in figures]
    
    try:
        # Encode all captions in batch (much faster)
        caption_embeds = sbert_model.encode(
            captions, 
            convert_to_tensor=True, 
            show_progress_bar=False,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        # Process each figure
        for i, f in enumerate(figures):
            best = -1.0  # Start with negative value since cosine can be negative
            best_query = None
            
            # Compare with each query embedding
            for q_name, q_embed in query_embeds.items():
                # Use cosine similarity directly on tensors
                sim = util.cos_sim(caption_embeds[i], q_embed).item()
                if sim > best:
                    best = float(sim)
                    best_query = q_name
            
            f["sbert_sim"] = best
            f["best_sbert_query"] = best_query
        
        # Clear GPU memory if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"⚠️ SBERT reranking failed: {e}")
        print("⚠️ Continuing with TF-IDF scores only")
        # Keep figures with default values
    
    return figures


# def clean_caption_text(text: str) -> str:
#     """
#     Cleans LaTeX-heavy captions without destroying circuit vocabulary.
#     """
#     text = text.lower()

#     # Remove LaTeX math
#     text = re.sub(r"\$.*?\$", " ", text)

#     # Remove figure references
#     text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

#     # Remove scientific notation
#     text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
#     text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

#     # Remove standalone numbers
#     text = re.sub(r"\b\d+\b", " ", text)

#     # Normalize punctuation
#     text = re.sub(r"[^a-z0-9\s]", " ", text)

#     # Normalize whitespace
#     text = re.sub(r"\s+", " ", text).strip()

#     return text


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
                # ✅ FIXED: Use .get() safely for all SBERT fields
                extracted.append({
                    "file": out,
                    "img_name": os.path.basename(candidate),
                    "caption": f["caption"],
                    "preprocessed_text": f["preprocessed_text"],
                    "similarity": f["similarity"],
                    "sbert_sim": f.get("sbert_sim", 0.0),
                    "best_sbert_query": f.get("best_sbert_query", None)
                })
                break

    return extracted


def test_sbert_implementation():
    """Test SBERT integration before full run."""
    print("\n🧪 Testing SBERT implementation...")
    
    try:
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = ["quantum circuit diagram", "gate sequence"]
        test_captions = [
            "Circuit diagram showing CNOT gates",
            "Figure 3: Energy levels of the system",
            "The quantum circuit implementation with Hadamard and CNOT gates"
        ]
        
        # Encode
        query_embeds = model.encode(test_queries, convert_to_tensor=True, normalize_embeddings=True)
        caption_embeds = model.encode(test_captions, convert_to_tensor=True, normalize_embeddings=True)
        
        # Compute similarities
        similarities = util.cos_sim(caption_embeds, query_embeds)
        
        print("\nTest Results:")
        for i, caption in enumerate(test_captions):
            print(f"\nCaption: {caption}")
            for j, query in enumerate(test_queries):
                sim = similarities[i][j].item()
                print(f"  Similarity to '{query}': {sim:.4f}")
        
        # Check score ranges
        print(f"\n✅ Score range: {similarities.min().item():.4f} to {similarities.max().item():.4f}")
        print("📝 Typical circuit-related captions score 0.3-0.8 with relevant queries")
        print(f"🔧 SBERT_MIN_SIM = {SBERT_MIN_SIM} should work well")
        
        # Clear test tensors
        del query_embeds, caption_embeds, similarities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"❌ SBERT test failed: {e}")
        print("Try: pip install sentence-transformers")
        return False


# =================================================
# MAIN
# =================================================

if __name__ == "__main__":
    print("🧹 Clearing previously saved images...")
    clear_output_dir(OUTPUT_DIR)
    
    # Test SBERT first
    if not test_sbert_implementation():
        print("❌ SBERT test failed. Exiting.")
        exit(1)
    
    # =================================================
    # Load Sentence-BERT model and prepare query embeddings
    # =================================================
    
    sbert_model = load_sbert_model()
    
    # Prepare optimized query embeddings for SBERT
    QUERY_EMBEDS_SBERT = prepare_sbert_queries(sbert_model)
    
    # Verify we have query embeddings
    if not QUERY_EMBEDS_SBERT:
        print("❌ No query embeddings created. Check your QUERY_SETS.")
        exit(1)
    
    print(f"✅ Prepared {len(QUERY_EMBEDS_SBERT)} query embeddings")
    
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

        # ---------- DEBUG PRINT (TF-IDF ONLY) ----------
        print("\n   🔍 Top captions by TF-IDF:")
        for i, f in enumerate(figures[:PRINT_TOP_CAPTIONS], start=1):
            print(
                f"\n   [{i}] tfidf={f['similarity']:.4f} "
                f"(raw={f['similarity_raw']:.4f}, neg={f['negative_tokens']}, pen={f['penalty']:.4f})"
            )
            print("   RAW:")
            print("   ", f["caption"])
            print("   PREPROCESSED:")
            print("   ", f["preprocessed_text"])
        # ----------------------------------

        # =================================================
        # 1️⃣ TF-IDF HARD GATE (OLD BEHAVIOR PRESERVED)
        # =================================================
        accepted_tfidf = [
            f for f in figures
            if f["similarity"] >= SIMILARITY_THRESHOLD
        ]

        if accepted_tfidf:
            papers_with_candidates += 1

        # Limit pool size for semantic rerank (efficiency + precision)
        accepted_tfidf = accepted_tfidf[:TOP_K_PER_PAPER * 3]

        # =================================================
        # 2️⃣ Sentence-BERT SEMANTIC RE-RANK (ORDERING ONLY)
        # =================================================
        if accepted_tfidf:
            accepted_tfidf = sbert_rerank(
                accepted_tfidf,
                sbert_model=sbert_model,
                query_embeds=QUERY_EMBEDS_SBERT
            )

            accepted_tfidf = sorted(
                accepted_tfidf,
                key=lambda x: x.get("sbert_sim", 0.0),
                reverse=True
            )

        # =================================================
        # 3️⃣ FINAL SELECTION
        # =================================================
        accepted = accepted_tfidf[:TOP_K_PER_PAPER]
        accepted = [
            f for f in accepted_tfidf
            if f.get("sbert_sim", 0.0) >= SBERT_MIN_SIM
        ]

        # =================================================
        # 4️⃣ IMAGE EXTRACTION (UNCHANGED)
        # =================================================
        extracted = extract_images(tar, accepted, pid, saved_uniques)

        if extracted:
            papers_with_extracted += 1

        total_saved += len(extracted)

        # =================================================
        # 5️⃣ LOGGING / RECORD KEEPING (Enhanced for debugging)
        # =================================================
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
                "sbert_sim": f.get("sbert_sim", None),
                "best_sbert_query": f.get("best_sbert_query", None),
                "selected": f in accepted,
                "extracted": e is not None
            }

            for k, v in f["similarities"].items():
                rec[f"sim_{k}"] = v

            text_records.append(rec)

        # =================================================
        # 6️⃣ ENHANCED DEBUG OUTPUT
        # =================================================
        print(f"\n📑 {pid}")
        print(f"   Figures found: {len(figures)} | TF-IDF candidates: {len(accepted_tfidf)} | Selected: {len(accepted)}")
        
        if accepted:
            print("\n   🔍 Top SBERT similarities for selected images:")
            for i, f in enumerate(accepted[:min(3, len(accepted))]):
                sbert_score = f.get("sbert_sim", 0.0)
                sbert_query = f.get("best_sbert_query", "None")
                print(f"   [{i+1}] SBERT={sbert_score:.4f} (query: {sbert_query}) | TF-IDF={f['similarity']:.4f}")
                print(f"       Caption: {f['caption'][:80]}...")
        
        for e in extracted:
            print(f"   ✔ {e['img_name']}  tfidf={e['similarity']:.3f} | sbert={e.get('sbert_sim', 0.0):.3f}")

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
    
    # =================================================
    # DEBUG STATISTICS
    # =================================================
    if not df.empty and "sbert_sim" in df.columns:
        print("\n================ DEBUG STATISTICS ================")
        print(f"Average TF-IDF similarity: {df['similarity'].mean():.4f}")
        
        sbert_scores = df['sbert_sim'].dropna()
        if len(sbert_scores) > 0:
            print(f"Average SBERT similarity: {sbert_scores.mean():.4f}")
            print(f"Min SBERT similarity: {sbert_scores.min():.4f}")
            print(f"Max SBERT similarity: {sbert_scores.max():.4f}")
            print(f"Selected images with SBERT >= {SBERT_MIN_SIM}: {(sbert_scores >= SBERT_MIN_SIM).sum()}")
            
            # Show distribution
            print("\nSBERT Score Distribution:")
            for threshold in [0.0, 0.2, 0.4, 0.6, 0.8]:
                count = (sbert_scores >= threshold).sum()
                print(f"  ≥{threshold:.1f}: {count} images")
        else:
            print("⚠️ No SBERT scores available")
    
    print("=========================================\n")