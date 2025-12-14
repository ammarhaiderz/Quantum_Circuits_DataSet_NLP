"""
Utility functions for arXiv download and file operations.
"""
import os
import time
import requests
import pandas as pd
import io

from config import CACHE_DIR, REQUEST_DELAY, OUTPUT_DIR, SUPPORTED_EXT


def clear_output_dir(directory=OUTPUT_DIR, extensions=SUPPORTED_EXT):
    """
    Clear the output directory of previously saved images.
    """
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in extensions):
            os.remove(fpath)


def read_arxiv_ids(filename):
    """
    Read arXiv IDs from file.
    """
    with open(filename, "r") as f:
        return [l.strip().replace("arXiv:", "") for l in f if l.strip()]


def download_source(arxiv_id):
    """
    Download arXiv source or load from cache.
    """
    cache_path = os.path.join(CACHE_DIR, f"{arxiv_id}.tar.gz")

    # Use cached version if available
    if os.path.exists(cache_path):
        print(f"📦 Using cached source for {arxiv_id}")
        with open(cache_path, "rb") as f:
            return io.BytesIO(f.read())

    # Respect arXiv delay (ONLY when downloading)
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
        print(f"❌ Download failed: {e}")

    return None


def save_results(text_records, output_dir=OUTPUT_DIR):
    """
    Save results to CSV file.
    """
    df = pd.DataFrame(text_records)
    df_path = os.path.join(output_dir, "caption_text_log.csv")
    df.to_csv(df_path, index=False)
    return df_path


def print_debug_info(figures, top_n=5):
    """
    Print debug information about top captions.
    """
    print("\n   🔍 Top captions by TF-IDF:")
    for i, f in enumerate(figures[:top_n], start=1):
        print(
            f"\n   [{i}] tfidf={f['similarity']:.4f} "
            f"(raw={f['similarity_raw']:.4f}, neg={f['negative_tokens']}, pen={f['penalty']:.4f})"
        )
        print("   RAW:")
        print("   ", f["caption"])
        print("   PREPROCESSED:")
        print("   ", f["preprocessed_text"])


def print_paper_summary(pid, figures, accepted_tfidf, accepted, extracted):
    """
    Print summary for a processed paper.
    """
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