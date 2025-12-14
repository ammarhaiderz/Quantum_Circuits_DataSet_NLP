"""
Main pipeline for quantum circuit extraction with SBERT.
"""
import tarfile
import os

from config import (
    ID_FILE, MAX_IMAGES, SIMILARITY_THRESHOLD, 
    TOP_K_PER_PAPER, PRINT_TOP_CAPTIONS, SBERT_MIN_SIM
)
from utils import (
    clear_output_dir, read_arxiv_ids, download_source, 
    save_results, print_debug_info, print_paper_summary
)
from scorer import TFIDFScorer
from scorer import EnhancedTFIDFScorer
from ranker import SBERTRanker
from extractor import FigureExtractor


class QuantumCircuitExtractorSBERT:
    """
    Main orchestrator for the quantum circuit extraction pipeline with SBERT.
    """
    
    def __init__(self):
        self.papers_checked = 0
        self.papers_with_figures = 0
        self.papers_with_candidates = 0
        self.papers_with_extracted = 0
        self.total_figures_seen = 0
        self.total_saved = 0
        self.saved_uniques = set()
        self.text_records = []
        
        # Initialize components
        # self.tfidf_scorer = TFIDFScorer()
            # Or use EnhancedTFIDFScorer for n-gram awareness
        self.tfidf_scorer = EnhancedTFIDFScorer()
        self.sbert_ranker = SBERTRanker()
        self.figure_extractor = FigureExtractor()
    
    def initialize_sbert(self):
        """Initialize SBERT model and prepare query embeddings."""
        if not self.sbert_ranker.load_model():
            return False
        
        if not self.sbert_ranker.test_implementation():
            return False
        
        self.sbert_ranker.prepare_query_embeddings()
        return True
    
    def process_paper(self, pid):
        """
        Process a single arXiv paper.
        """
        src = download_source(pid)
        if not src:
            return False

        try:
            tar = tarfile.open(fileobj=src, mode="r:gz")
        except Exception:
            return False

        # Extract figures from LaTeX
        figures = []
        for m in tar.getmembers():
            if m.name.endswith(".tex"):
                try:
                    tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                    figures.extend(self.figure_extractor.extract_figures_from_tex(tex))
                except Exception:
                    pass

        if not figures:
            return False

        self.papers_with_figures += 1
        self.total_figures_seen += len(figures)

        # Step 1: TF-IDF scoring
        figures = self.tfidf_scorer.score_figures(figures)
        print_debug_info(figures, PRINT_TOP_CAPTIONS)

        # Step 2: TF-IDF hard gate
        accepted_tfidf = [
            f for f in figures
            if f["similarity"] >= SIMILARITY_THRESHOLD
        ]

        if accepted_tfidf:
            self.papers_with_candidates += 1

        # Limit pool size for semantic rerank
        accepted_tfidf = accepted_tfidf[:TOP_K_PER_PAPER * 3]

        # Step 3: SBERT semantic re-ranking
        if accepted_tfidf:
            accepted_tfidf = self.sbert_ranker.rerank_figures(accepted_tfidf)
            accepted_tfidf = sorted(
                accepted_tfidf,
                key=lambda x: x.get("sbert_sim", 0.0),
                reverse=True
            )

        # Step 4: Final selection with SBERT threshold
        accepted = accepted_tfidf[:TOP_K_PER_PAPER]
        accepted = [
            f for f in accepted_tfidf
            if f.get("sbert_sim", 0.0) >= SBERT_MIN_SIM
        ]

        # Step 5: Image extraction
        extracted = self.figure_extractor.extract_images(
            tar, accepted, pid, self.saved_uniques
        )

        if extracted:
            self.papers_with_extracted += 1

        self.total_saved += len(extracted)

        # Step 6: Log results
        self._log_results(figures, extracted, pid, accepted)

        # Step 7: Print paper summary
        print_paper_summary(pid, figures, accepted_tfidf, accepted, extracted)
        print(f"📊 Total saved: {self.total_saved}/{MAX_IMAGES}")

        return True
    
    def _log_results(self, figures, extracted, pid, accepted):
        """Log results for this paper."""
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

            # Store per-query similarities
            for k, v in f["similarities"].items():
                rec[f"sim_{k}"] = v

            self.text_records.append(rec)
    
    def run(self):
        """
        Run the complete extraction pipeline.
        """
        print("🧹 Clearing previously saved images...")
        clear_output_dir()
        
        # Initialize SBERT
        print("\n🔧 Initializing SBERT model...")
        if not self.initialize_sbert():
            print("❌ SBERT initialization failed. Exiting.")
            exit(1)
        
        arxiv_ids = read_arxiv_ids(ID_FILE)

        for pid in arxiv_ids:
            if self.total_saved >= MAX_IMAGES:
                break

            self.papers_checked += 1
            self.process_paper(pid)
        
        # Save results
        df_path = save_results(self.text_records)
        
        # Print summary
        self._print_summary(df_path)
    
    def _print_summary(self, df_path):
        """Print summary statistics."""
        print("\n================ SUMMARY ================")
        print(f"Papers checked: {self.papers_checked}")
        print(f"Papers with figures: {self.papers_with_figures}")
        print(f"Papers with candidates: {self.papers_with_candidates}")
        print(f"Papers with extracted images: {self.papers_with_extracted}")
        print(f"Total figures seen: {self.total_figures_seen}")
        print(f"Total images saved: {self.total_saved}")
        print(f"SBERT threshold used: {SBERT_MIN_SIM}")
        print(f"Caption log saved to: {df_path}")
        
        # Debug statistics
        if self.text_records and "sbert_sim" in self.text_records[0]:
            self._print_sbert_statistics()

    def _print_sbert_statistics(self):
        """Print aggregate SBERT score statistics for debugging."""
        scores = [
            r.get("sbert_sim") for r in self.text_records
            if r.get("sbert_sim") is not None
        ]
        if not scores:
            print("\n(No SBERT scores recorded.)")
            return

        n = len(scores)
        s_min = min(scores)
        s_max = max(scores)
        s_mean = sum(scores) / n if n else 0.0
        above = sum(1 for s in scores if s >= SBERT_MIN_SIM)
        selected = sum(1 for r in self.text_records if r.get("selected"))
        extracted = sum(1 for r in self.text_records if r.get("extracted"))

        # Best SBERT query distribution (top 5)
        q_counts = {}
        for r in self.text_records:
            q = r.get("best_sbert_query")
            if q:
                q_counts[q] = q_counts.get(q, 0) + 1
        top_queries = sorted(q_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\n---------------- SBERT Stats ----------------")
        print(f"Records with SBERT scores: {n}")
        print(f"Score min/mean/max: {s_min:.4f} / {s_mean:.4f} / {s_max:.4f}")
        print(f">= threshold ({SBERT_MIN_SIM}): {above}")
        print(f"Selected: {selected} | Extracted: {extracted}")
        if top_queries:
            print("Top best_sbert_query values:")
            for q, c in top_queries:
                print(f"  {q}: {c}")


if __name__ == "__main__":
    extractor = QuantumCircuitExtractorSBERT()
    extractor.run()