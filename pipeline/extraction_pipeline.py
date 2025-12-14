"""
Main pipeline for quantum circuit image extraction.
"""

import pandas as pd
from typing import List, Tuple
import tarfile
import os

from models.figure_data import Figure, ExtractedImage
from core.preprocessor import TextPreprocessor
from core.tfidf_filter import TfidfFilter
from core.sbert_reranker import SbertReranker
from core.image_extractor import ImageExtractor
from config.settings import (
    MAX_IMAGES, TOP_K_PER_PAPER, PRINT_TOP_CAPTIONS,
    OUTPUT_DIR, ENABLE_DEBUG_PRINTS, SBERT_MIN_SIM,
    USE_COMBINED_SCORE, TFIDF_WEIGHT, SBERT_WEIGHT, COMBINED_THRESHOLD
)


class ExtractionPipeline:
    """Orchestrates the entire extraction pipeline."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf_filter = TfidfFilter(self.preprocessor)
        self.sbert_reranker = SbertReranker()
        self.image_extractor = ImageExtractor(self.preprocessor)
        
        # Statistics
        self.stats = {
            'papers_checked': 0,
            'papers_with_figures': 0,
            'papers_with_candidates': 0,
            'papers_with_extracted': 0,
            'total_figures_seen': 0,
            'total_saved': 0
        }
        
        self.text_records = []
        self.all_extracted: List[ExtractedImage] = []
    
    def initialize(self) -> bool:
        """Initialize the pipeline components."""
        print("🚀 Initializing Quantum Circuit Image Extractor Pipeline")
        
        # Clear output directory
        self.image_extractor.clear_output_dir()
        
        # # Test SBERT
        # if not self.sbert_reranker.test_implementation():
        #     print("❌ SBERT test failed. Exiting.")
        #     return False
        
        # Load SBERT model
        try:
            self.sbert_reranker.load_model()
            self.sbert_reranker.prepare_query_embeddings()
        except Exception as e:
            print(f"❌ Failed to initialize SBERT: {e}")
            return False
        
        print("✅ Pipeline initialized successfully")
        return True
    
    def process_paper(self, paper_id: str) -> Tuple[List[ExtractedImage], List[Figure]]:
        """Process a single paper."""
        self.stats['papers_checked'] += 1
        
        # Cache PDF paper as we process it
        self.image_extractor.download_pdf_paper(paper_id)
        
        src = self.image_extractor.download_source(paper_id)
        if not src:
            return [], []
        
        try:
            tar = tarfile.open(fileobj=src, mode="r:gz")
        except Exception as e:
            print(f"❌ Failed to open tar file for {paper_id}: {e}")
            return [], []
        
        # Extract figures from LaTeX
        figures = []
        for m in tar.getmembers():
            if m.name.endswith(".tex"):
                try:
                    tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                    figures.extend(self.image_extractor.extract_figures_from_tex(tex))
                except Exception as e:
                    print(f"⚠️ Failed to parse {m.name}: {e}")
        
        if not figures:
            return [], []
        
        # Set paper ID for all figures
        for f in figures:
            f.paper_id = paper_id
        
        self.stats['papers_with_figures'] += 1
        self.stats['total_figures_seen'] += len(figures)
        
        # Apply TF-IDF filtering
        figures = self.tfidf_filter.filter_figures(figures)
        
        # Debug print TF-IDF results
        if ENABLE_DEBUG_PRINTS:
            self._print_tfidf_debug(figures)
        
        # Get TF-IDF accepted figures
        accepted_tfidf = self.tfidf_filter.get_accepted_figures(figures)
        
        if accepted_tfidf:
            self.stats['papers_with_candidates'] += 1
        
        # Limit pool for SBERT reranking
        accepted_tfidf = accepted_tfidf[:TOP_K_PER_PAPER * 3]
        
        # Initialize accepted list
        accepted = []
        
        # Apply SBERT reranking
        if accepted_tfidf:
            accepted_tfidf = self.sbert_reranker.rerank_figures(accepted_tfidf)
            
            # Compute combined scores
            if USE_COMBINED_SCORE:
                for f in accepted_tfidf:
                    # Normalize scores to [0, 1] range for fair weighting
                    tfidf_norm = min(f.similarity / 1.0, 1.0)  # TF-IDF rarely > 1.0
                    sbert_norm = f.sbert_sim  # Already in [0, 1]
                    f.combined_score = TFIDF_WEIGHT * tfidf_norm + SBERT_WEIGHT * sbert_norm
                
                # Sort by combined score
                accepted_tfidf = sorted(
                    accepted_tfidf,
                    key=lambda x: x.combined_score,
                    reverse=True
                )
                
                # Final selection: filter by combined threshold
                accepted = [f for f in accepted_tfidf if f.combined_score >= COMBINED_THRESHOLD]
            else:
                # Legacy cascade approach: sort by SBERT only
                accepted_tfidf = sorted(
                    accepted_tfidf,
                    key=lambda x: x.sbert_sim,
                    reverse=True
                )
                # Final selection: filter by SBERT threshold
                accepted = [f for f in accepted_tfidf if f.sbert_sim >= SBERT_MIN_SIM]
            
            accepted = accepted[:TOP_K_PER_PAPER]
        
        # Mark selected figures
        for f in accepted:
            f.selected = True
        
        # Extract images
        extracted = self.image_extractor.extract_images(tar, accepted, paper_id)
        
        if extracted:
            self.stats['papers_with_extracted'] += 1
        
        self.stats['total_saved'] += len(extracted)
        self.all_extracted.extend(extracted)
        
        # Create records for DataFrame
        self._create_records(paper_id, figures, extracted)
        
        # Print summary for this paper
        self._print_paper_summary(paper_id, figures, accepted_tfidf, accepted, extracted)
        
        return extracted, figures
    
    def _create_records(self, paper_id: str, figures: List[Figure], extracted: List[ExtractedImage]):
        """Create records for DataFrame export."""
        extracted_lookup = {e.img_name: e for e in extracted}
        
        for f in figures:
            img_name = os.path.basename(f.img_path)
            e = extracted_lookup.get(img_name)
            
            # Calculate token overlap for debugging
            tokens = set(f.preprocessed_text.split())
            overlap_count = len(tokens & self.tfidf_filter.ALLOWED_VOCAB)
            
            # Calculate token overlap for debugging
            tokens = set(f.preprocessed_text.split())
            overlap_count = len(tokens & self.tfidf_filter.ALLOWED_VOCAB)
            
            rec = {
                "paper_id": paper_id,
                "img_name": img_name,
                "image_path": e.file_path if e else None,
                "raw_caption": f.caption,
                "preprocessed_text": f.preprocessed_text,
                "similarity": f.similarity,
                "similarity_raw": f.similarity_raw,
                "negative_tokens": f.negative_tokens,
                "penalty": f.penalty,
                "token_overlap_count": overlap_count,
                "best_query": f.best_query,
                "sbert_sim": f.sbert_sim,
                "best_sbert_query": f.best_sbert_query,
                "combined_score": f.combined_score,
                "selected": f.selected,
                "extracted": f.extracted,
                **{f"sim_{k}": v for k, v in f.similarities.items()}
            }
            
            self.text_records.append(rec)

    def _print_tfidf_debug(self, figures: List[Figure]):
        """Print debug information for TF-IDF results."""
        if not figures:
            return

        print("\n   [SEARCH] Top captions by TF-IDF:")
        sorted_figures = sorted(
            figures, key=lambda x: x.similarity, reverse=True
        )

        for i, f in enumerate(sorted_figures[:PRINT_TOP_CAPTIONS], start=1):
            print(
                f"\n   [{i}] tfidf={f.similarity:.4f} "
                f"(raw={f.similarity_raw:.4f}, "
                f"neg={f.negative_tokens}, pen={f.penalty:.4f})"
            )
            print("   RAW:")
            caption_preview = (
                f.caption[:100] + ("..." if len(f.caption) > 100 else "")
            )
            print("   " + caption_preview)
            print("   PREPROCESSED:")
            print("   " + f.preprocessed_text)

    def _print_paper_summary(self, paper_id, figures, accepted_tfidf, accepted, extracted):
        """Simple paper summary print."""
        print(f"\n[INFO] Paper: {paper_id}")
        print(f"  Figures found: {len(figures)}")
        print(f"  TF-IDF candidates: {len(accepted_tfidf)}")
        print(f"  SBERT selected: {len(accepted)}")
        print(f"  Images saved: {len(extracted)}")
        
        if extracted:
            for img in extracted[:3]:  # Show first 3
                print(f"    ✓ {img.img_name}")
            if len(extracted) > 3:
                print(f"    ... and {len(extracted) - 3} more")