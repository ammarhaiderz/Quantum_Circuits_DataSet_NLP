"""
Main pipeline for quantum circuit image extraction.
"""

import pandas as pd
from typing import List, Tuple
import tarfile
import os
import shutil
from pathlib import Path

from models.figure_data import Figure, ExtractedImage
from core.preprocessor import TextPreprocessor
from core.tfidf_filter import TfidfFilter
from core.sbert_reranker import SbertReranker
from core.image_extractor import ImageExtractor
from config.settings import (
    MAX_IMAGES, TOP_K_PER_PAPER, PRINT_TOP_CAPTIONS,
    OUTPUT_DIR, ENABLE_DEBUG_PRINTS, SBERT_MIN_SIM,
    USE_COMBINED_SCORE, TFIDF_WEIGHT, SBERT_WEIGHT, COMBINED_THRESHOLD,
    SIMILARITY_THRESHOLD
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

        # Start fresh: remove contents of `circuit_images/` to avoid stale outputs
        ci = Path('circuit_images')
        if ci.exists():
            for p in ci.iterdir():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except Exception as e:
                    print(f"⚠️ Failed to remove {p}: {e}")
        else:
            ci.mkdir(parents=True, exist_ok=True)

        # Ensure common subdirectories exist after cleanup
        (ci / 'live_blocks').mkdir(parents=True, exist_ok=True)
        (ci / 'rendered_pdflatex').mkdir(parents=True, exist_ok=True)
        (ci / 'blocks').mkdir(parents=True, exist_ok=True)
        (ci / 'rendered').mkdir(parents=True, exist_ok=True)

        # Clear output directory (also clears OUTPUT_DIR images)
        self.image_extractor.clear_output_dir()
        # Remove previous circuits JSONL so we start fresh
        try:
            data_file = Path('data') / 'circuits.jsonl'
            if data_file.exists():
                data_file.unlink()
        except Exception:
            pass
        # Ensure data files exist and start fresh
        try:
            data_dir = Path('data')
            data_dir.mkdir(parents=True, exist_ok=True)
            jsonl = data_dir / 'circuits.jsonl'
            jsonf = data_dir / 'circuits.json'
            # create empty JSONL file (overwrite if present)
            with open(jsonl, 'w', encoding='utf-8') as f:
                pass
            # create empty JSON array for circuits.json
            with open(jsonf, 'w', encoding='utf-8') as f:
                f.write('[]')
        except Exception:
            pass
        
        # Reset the processed blocks set to start fresh
        try:
            from core.live_latex_extractor import reset_processed_blocks
            reset_processed_blocks()
        except Exception:
            pass
        
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
        print(f"\n{'='*80}")
        print(f"📄 PAPER: {paper_id}")
        print(f"{'='*80}")
        
        # Cache PDF paper as we process it
        self.image_extractor.download_pdf_paper(paper_id)
        
        src = self.image_extractor.download_source(paper_id)
        if not src:
            print(f"❌ Failed to download source for {paper_id}")
            return [], []
        
        try:
            tar = tarfile.open(fileobj=src, mode="r:gz")
        except Exception as e:
            print(f"❌ Failed to open tar file for {paper_id}: {e}")
            return [], []
        
        # Extract figures from LaTeX
        figures = []
        figure_counter = 1
        try:
            for m in tar.getmembers():
                if m.name.endswith(".tex"):
                    try:
                        tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                        res = self.image_extractor.extract_figures_from_tex(tex, paper_id=paper_id)
                        # extract_figures_from_tex now may return (figures, figure_counter)
                        if isinstance(res, tuple):
                            new_figs, figure_counter = res
                            figures.extend(new_figs)
                        else:
                            figures.extend(res)
                    except Exception as e:
                        print(f"⚠️ Failed to parse {m.name}: {e}")
        except KeyboardInterrupt:
            # Ensure tarfile is closed on user interrupt and re-raise to allow
            # program to terminate normally.
            try:
                tar.close()
            except Exception:
                pass
            raise
        except Exception as e:
            try:
                tar.close()
            except Exception:
                pass
            print(f"⚠️ Error iterating archive for {paper_id}: {e}")
            return [], []
        
        if not figures:
            print(f"⚠️ No figures found in LaTeX")
            return [], []

        # Try to populate page numbers for records from this paper (best-effort)
        try:
            from core.circuit_store import update_pages_in_jsonl
            try:
                updated = update_pages_in_jsonl(paper_id)
                if updated:
                    print(f"   ✓ Updated {updated} page numbers for {paper_id}")
            except Exception:
                pass
        except Exception:
            # circuit_store or PyMuPDF not available; skip silently
            pass
        
        # Set paper ID for all figures
        for f in figures:
            f.paper_id = paper_id
        
        self.stats['papers_with_figures'] += 1
        self.stats['total_figures_seen'] += len(figures)
        print(f"\n🔍 GATE 0: FIGURE EXTRACTION")
        print(f"   ✓ Figures extracted: {len(figures)}")
        
        # Apply TF-IDF filtering
        figures = self.tfidf_filter.filter_figures(figures)
        
        # Debug print TF-IDF results
        if ENABLE_DEBUG_PRINTS:
            self._print_tfidf_debug(figures)
        
        # Get TF-IDF accepted figures
        accepted_tfidf = self.tfidf_filter.get_accepted_figures(figures)
        
        print(f"\n🔍 GATE 1: TF-IDF FILTER (threshold={SIMILARITY_THRESHOLD})")
        print(f"   → Input: {len(figures)} figures")
        print(f"   ✓ Passed: {len(accepted_tfidf)} figures")
        print(f"   ✗ Rejected: {len(figures) - len(accepted_tfidf)} figures")
        
        if accepted_tfidf:
            self.stats['papers_with_candidates'] += 1
            # Statistics for tuning
            tfidf_scores = [f.similarity for f in accepted_tfidf]
            raw_scores = [f.similarity_raw for f in accepted_tfidf]
            neg_counts = [f.negative_tokens for f in accepted_tfidf]
            print(f"   📊 TF-IDF scores - Min: {min(tfidf_scores):.4f}, Max: {max(tfidf_scores):.4f}, Avg: {sum(tfidf_scores)/len(tfidf_scores):.4f}")
            print(f"   📊 Raw scores - Min: {min(raw_scores):.4f}, Max: {max(raw_scores):.4f}")
            print(f"   📊 Negative tokens - Min: {min(neg_counts)}, Max: {max(neg_counts)}, Avg: {sum(neg_counts)/len(neg_counts):.1f}")
        
        # Limit pool for SBERT reranking
        accepted_tfidf = accepted_tfidf[:TOP_K_PER_PAPER * 3]
        print(f"   → Limited to top {len(accepted_tfidf)} for SBERT reranking")
        
        # Initialize accepted list
        accepted = []
        
        # Apply SBERT reranking
        if accepted_tfidf:
            accepted_tfidf = self.sbert_reranker.rerank_figures(accepted_tfidf)
            
            print(f"\n🔍 GATE 2: SBERT RERANKING")
            print(f"   → Input: {len(accepted_tfidf)} figures")

            sbert_scores = [f.sbert_sim for f in accepted_tfidf]
            print(f"   📊 SBERT scores - Min: {min(sbert_scores):.4f}, Max: {max(sbert_scores):.4f}, Avg: {sum(sbert_scores)/len(sbert_scores):.4f}")
            
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
                
                combined_scores = [f.combined_score for f in accepted_tfidf]
                print(f"\n🔍 GATE 3: COMBINED SCORE FILTER (threshold={COMBINED_THRESHOLD})")
                print(f"   Weights: TF-IDF={TFIDF_WEIGHT}, SBERT={SBERT_WEIGHT}")
                print(f"   📊 Combined scores - Min: {min(combined_scores):.4f}, Max: {max(combined_scores):.4f}, Avg: {sum(combined_scores)/len(combined_scores):.4f}")
            else:
                # Legacy cascade approach: sort by SBERT only
                accepted_tfidf = sorted(
                    accepted_tfidf,
                    key=lambda x: x.sbert_sim,
                    reverse=True
                )
                # Final selection: filter by SBERT threshold
                accepted = [f for f in accepted_tfidf if f.sbert_sim >= SBERT_MIN_SIM]
                
                print(f"\n🔍 GATE 3: SBERT FILTER (threshold={SBERT_MIN_SIM})")
            
            # Common pass/reject logging for both modes
            print(f"   ✓ Passed: {len(accepted)} figures")
            print(f"   ✗ Rejected: {len(accepted_tfidf) - len(accepted)} figures")

            accepted = accepted[:TOP_K_PER_PAPER]
            print(f"   → Limited to top {TOP_K_PER_PAPER} for final output")
        
        # Mark selected figures
        for f in accepted:
            f.selected = True
        
        # Extract images
        extracted = self.image_extractor.extract_images(tar, accepted, paper_id)
        
        print(f"\n🔍 GATE 4: IMAGE EXTRACTION")
        print(f"   → Input: {len(accepted)} figures")
        print(f"   ✓ Successfully extracted: {len(extracted)} images")
        print(f"   ✗ Failed: {len(accepted) - len(extracted)} images")
        
        if extracted:
            self.stats['papers_with_extracted'] += 1
        
        self.stats['total_saved'] += len(extracted)
        self.all_extracted.extend(extracted)
        
        print(f"\n📈 CUMULATIVE STATS")
        print(f"   Total papers processed: {self.stats['papers_checked']}")
        print(f"   Total images saved: {self.stats['total_saved']}/{MAX_IMAGES}")
        
        # Create records for DataFrame (only TF-IDF accepted figures, not all)
        self._create_records(paper_id, accepted_tfidf, extracted)
        
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