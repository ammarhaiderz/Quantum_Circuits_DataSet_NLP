"""
Main pipeline integrating all components.
"""

import os
import sys
from typing import List, Dict, Tuple
import time

# Fix imports for your structure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.figure_data import Figure, ExtractedImage
    HAS_FIGURE_MODEL = True
except ImportError as e:
    print(f"⚠️ Could not import Figure/ExtractedImage: {e}")
    HAS_FIGURE_MODEL = False
    
    # Create minimal fallback classes
    class Figure:
        def __init__(self, caption="", image_path="", paper_id=""):
            self.caption = caption or ""
            self.image_path = image_path or ""
            self.paper_id = paper_id or ""
            self.similarity = 0.0
            self.sbert_sim = 0.0
            self.best_query = None
            self.best_sbert_query = None
            self.combined_score = 0.0
            self.zero_shot_is_circuit = False
            self.zero_shot_confidence = 0.0
            self.zero_shot_label = ""
            self.negative_penalty = 0.0
            self.preprocessed_text = ""

# Now import other core modules
try:
    from core.preprocessor import TextPreprocessor
    from core.tfidf_filter import FixedTfidfFilter
    from core.sbert_reranker import FixedSbertReranker
    from core.zero_shot_classifier import FastZeroShotClassifier
    from core.unified_scorer import UnifiedScorer
    from config.settings import (
        USE_ZERO_SHOT_PREFILTER,
        TOP_K_PER_PAPER,
        PRINT_TOP_CAPTIONS,
        ENABLE_DEBUG_PRINTS
    )
except ImportError as e:
    print(f"❌ Failed to import core modules: {e}")
    raise

class QuantumCircuitPipeline:
    """Main pipeline for quantum circuit detection."""
    
    def __init__(self):
        """Initialize all components."""
        print("=" * 60)
        print("QUANTUM CIRCUIT DETECTION PIPELINE")
        print("=" * 60)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.tfidf_filter = FixedTfidfFilter(self.preprocessor)
        
        # Optional zero-shot prefilter
        if USE_ZERO_SHOT_PREFILTER:
            self.zero_shot = FastZeroShotClassifier()
        else:
            self.zero_shot = None
        
        # SBERT reranker
        self.sbert = FixedSbertReranker(model_name='all-mpnet-base-v2')
        
        # Unified scorer
        self.scorer = UnifiedScorer()
        
        print("✅ Pipeline initialized")
    
    def process_figures(self, figures: List[Figure]) -> List[Figure]:
        """
        Process figures through the complete pipeline.
        
        Returns:
            List of top quantum circuit figures
        """
        start_time = time.time()
        
        if not figures:
            print("⚠️ No figures to process")
            return []
        
        print(f"\n📊 Starting with {len(figures)} figures")
        
        # Step 1: Optional zero-shot prefiltering
        if self.zero_shot:
            print("\n1️⃣ Zero-shot prefiltering...")
            figures = self.zero_shot.batch_classify(figures)
            
            # Filter to only circuit candidates
            original_count = len(figures)
            figures = [f for f in figures if getattr(f, 'zero_shot_is_circuit', True)]
            
            if ENABLE_DEBUG_PRINTS:
                print(f"   → {len(figures)}/{original_count} identified as potential circuits")
                if figures:
                    for f in figures[:3]:
                        print(f"     - '{f.caption[:80]}...' (conf: {f.zero_shot_confidence:.2f})")
        
        # Step 2: TF-IDF filtering
        print("\n2️⃣ TF-IDF filtering...")
        figures = self.tfidf_filter.filter_figures(figures)
        
        # Get accepted figures
        tfidf_accepted = self.tfidf_filter.get_accepted_figures(figures)
        
        if ENABLE_DEBUG_PRINTS:
            print(f"   → {len(tfidf_accepted)}/{len(figures)} passed TF-IDF threshold")
            if tfidf_accepted:
                # Sort by TF-IDF score for display
                sorted_by_tfidf = sorted(tfidf_accepted, key=lambda x: x.similarity, reverse=True)
                for f in sorted_by_tfidf[:PRINT_TOP_CAPTIONS]:
                    print(f"     - Score: {f.similarity:.3f} | '{f.caption[:80]}...'")
        
        if not tfidf_accepted:
            print("⚠️ No figures passed TF-IDF filtering")
            return []
        
        # Step 3: SBERT reranking
        print("\n3️⃣ SBERT semantic reranking...")
        
        # Load SBERT model
        try:
            self.sbert.load_model()
            self.sbert.prepare_query_embeddings()
            
            # Rerank accepted figures
            tfidf_accepted = self.sbert.rerank_figures(tfidf_accepted)
            
            if ENABLE_DEBUG_PRINTS:
                # Sort by SBERT score for display
                sorted_by_sbert = sorted(tfidf_accepted, key=lambda x: x.sbert_sim, reverse=True)
                print(f"   → SBERT scores range: {min(f.sbert_sim for f in tfidf_accepted):.3f} to {max(f.sbert_sim for f in tfidf_accepted):.3f}")
                
                for f in sorted_by_sbert[:PRINT_TOP_CAPTIONS]:
                    print(f"     - Score: {f.sbert_sim:.3f} | Query: {f.best_sbert_query} | '{f.caption[:80]}...'")
                    
        except Exception as e:
            print(f"⚠️ SBERT failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue without SBERT scores
            for f in tfidf_accepted:
                f.sbert_sim = 0.0
                f.best_sbert_query = None
        
        # Step 4: Unified scoring
        print("\n4️⃣ Final scoring and selection...")
        final_figures = self.scorer.score_figures(tfidf_accepted)
        
        # Take top K
        top_figures = final_figures[:TOP_K_PER_PAPER]
        
        # Print results
        print(f"\n✅ Processing complete!")
        print(f"   ⏱️  Time: {time.time() - start_time:.1f}s")
        print(f"   📈 Input: {len(figures)} figures")
        print(f"   🎯 Output: {len(top_figures)} quantum circuits")
        
        if top_figures and ENABLE_DEBUG_PRINTS:
            print(f"\n🏆 Top {min(3, len(top_figures))} circuits:")
            for i, fig in enumerate(top_figures[:3], 1):
                score = getattr(fig, 'combined_score', fig.sbert_sim)
                print(f"   {i}. Score: {score:.3f} | {fig.best_sbert_query or fig.best_query}")
                print(f"      '{fig.caption}'")
        
        return top_figures
    
    def process_paper(self, paper_id: str, figures: List[Figure]) -> Dict:
        """Process figures from a single paper."""
        print(f"\n📄 Processing paper: {paper_id}")
        
        # Process figures
        selected = self.process_figures(figures)
        
        # Prepare results
        result = {
            'paper_id': paper_id,
            'total_figures': len(figures),
            'selected_figures': len(selected),
            'figures': selected
        }
        
        return result