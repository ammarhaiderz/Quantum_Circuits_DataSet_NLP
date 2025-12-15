"""
Sentence-BERT reranking for improved semantic matching.
"""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional
import numpy as np

from models.figure_data import Figure
from config.queries import QUERY_SETS
from config.settings import SBERT_MIN_SIM


class SbertReranker:
    """Handles Sentence-BERT based reranking."""
    
    def __init__(self, model_name: str = 'allenai-specter'):  # Scientific paper embeddings model
        self.model = None
        self.model_name = model_name
        self.query_embeds: Dict[str, torch.Tensor] = {}
    
    def load_model(self) -> SentenceTransformer:
        """Load Sentence-BERT model."""
        print("📦 Loading Sentence-BERT model...")
        try:
            self.model = SentenceTransformer(self.model_name)
            test_embed = self.model.encode("test", convert_to_tensor=True)
            print(f"✅ SBERT model loaded (embedding dim: {test_embed.shape[0]})")
            return self.model
        except Exception as e:
            print(f"❌ Failed to load SBERT model: {e}")
            raise
    
    def prepare_query_embeddings(self) -> Dict[str, torch.Tensor]:
        """Prepare query embeddings for all query sets (individual lines, not averaged)."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.query_embeds = {}  # Now stores list of embeddings per category
        
        for name, query_text in QUERY_SETS.items():
            lines = [line.strip() for line in query_text.strip().split('\n') if line.strip()]
            
            if not lines:
                print(f"⚠️ Warning: Empty query set for {name}")
                continue
            
            try:
                # Encode each line separately (no averaging)
                line_embeds = self.model.encode(lines, convert_to_tensor=True, normalize_embeddings=True)
                self.query_embeds[name] = line_embeds  # Store all line embeddings
                print(f"✅ Encoded query set '{name}' with {len(lines)} individual lines")
            except Exception as e:
                print(f"❌ Failed to encode query set '{name}': {e}")
                combined_text = " ".join(lines)
                single_embed = self.model.encode(combined_text, convert_to_tensor=True, normalize_embeddings=True)
                self.query_embeds[name] = single_embed.unsqueeze(0)  # Make it a list of 1
        
        return self.query_embeds
    
    def rerank_figures(self, figures: List[Figure]) -> List[Figure]:
        """
        Re-rank figures using Sentence-BERT similarity.
        Adds sbert_sim and best_sbert_query fields to figures.
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not figures or not self.query_embeds:
            return figures
        
        # Initialize SBERT fields
        for f in figures:
            f.best_sbert_query = None
            f.sbert_sim = 0.0
        
        if not figures or not self.query_embeds:
            return figures
        
        # Use RAW captions (not preprocessed/stemmed) for BERT - sanitize inputs
        captions = [(f.caption or "") for f in figures]
        
        # Filter out empty captions
        valid_indices = [i for i, cap in enumerate(captions) if cap.strip()]
        if not valid_indices:
            print("⚠️ No valid captions for SBERT")
            return figures
        
        valid_captions = [captions[i] for i in valid_indices]
        
        print(f"🔍 SBERT processing {len(valid_captions)}/{len(captions)} non-empty captions")
        print(f"   Sample caption: '{valid_captions[0][:80]}...'")
        
        try:
            caption_embeds = self.model.encode(
                valid_captions,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            print(f"   Embeddings shape: {caption_embeds.shape}")
            
            # Flatten all query embeddings into one matrix for batch similarity (performance optimization)
            all_query_embeds = []
            query_category_map = []  # Track which category each query line belongs to
            for q_name, q_embeds in self.query_embeds.items():
                all_query_embeds.append(q_embeds)
            
            # Flatten all query embeddings into one matrix for batch similarity (performance optimization)
                query_category_map.extend([q_name] * len(q_embeds))
            
            all_query_embeds = torch.cat(all_query_embeds, dim=0)
            
            # Compute all similarities in one batch operation
            all_sims = util.cos_sim(caption_embeds, all_query_embeds)  # Shape: (n_captions, n_all_queries)
            
            # Process each valid figure - find best query match
            for j, idx in enumerate(valid_indices):
                f = figures[idx]
                caption_sims = all_sims[j]  # All similarities for this caption
                best_query_idx = caption_sims.argmax().item()
                best_sim = caption_sims[best_query_idx].item()
                best_query = query_category_map[best_query_idx]
                
                f.sbert_sim = float(best_sim)
                f.best_sbert_query = best_query
            
            # Log sample scores for debugging
            scored_figures = [f for f in figures if f.sbert_sim > 0]
            if scored_figures:
                max_score = max(f.sbert_sim for f in scored_figures)
                print(f"   SBERT scores: max={max_score:.3f}, {len(scored_figures)} non-zero scores")
            else:
                print(f"   ⚠️ WARNING: All SBERT scores are 0.0!")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ SBERT reranking failed: {e}")
            print("⚠️ Continuing with TF-IDF scores only")
        
        return figures
    
    def get_best_figures(self, figures: List[Figure], top_k: int) -> List[Figure]:
        """Get top figures based on SBERT similarity."""
        if not figures:
            return []
        
        # Sort by SBERT similarity
        sorted_figures = sorted(figures, key=lambda x: x.sbert_sim, reverse=True)
        
        # Filter by minimum similarity
        filtered = [f for f in sorted_figures if f.sbert_sim >= SBERT_MIN_SIM]
        
        return filtered[:top_k]
    
    def test_implementation(self) -> bool:
        """Test SBERT implementation."""
        print("\n🧪 Testing SBERT implementation...")
        
        try:
            model = SentenceTransformer(self.model_name)
            
            test_queries = ["quantum circuit diagram", "gate sequence"]
            test_captions = [
                "Circuit diagram showing CNOT gates",
                "Figure 3: Energy levels of the system",
                "The quantum circuit implementation with Hadamard and CNOT gates"
            ]
            
            query_embeds = model.encode(test_queries, convert_to_tensor=True, normalize_embeddings=True)
            caption_embeds = model.encode(test_captions, convert_to_tensor=True, normalize_embeddings=True)
            similarities = util.cos_sim(caption_embeds, query_embeds)
            
            print("\nTest Results:")
            for i, caption in enumerate(test_captions):
                print(f"\nCaption: {caption}")
                for j, query in enumerate(test_queries):
                    sim = similarities[i][j].item()
                    print(f"  Similarity to '{query}': {sim:.4f}")
            
            print(f"\n✅ Score range: {similarities.min().item():.4f} to {similarities.max().item():.4f}")
            print(f"🔧 SBERT_MIN_SIM = {SBERT_MIN_SIM} should work well")
            
            del query_embeds, caption_embeds, similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
            
        except Exception as e:
            print(f"❌ SBERT test failed: {e}")
            return False