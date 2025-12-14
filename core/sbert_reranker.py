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
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):  # Upgraded from MiniLM for better accuracy
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
        """Prepare query embeddings for all query sets."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from core.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        self.query_embeds = {}
        
        for name, query_text in QUERY_SETS.items():
            lines = [line.strip() for line in query_text.strip().split('\n') if line.strip()]
            
            if not lines:
                print(f"⚠️ Warning: Empty query set for {name}")
                continue
            
            try:
                # Light clean queries to match caption preprocessing
                cleaned_lines = [preprocessor.light_clean_for_sbert(line) for line in lines]
                line_embeds = self.model.encode(cleaned_lines, convert_to_tensor=True, normalize_embeddings=True)
                avg_embed = torch.mean(line_embeds, dim=0)
                # Renormalize after averaging
                avg_embed = avg_embed / torch.norm(avg_embed)
                self.query_embeds[name] = avg_embed
                print(f"✅ Encoded query set '{name}' with {len(lines)} lines")
            except Exception as e:
                print(f"❌ Failed to encode query set '{name}': {e}")
                combined_text = " ".join(lines)
                self.query_embeds[name] = self.model.encode(combined_text, convert_to_tensor=True)
        
        return self.query_embeds
    
    def rerank_figures(self, figures: List[Figure]) -> List[Figure]:
        """
        Re-rank figures using Sentence-BERT similarity.
        Adds sbert_sim and best_sbert_query fields to figures.
        """
        if not figures or not self.query_embeds:
            return figures
        
        # Initialize SBERT fields
        for f in figures:
            f.best_sbert_query = None
            f.sbert_sim = 0.0
        
        # Light clean captions for SBERT (preserve natural language)
        from core.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        captions = [preprocessor.light_clean_for_sbert(f.caption) for f in figures]
        
        try:
            caption_embeds = self.model.encode(
                captions,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Process each figure
            for i, f in enumerate(figures):
                best_sim = -1.0
                best_query = None
                
                for q_name, q_embed in self.query_embeds.items():
                    sim = util.cos_sim(caption_embeds[i], q_embed).item()
                    if sim > best_sim:
                        best_sim = float(sim)
                        best_query = q_name
                
                f.sbert_sim = best_sim
                f.best_sbert_query = best_query
            
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