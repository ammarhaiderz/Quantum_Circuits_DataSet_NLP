"""
FIXED Sentence-BERT reranking for quantum circuit detection.
"""

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from typing import List, Dict
import numpy as np

# Fix imports for your structure
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.figure_data import Figure
    HAS_FIGURE_MODEL = True
except ImportError:
    print("⚠️ Using fallback Figure class in sbert_reranker")
    HAS_FIGURE_MODEL = False
    class Figure:
        def __init__(self):
            self.caption = ""
            self.sbert_sim = 0.0
            self.best_sbert_query = None
            self.cross_scores = {}

from config.settings import SBERT_MIN_SIM

class FixedSbertReranker:
    """Fixed SBERT implementation with optimized queries."""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize with a proper sentence transformer model.
        
        Recommended models:
        - 'all-mpnet-base-v2': Best general purpose
        - 'all-MiniLM-L6-v2': Fastest
        - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A
        """
        self.model = None
        self.cross_encoder = None
        self.model_name = model_name
        self.query_embeds = {}
    
    def create_optimized_queries(self) -> Dict[str, str]:
        """Create optimized query prompts for each category."""
        return {
            "circuit_fundamentals": """
                A schematic diagram of a quantum circuit showing gates on qubit wires.
                Includes gate symbols like CNOT, Hadamard, Pauli, Toffoli, SWAP.
                Shows horizontal lines representing qubits with gates arranged in sequence.
                May include measurement operations and classical control lines.
            """,
            
            "quantum_algorithms": """
                A quantum circuit implementing a specific quantum algorithm.
                Shows algorithm-specific patterns like QFT, Grover oracle, phase estimation.
                Circuit structure matches known algorithms (Deutsch-Jozsa, Bernstein-Vazirani, Shor, Grover).
                Includes specialized components like oracles or Fourier transform blocks.
            """,
            
            "variational_circuits": """
                A parameterized quantum circuit (ansatz) for variational algorithms.
                Shows parameterized rotation gates (RX, RY, RZ) with angle symbols (θ, φ).
                Circuit has repetitive layers with entangling gates between rotations.
                Used for VQE, QAOA, or quantum machine learning applications.
            """,
            
            "communication_and_entanglement": """
                A quantum circuit for quantum communication protocols or entanglement generation.
                Shows Bell state preparation, quantum teleportation, or superdense coding.
                Includes EPR pairs, Bell measurements, and classical communication channels.
            """
        }
    
    def load_model(self) -> SentenceTransformer:
        """Load Sentence-BERT model."""
        print("📦 Loading Sentence-BERT model...")
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Optional: Load cross-encoder for better precision
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Cross-encoder loaded for two-stage ranking")
            except:
                print("⚠️ Cross-encoder not available, using bi-encoder only")
            
            test_embed = self.model.encode("test", convert_to_tensor=True)
            print(f"✅ SBERT model loaded: {self.model_name} (dim: {test_embed.shape[0]})")
            return self.model
        except Exception as e:
            print(f"❌ Failed to load SBERT model: {e}")
            raise
    
    def prepare_query_embeddings(self):
        """Prepare embeddings for optimized queries."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        optimized_queries = self.create_optimized_queries()
        
        # Encode each query separately (no concatenation)
        for name, query in optimized_queries.items():
            try:
                # Clean and encode
                clean_query = query.strip()
                embed = self.model.encode(
                    clean_query,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                self.query_embeds[name] = embed
            except Exception as e:
                print(f"⚠️ Failed to encode query '{name}': {e}")
        
        print(f"✅ Prepared embeddings for {len(self.query_embeds)} query categories")
        return self.query_embeds
    
    def rerank_figures(self, figures: List[Figure]) -> List[Figure]:
        """
        Re-rank figures using Sentence-BERT similarity.
        Uses two-stage approach if cross-encoder is available.
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not figures:
            return figures
        
        # Initialize SBERT fields
        for f in figures:
            f.best_sbert_query = None
            f.sbert_sim = 0.0
        
        # Use clean captions for BERT
        captions = [(f.caption or "") for f in figures]
        
        # Filter empty captions
        valid_indices = [i for i, cap in enumerate(captions) if cap.strip()]
        if not valid_indices:
            print("⚠️ No valid captions for SBERT")
            return figures
        
        valid_captions = [captions[i] for i in valid_indices]
        
        print(f"🔍 SBERT processing {len(valid_captions)} captions")
        
        try:
            # Encode captions
            caption_embeds = self.model.encode(
                valid_captions,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Get query embeddings (lazy load if needed)
            if not self.query_embeds:
                self.prepare_query_embeddings()
            
            # Prepare all query embeddings
            query_names = []
            query_embeds_list = []
            
            for name, embed in self.query_embeds.items():
                query_names.append(name)
                query_embeds_list.append(embed)
            
            all_query_embeds = torch.stack(query_embeds_list, dim=0)
            
            # Calculate similarities
            similarities = util.cos_sim(caption_embeds, all_query_embeds)
            
            # Process each valid figure
            for j, idx in enumerate(valid_indices):
                f = figures[idx]
                caption_sims = similarities[j]
                best_score = torch.max(caption_sims).item()
                best_idx = torch.argmax(caption_sims).item()
                
                f.sbert_sim = best_score
                f.best_sbert_query = query_names[best_idx]
            
            # Optional: Two-stage ranking with cross-encoder
            if self.cross_encoder and len(figures) > 0:
                figures = self._apply_cross_encoder(figures)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ SBERT reranking failed: {e}")
            import traceback
            traceback.print_exc()
        
        return figures
    
    def _apply_cross_encoder(self, figures: List[Figure], top_k: int = 20) -> List[Figure]:
        """Apply cross-encoder to top candidates for better precision."""
        # Get top candidates by bi-encoder score
        candidates = sorted(figures, key=lambda x: x.sbert_sim, reverse=True)
        candidates = candidates[:min(top_k * 3, len(candidates))]
        
        # Prepare pairs for cross-encoder
        pairs = []
        figure_indices = []
        optimized_queries = self.create_optimized_queries()
        
        for i, fig in enumerate(candidates):
            caption = fig.caption or ""
            for query_name, query_text in optimized_queries.items():
                pairs.append([query_text, caption])
                figure_indices.append((i, query_name))
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Update scores
        for fig in candidates:
            fig.cross_scores = {}
        
        for (fig_idx, query_name), score in zip(figure_indices, cross_scores):
            fig = candidates[fig_idx]
            fig.cross_scores[query_name] = float(score)
        
        # Update best scores
        for fig in candidates:
            if fig.cross_scores:
                best_query = max(fig.cross_scores, key=fig.cross_scores.get)
                best_score = fig.cross_scores[best_query]
                
                # Blend with original score
                fig.sbert_sim = 0.7 * best_score + 0.3 * fig.sbert_sim
                fig.best_sbert_query = best_query
        
        return candidates
    
    def get_best_figures(self, figures: List[Figure], top_k: int) -> List[Figure]:
        """Get top figures based on SBERT similarity."""
        if not figures:
            return []
        
        # Filter by minimum similarity if in cascade mode
        filtered = [f for f in figures if f.sbert_sim >= SBERT_MIN_SIM]
        filtered.sort(key=lambda x: x.sbert_sim, reverse=True)
        
        return filtered[:top_k]