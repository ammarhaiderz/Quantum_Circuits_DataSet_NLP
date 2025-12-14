"""
SBERT-based semantic reranking module.
"""
import torch
from sentence_transformers import SentenceTransformer, util

from config import (
    SBERT_MODEL_NAME, SBERT_BATCH_SIZE,
    SBERT_NORMALIZE_EMBEDDINGS, SBERT_MIN_SIM
)


class SBERTRanker:
    """Handles SBERT-based semantic reranking."""
    
    def __init__(self):
        self.model = None
        self.query_embeds = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load Sentence-BERT model."""
        print("📦 Loading Sentence-BERT model...")
        try:
            self.model = SentenceTransformer(SBERT_MODEL_NAME)
            # Test the model
            test_embed = self.model.encode("test", convert_to_tensor=True)
            print(
                "✅ SBERT model loaded successfully "
                f"(embedding dim: {test_embed.shape[0]})"
            )
            return True
        except Exception as e:
            print(f"❌ Failed to load SBERT model: {e}")
            print("Try: pip install sentence-transformers")
            return False
    
    def prepare_query_embeddings(self):
        """Prepare embeddings for all query strings."""
        import config as cfg

        query_texts = []

        # Prefer enhanced structure if present, else fall back to QUERY_SETS
        if hasattr(cfg, "ENHANCED_QUERY_SETS") and cfg.ENHANCED_QUERY_SETS:
            enhanced = cfg.ENHANCED_QUERY_SETS
            for _category, data in enhanced.items():
                # Expect data like { "positive": [["quantum","circuit"], ...] }
                positives = (
                    data.get("positive", [])
                    if isinstance(data, dict)
                    else []
                )
                for ngram_list in positives:
                    if isinstance(ngram_list, (list, tuple)):
                        query_texts.append(" ".join(ngram_list))
        else:
            # Old format: BLOCK strings under categories
            for _category, query_block in cfg.QUERY_SETS.items():
                lines = [
                    line.strip()
                    for line in query_block.strip().split('\n')
                    if line.strip()
                ]
                query_texts.extend(lines)
        
        # Remove duplicates
        query_texts = list(set(query_texts))

        print(
            "📝 Preparing SBERT embeddings for "
            f"{len(query_texts)} unique queries..."
        )
        
        # Create embeddings
        self.query_embeddings = self.model.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        self.query_texts = query_texts
        # Build name->embedding lookup for efficient similarity
        self.query_embeds = {
            qt: self.query_embeddings[i]
            for i, qt in enumerate(self.query_texts)
        }
        print(f"✅ SBERT query embeddings prepared: {len(query_texts)} queries")
    
    def rerank_figures(self, figures):
        """
        Re-rank candidates using Sentence-BERT similarity.
        Returns figures with SBERT similarity scores.
        """
        if not figures or not self.query_embeds:
            return figures
        
        # Initialize SBERT fields for all figures
        for f in figures:
            f["best_sbert_query"] = None
            f["sbert_sim"] = 0.0
        
        # Import preprocessor for light cleaning
        from preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

        # Extract and lightly clean captions for SBERT
        # Use light_clean_for_sbert to preserve natural language
        captions = [
            preprocessor.light_clean_for_sbert(f["caption"])
            for f in figures
        ]

        try:
            # Encode all captions in batch
            caption_embeds = self.model.encode(
                captions,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=SBERT_NORMALIZE_EMBEDDINGS,
                batch_size=SBERT_BATCH_SIZE
            ).to(self.device)

            # Process each figure
            for i, f in enumerate(figures):
                best = -1.0  # Start with negative value
                best_query = None

                # Compare with each query embedding
                for q_name, q_embed in self.query_embeds.items():
                    sim = util.cos_sim(caption_embeds[i], q_embed).item()
                    if sim > best:
                        best = float(sim)
                        best_query = q_name

                f["sbert_sim"] = best
                f["best_sbert_query"] = best_query

            # Clean up GPU memory
            self._cleanup_memory(caption_embeds)

        except Exception as e:
            print(f"⚠️ SBERT reranking failed: {e}")
            print("⚠️ Continuing with TF-IDF scores only")
        
        return figures
    
    def _cleanup_memory(self, tensor):
        """Clean up GPU memory."""
        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_implementation(self):
        """Test SBERT integration."""
        print("\n🧪 Testing SBERT implementation...")
        
        try:
            # Test queries
            test_queries = ["quantum circuit diagram", "gate sequence"]
            test_captions = [
                "Circuit diagram showing CNOT gates",
                "Figure 3: Energy levels of the system",
                "The quantum circuit with Hadamard and CNOT gates"
            ]
            
            # Encode
            query_embeds = self.model.encode(
                test_queries,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            caption_embeds = self.model.encode(
                test_captions,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # Compute similarities
            similarities = util.cos_sim(caption_embeds, query_embeds)
            
            print("\nTest Results:")
            for i, caption in enumerate(test_captions):
                print(f"\nCaption: {caption}")
                for j, query in enumerate(test_queries):
                    sim = similarities[i][j].item()
                    print(f"  Similarity to '{query}': {sim:.4f}")
            
            # Check score ranges
            s_min = similarities.min().item()
            s_max = similarities.max().item()
            print(f"\n✅ Score range: {s_min:.4f} to {s_max:.4f}")
            print(
                "📝 Typical circuit captions score 0.3-0.8 "
                "with relevant queries"
            )
            print(f"🔧 SBERT_MIN_SIM = {SBERT_MIN_SIM} should work well")
            
            # Clean up
            self._cleanup_memory(query_embeds)
            self._cleanup_memory(caption_embeds)
            
            return True
            
        except Exception as e:
            print(f"❌ SBERT test failed: {e}")
            return False
