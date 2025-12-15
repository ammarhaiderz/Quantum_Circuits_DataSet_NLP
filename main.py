"""
Main script - simplified version for testing
"""

import os
import sys
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    ID_FILE,
    MAX_IMAGES,
    setup_directories,
    ENABLE_DEBUG_PRINTS
)

try:
    from utils.file_utils import FileUtils
    from utils.logging_utils import Logger
    HAS_UTILS = True
except ImportError:
    print("⚠️ Utility modules not found, using simplified version")
    HAS_UTILS = False
    
    class Logger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    
    class FileUtils:
        @staticmethod
        def read_arxiv_ids(filepath):
            try:
                with open(filepath, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            except:
                return ["2105.02745", "2003.08953", "1903.08126"]  # Sample IDs

from models.figure_data import Figure
from core.main_pipeline import QuantumCircuitPipeline

def load_sample_figures(arxiv_id: str) -> List[Figure]:
    """Load or create sample figures for testing."""
    # This is a placeholder - replace with your actual figure extraction
    sample_captions = [
        f"Quantum circuit diagram for {arxiv_id} showing CNOT gates",
        f"Figure 2: Energy levels in {arxiv_id}",
        f"Circuit schematic of variational quantum eigensolver",
        f"Plot of simulation results for {arxiv_id}",
        f"Grover algorithm implementation with oracle circuit",
    ]
    
    figures = []
    for i, caption in enumerate(sample_captions, 1):
        fig = Figure(
            caption=caption,
            image_path=f"extracted/{arxiv_id}_fig{i}.png",
            paper_id=arxiv_id
        )
        figures.append(fig)
    
    return figures

def main():
    """Main function."""
    print("=" * 60)
    print("🚀 QUANTUM CIRCUIT EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Setup
    setup_directories()
    logger = Logger() if HAS_UTILS else Logger()
    
    # Initialize pipeline
    pipeline = QuantumCircuitPipeline()
    
    # Read arXiv IDs
    if os.path.exists(ID_FILE):
        arxiv_ids = FileUtils.read_arxiv_ids(ID_FILE)
    else:
        # Create a sample ID file
        with open(ID_FILE, 'w') as f:
            f.write("2105.02745\n2003.08953\n1903.08126\n")
        arxiv_ids = FileUtils.read_arxiv_ids(ID_FILE)
    
    logger.info(f"Loaded {len(arxiv_ids)} arXiv IDs")
    
    # Process each paper
    results = []
    total_saved = 0
    
    for i, arxiv_id in enumerate(arxiv_ids, 1):
        print(f"\n📚 [{i}/{len(arxiv_ids)}] Processing {arxiv_id}")
        
        # Stop if we have enough images
        if total_saved >= MAX_IMAGES:
            logger.info(f"🎯 Reached maximum images limit ({MAX_IMAGES})")
            break
        
        # Load figures (replace with your actual extraction)
        figures = load_sample_figures(arxiv_id)
        
        if not figures:
            logger.warning(f"No figures found for {arxiv_id}")
            continue
        
        # Process through quantum circuit pipeline
        selected = pipeline.process_figures(figures)
        
        # Save results
        for fig in selected:
            if total_saved >= MAX_IMAGES:
                break
            
            # Create record
            record = {
                'arxiv_id': arxiv_id,
                'caption': fig.caption,
                'image_path': fig.image_path,
                'similarity': fig.similarity,
                'sbert_sim': fig.sbert_sim,
                'best_query': fig.best_query,
                'best_sbert_query': fig.best_sbert_query,
                'combined_score': getattr(fig, 'combined_score', 0.0),
                'zero_shot_confidence': getattr(fig, 'zero_shot_confidence', 0.0),
                'zero_shot_label': getattr(fig, 'zero_shot_label', ''),
            }
            results.append(record)
            total_saved += 1
            
            logger.info(f"✅ Saved circuit {total_saved}: {fig.caption[:80]}...")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = "quantum_circuits_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n📄 Results saved to {output_file}")
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"   Papers processed: {len(arxiv_ids)}")
        print(f"   Circuits detected: {len(df)}")
        
        if 'best_sbert_query' in df.columns:
            print(f"\n🔍 Distribution by query type:")
            query_counts = df['best_sbert_query'].value_counts()
            for query, count in query_counts.items():
                print(f"   {query}: {count}")
    
    else:
        print("⚠️ No quantum circuits were detected")
    
    print("\n✨ Processing completed!")

if __name__ == "__main__":
    main()