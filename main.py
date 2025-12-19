import pandas as pd
from typing import List
import json

from config.settings import (
    ID_FILE,
    MAX_IMAGES,
    TOP_K_PER_PAPER,
    SIMILARITY_THRESHOLD,
    SBERT_MIN_SIM,
    USE_COMBINED_SCORE,
    TFIDF_WEIGHT,
    SBERT_WEIGHT,
    COMBINED_THRESHOLD,
    USE_STEMMING,
    USE_STOPWORDS,
    NORMALIZE_HYPHENS,
    USE_NEGATIVE_PENALTY,
    NEGATIVE_PENALTY_ALPHA,
    USE_CUSTOM_TFIDF_FEATURES,
    setup_directories,
)
from utils.file_utils import FileUtils
from utils.logging_utils import Logger
from pipeline.extraction_pipeline import ExtractionPipeline
from core.arxiv_validator import ArxivFilter


def print_final_stats(logger, pipeline, processed_count, skipped_count, arxiv_filter):
    print(f"\n{'='*80}")
    print(f"📊 FINAL EXTRACTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n🎯 PAPERS:")
    print(f"   Total papers checked: {processed_count + skipped_count}")
    print(f"   Quantum papers processed: {processed_count}")
    print(f"   Non-quantum papers skipped: {skipped_count}")
    if processed_count > 0:
        print(f"   Quantum percentage: {100*processed_count/(processed_count+skipped_count):.1f}%")
    
    print(f"\n📸 EXTRACTION FUNNEL:")
    print(f"   Total figures found: {pipeline.stats['total_figures_seen']}")
    if pipeline.stats['total_figures_seen'] > 0:
        papers_with_figures = pipeline.stats['papers_with_figures']
        papers_with_candidates = pipeline.stats['papers_with_candidates']
        papers_with_extracted = pipeline.stats['papers_with_extracted']
        
        print(f"   Papers with figures: {papers_with_figures}")
        print(f"   Papers with TF-IDF candidates: {papers_with_candidates} ({100*papers_with_candidates/papers_with_figures:.1f}%)")
        print(f"   Papers with SBERT-selected figures: {papers_with_extracted} ({100*papers_with_extracted/papers_with_figures:.1f}%)")
        print(f"   Final images saved: {pipeline.stats['total_saved']} ({100*pipeline.stats['total_saved']/pipeline.stats['total_figures_seen']:.1f}% of figures)")
    
    logger.info(f"\nImages saved: {pipeline.stats['total_saved']}")

    final_stats = arxiv_filter.get_cache_stats()
    logger.info(f"\nCache now has {final_stats['total']} entries")
    logger.info(f"Next run will be faster as {final_stats['total']} papers are cached")


def main():
    # Setup
    setup_directories()
    logger = Logger()
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return
    # Print configuration settings for tuning reference
    print(f"\n{'='*80}")
    print(f"⚙️  PIPELINE CONFIGURATION")
    print(f"{'='*80}")
    print(f"📊 THRESHOLDS & LIMITS:")
    print(f"   MAX_IMAGES: {MAX_IMAGES}")
    print(f"   TOP_K_PER_PAPER: {TOP_K_PER_PAPER}")
    print(f"   SIMILARITY_THRESHOLD (TF-IDF gate): {SIMILARITY_THRESHOLD}")

    if USE_COMBINED_SCORE:
        print(f"\n📊 SCORING MODE: COMBINED WEIGHTED")
        print(f"   TF-IDF Weight: {TFIDF_WEIGHT}")
        print(f"   SBERT Weight: {SBERT_WEIGHT}")
        print(f"   Combined Threshold: {COMBINED_THRESHOLD}")
    else:
        print(f"\n📊 SCORING MODE: CASCADE GATES")
        print(f"   SBERT_MIN_SIM (SBERT gate): {SBERT_MIN_SIM}")

    print(f"\n📊 TEXT PROCESSING:")
    print(f"   USE_STEMMING: {USE_STEMMING}")
    print(f"   USE_STOPWORDS: {USE_STOPWORDS}")
    print(f"   NORMALIZE_HYPHENS: {NORMALIZE_HYPHENS}")

    print(f"\n📊 NEGATIVE PENALTY:")
    print(f"   USE_NEGATIVE_PENALTY: {USE_NEGATIVE_PENALTY}")
    if USE_NEGATIVE_PENALTY:
        print(f"   NEGATIVE_PENALTY_ALPHA: {NEGATIVE_PENALTY_ALPHA}")

    print(f"\n📊 CUSTOM TF-IDF FEATURES:")
    print(f"   USE_CUSTOM_TFIDF_FEATURES: {USE_CUSTOM_TFIDF_FEATURES}")

    print(f"{'='*80}\n")
    
    # Read arXiv IDs
    arxiv_ids = FileUtils.read_arxiv_ids(ID_FILE)
    logger.info(f" Loaded {len(arxiv_ids)} arXiv IDs from {ID_FILE}")
    
    # Initialize quantum paper filter with caching
    arxiv_filter = ArxivFilter()
    cache_stats = arxiv_filter.get_cache_stats()
    logger.info(f" Cache has {cache_stats['total']} entries ({cache_stats['quantum_percentage']:.1f}% quantum)")
    
    processed_count = 0
    skipped_count = 0
    latex_circuit_embedded = True
    collected_250_stop_latex_circuit_img = True

    if collected_250_stop_latex_circuit_img:
        logger.info(" Already collected 250 images, skipping further processing.")
        try:
            for arxiv_id in arxiv_ids:
                # Stop if we have enough images
                if pipeline.stats['total_saved'] >= MAX_IMAGES:
                    logger.info(f" Reached maximum images limit ({MAX_IMAGES})")
                    break
                
                # Check if quantum paper
                if not arxiv_filter.is_quantum_paper(arxiv_id):
                    skipped_count += 1
                    logger.info(f" Skipped {skipped_count} non-quantum papers so far...")
                    continue
                
                # Process quantum paper
                processed_count += 1
                extracted, figures = pipeline.process_paper(arxiv_id)
                
                # Update progress
                if processed_count % 10 == 0:
                    logger.info(
                        f" Processed {processed_count} papers, "
                        f"saved {pipeline.stats['total_saved']}/{MAX_IMAGES} images"
                    )

                # check for 250 images in circuit.json once 250 reached save checkpoint information in latex_code_circuit_checkpoint.jsonl
                

                # open circuits.json using UTF-8 to avoid platform default codec issues
                with open("./data/circuits.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                num_keys = len(data)
                if num_keys >= 250 and latex_circuit_embedded == True:
                    with open("./data/latex_code_circuit_checkpoint.json", "w") as checkpoint_file:
                        for fig in figures:
                            record = {
                                "total_papers_processed": processed_count + skipped_count
                            }
                            checkpoint_file.write(json.dumps(record) + "\n")
                            latex_circuit_embedded = False
                            collected_250_stop_latex_circuit_img = False
                    logger.info(" Reached 250 images, saved checkpoint to latex_code_circuit_checkpoint.jsonl")
                

        except KeyboardInterrupt:
            logger.warning(
                "\n⚠️ Keyboard interrupt detected — saving results..."
            )

    # Save results (runs after normal completion OR keyboard interrupt)
    if pipeline.text_records:
        df = pd.DataFrame(pipeline.text_records)
        FileUtils.save_dataframe(df)
        logger.print_statistics(pipeline.stats, df)
    else:
        logger.warning("No images were extracted")

    print_final_stats(logger, pipeline, processed_count, skipped_count, arxiv_filter)
    logger.info("Extraction completed!")


if __name__ == "__main__":
    main()
