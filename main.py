import pandas as pd
from typing import List

from config.settings import ID_FILE, MAX_IMAGES, setup_directories
from utils.file_utils import FileUtils
from utils.logging_utils import Logger
from pipeline.extraction_pipeline import ExtractionPipeline
from core.arxiv_validator import ArxivFilter


def print_final_stats(logger, pipeline, processed_count, skipped_count, arxiv_filter):
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total papers checked: {processed_count + skipped_count}")
    logger.info(f"Quantum papers processed: {processed_count}")
    logger.info(f"Non-quantum papers skipped: {skipped_count}")
    logger.info(f"Images saved: {pipeline.stats['total_saved']}")

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
    
    # Read arXiv IDs
    arxiv_ids = FileUtils.read_arxiv_ids(ID_FILE)
    logger.info(f" Loaded {len(arxiv_ids)} arXiv IDs from {ID_FILE}")
    
    # Initialize quantum paper filter with caching
    arxiv_filter = ArxivFilter()
    cache_stats = arxiv_filter.get_cache_stats()
    logger.info(f" Cache has {cache_stats['total']} entries ({cache_stats['quantum_percentage']:.1f}% quantum)")
    
    processed_count = 0
    skipped_count = 0

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

    except KeyboardInterrupt:
        logger.warning(
            "\n⚠️ Keyboard interrupt detected — saving results..."
        )

    # Save results (runs after normal completion OR keyboard interrupt)
    if pipeline.text_records:
        df = pd.DataFrame(pipeline.text_records)
        FileUtils.save_dataframe(df)
        
        # Print current threshold settings
        from config.settings import (
            SIMILARITY_THRESHOLD, SBERT_MIN_SIM,
            USE_COMBINED_SCORE, TFIDF_WEIGHT, SBERT_WEIGHT, COMBINED_THRESHOLD
        )
        logger.info("\n Current Settings:")
        logger.info(f"  SIMILARITY_THRESHOLD (TF-IDF gate): {SIMILARITY_THRESHOLD}")
        
        if USE_COMBINED_SCORE:
            logger.info(f"  Scoring Mode: COMBINED WEIGHTED")
            logger.info(f"    TF-IDF Weight: {TFIDF_WEIGHT}")
            logger.info(f"    SBERT Weight: {SBERT_WEIGHT}")
            logger.info(f"    Combined Threshold: {COMBINED_THRESHOLD}")
        else:
            logger.info(f"  Scoring Mode: CASCADE GATES")
            logger.info(f"    SBERT_MIN_SIM (SBERT gate): {SBERT_MIN_SIM}")
        
        logger.print_statistics(pipeline.stats, df)
    else:
        logger.warning("No images were extracted")

    print_final_stats(logger, pipeline, processed_count, skipped_count, arxiv_filter)
    logger.info("Extraction completed!")


if __name__ == "__main__":
    main()
