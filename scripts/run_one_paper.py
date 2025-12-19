import sys
import json
from pipeline.extraction_pipeline import ExtractionPipeline
from utils.logging_utils import Logger


def run_one(paper_id: str):
    logger = Logger()
    pipeline = ExtractionPipeline()
    # Avoid heavy SBERT model load for quick include/skipped test: do not call initialize()
    # instead, noop SBERT methods so process_paper can run faster for our inspection.
    try:
        pipeline.sbert_reranker.load_model = lambda: None
        pipeline.sbert_reranker.prepare_query_embeddings = lambda: None
        pipeline.sbert_reranker.rerank_figures = lambda figs: figs
    except Exception:
        pass

    print(f"Running pipeline for paper: {paper_id} (SBERT load skipped for speed)")
    extracted, figures = pipeline.process_paper(paper_id)

    print(f"Done. Extracted images: {len(extracted)}; Figures parsed: {len(figures)}")

    # Save minimal summary
    summary = {
        "paper_id": paper_id,
        "figures_parsed": len(figures),
        "images_extracted": len(extracted)
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pid = sys.argv[1]
    else:
        pid = '2404.14865'
    sys.exit(run_one(pid))
