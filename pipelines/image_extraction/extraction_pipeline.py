"""
Main pipeline for quantum circuit image extraction.
"""

from typing import List, Tuple
import re
import tarfile
import os
import shutil
from pathlib import Path

from shared.figure_data import Figure, ExtractedImage
from shared.preprocessor import TextPreprocessor
from pipelines.image_extraction.tfidf_filter import TfidfFilter
from pipelines.image_extraction.sbert_reranker import SbertReranker
from pipelines.image_extraction.image_extractor import ImageExtractor
from config.settings import (
    MAX_IMAGES, TOP_K_PER_PAPER, PRINT_TOP_CAPTIONS,
    OUTPUT_DIR, ENABLE_DEBUG_PRINTS, SBERT_MIN_SIM,
    USE_COMBINED_SCORE, TFIDF_WEIGHT, SBERT_WEIGHT, COMBINED_THRESHOLD,
    SIMILARITY_THRESHOLD
)
from core.circuit_store import set_quantum_problem_model


class ExtractionPipeline:
    """Orchestrates the entire extraction pipeline."""
    
    def __init__(self):
        """Initialize pipeline components and statistics containers."""
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
        """Initialize components, clean outputs, and load models.

        Returns
        -------
        bool
            ``True`` on successful initialization; ``False`` otherwise.
        """
        print("[INIT] Initializing Quantum Circuit Image Extractor Pipeline")

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
                    print(f"[WARN] Failed to remove {p}: {e}")
        else:
            ci.mkdir(parents=True, exist_ok=True)

        # Ensure common subdirectories exist after cleanup
        (ci / 'live_blocks').mkdir(parents=True, exist_ok=True)
        (ci / 'rendered_pdflatex').mkdir(parents=True, exist_ok=True)

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
            from pipelines.latex_render.live_latex_extractor import reset_processed_blocks
            reset_processed_blocks()
        except Exception:
            pass
        
        
        # Load SBERT model
        try:
            self.sbert_reranker.load_model()
            self.sbert_reranker.prepare_query_embeddings()
            # Register the loaded SBERT model for quantum_problem classification
            try:
                set_quantum_problem_model(self.sbert_reranker.model)
                print("[OK] Quantum-problem classifier enabled (SBERT model registered)")
            except Exception as e:
                print(f"[WARN] Could not register quantum-problem classifier: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize SBERT: {e}")
            return False
        
        print("[OK] Pipeline initialized successfully")
        return True
    
    def process_paper(self, paper_id: str) -> Tuple[List[ExtractedImage], List[Figure]]:
        """Process a single paper end-to-end.

        Parameters
        ----------
        paper_id : str
            arXiv paper identifier.

        Returns
        -------
        tuple[list[ExtractedImage], list[Figure]]
            Extracted images and all parsed figures (with scores/metadata).
        """
        self.stats['papers_checked'] += 1
        print(f"\n{'='*80}")
        print(f"[PAPER] {paper_id}")
        print(f"{'='*80}")
        
        # Cache PDF paper as we process it
        self.image_extractor.download_pdf_paper(paper_id)
        
        src = self.image_extractor.download_source(paper_id)
        if not src:
            print(f"[ERROR] Failed to download source for {paper_id}")
            return [], []
        
        try:
            tar = tarfile.open(fileobj=src, mode="r:gz")
        except Exception as e:
            print(f"[ERROR] Failed to open tar file for {paper_id}: {e}")
            return [], []
        
        # Extract figures from LaTeX
        figures = []
        # Reset live-render numbering per paper
        self.image_extractor.figure_counter = 1

        # Heuristic: only process .tex files that are actually included from a
        # root document. Many arXiv source tarballs contain extra/auxiliary .tex
        # files that are not part of the compiled PDF; scanning only included
        # files avoids rendering unrelated circuits.
        try:
            # Read all .tex members first
            tex_members = {}
            for m in tar.getmembers():
                if m.name.endswith('.tex'):
                    try:
                        tex_members[m.name] = tar.extractfile(m).read().decode('utf-8', 'ignore')
                    except Exception:
                        tex_members[m.name] = ''

            def find_included_tex_names(tex_dict: dict) -> set:
                r"""Return set of tex member names that are directly referenced from a
                single chosen main/root file.

                Behavior changed: do NOT recursively traverse included files. Only
                parse the chosen root's text for top-level \input/\include/.. targets
                and match those targets against archive member names (by suffix).
                """
                include_re = re.compile(r"\\(?:input|include|subfile|import|subimport)\{([^}]+)\}")

                # choose a single main root as before (prefer \documentclass)
                docclass_roots = [name for name, txt in tex_dict.items() if '\\documentclass' in txt]
                if docclass_roots:
                    if 'main.tex' in docclass_roots:
                        root = 'main.tex'
                    else:
                        root = sorted(docclass_roots, key=lambda s: (s.count('/'), len(s)))[0]
                else:
                    begin_roots = [name for name, txt in tex_dict.items() if '\\begin{document}' in txt]
                    if begin_roots:
                        root = sorted(begin_roots, key=lambda s: (s.count('/'), len(s)))[0]
                    else:
                        tops = [name for name in tex_dict.keys() if '/' not in name and '\\' not in name]
                        if tops:
                            root = tops[0]
                        else:
                            root = sorted(tex_dict.keys(), key=lambda s: (s.count('/'), len(s)))[0]

                # Start with root itself
                referenced = set([root])
                txt = tex_dict.get(root, '')
                for mm in include_re.finditer(txt):
                    ref = mm.group(1).strip()
                    # normalize: remove leading ./ and trailing .tex
                    ref_norm = ref.lstrip('./')
                    if ref_norm.endswith('.tex'):
                        ref_norm = ref_norm[:-4]

                    # match against available member names by suffix only (no recursion)
                    for member_name in tex_dict.keys():
                        cand = member_name
                        cand_norm = cand[:-4] if cand.endswith('.tex') else cand
                        if cand_norm == ref_norm or cand_norm.endswith('/' + ref_norm) or cand_norm.endswith(ref_norm):
                            referenced.add(member_name)

                return referenced

            included_names = find_included_tex_names(tex_members)

            # If nothing found (empty included_names), fall back to processing all .tex files
            if not included_names:
                included_iter = [name for name in tex_members.keys()]
                chosen_root = None
            else:
                included_iter = included_names
                # the find_included_tex_names now selects a single root; try to surface it
                # find the single root as the one that is referenced but not referenced by others
                chosen_root = None
                # attempt to infer root by checking which included file appears at start of traversal
                # (we used a deterministic choice in traversal - select the first of included_iter by path depth)
                try:
                    chosen_root = sorted(list(included_iter), key=lambda s: (s.count('/'), len(s)))[0]
                except Exception:
                    chosen_root = None

            # Debug: report which .tex files are being skipped to help triage auxiliary files
            try:
                if ENABLE_DEBUG_PRINTS:
                    all_names = set(tex_members.keys())
                    skipped = sorted(list(all_names - set(included_iter)))
                    if chosen_root:
                        print(f"   [INFO] Chosen main root .tex: {chosen_root}")
                    # Print the canonical included list (sorted) so the chosen root is obvious
                    try:
                        inc_list = sorted(list(included_iter), key=lambda s: (s.count('/'), s))
                    except Exception:
                        inc_list = list(included_iter)
                    print(f"   [INFO] Included .tex files ({len(inc_list)}): {inc_list}")
                    if skipped:
                        print(f"   [INFO] Skipping {len(skipped)} auxiliary .tex files (not included by root): {skipped[:10]}{'...' if len(skipped)>10 else ''}")
            except Exception:
                pass

            for name in included_iter:
                try:
                    tex = tex_members.get(name, '')
                    figures.extend(self.image_extractor.extract_figures_from_tex(tex, paper_id=paper_id))
                except Exception as e:
                    print(f"[WARN] Failed to parse {name}: {e}")
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
            print(f"[WARN] Error iterating archive for {paper_id}: {e}")
            return [], []
        
        if not figures:
            print(f"[WARN] No figures found in LaTeX")
            return [], []

        # Try to populate page numbers for records from this paper (best-effort)
        try:
            from core.circuit_store import update_pages_in_jsonl
            try:
                updated = update_pages_in_jsonl(paper_id)
                if updated:
                    print(f"   [OK] Updated {updated} page numbers for {paper_id}")
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
        print(f"\n[GATE 0] FIGURE EXTRACTION")
        print(f"   [OK] Figures extracted: {len(figures)}")
        
        # Apply TF-IDF filtering
        figures = self.tfidf_filter.filter_figures(figures)
        
        # Debug print TF-IDF results
        if ENABLE_DEBUG_PRINTS:
            self._print_tfidf_debug(figures)
        
        # Get TF-IDF accepted figures
        accepted_tfidf = self.tfidf_filter.get_accepted_figures(figures)
        
        print(f"\n GATE 1: TF-IDF FILTER (threshold={SIMILARITY_THRESHOLD})")
        print(f" Input: {len(figures)} figures")
        print(f" Passed: {len(accepted_tfidf)} figures")
        print(f" Rejected: {len(figures) - len(accepted_tfidf)} figures")
        
        if accepted_tfidf:
            self.stats['papers_with_candidates'] += 1
            # Statistics for tuning
            tfidf_scores = [f.similarity for f in accepted_tfidf]
            raw_scores = [f.similarity_raw for f in accepted_tfidf]
            neg_counts = [f.negative_tokens for f in accepted_tfidf]
            print(f"   [STATS] TF-IDF scores - Min: {min(tfidf_scores):.4f}, Max: {max(tfidf_scores):.4f}, Avg: {sum(tfidf_scores)/len(tfidf_scores):.4f}")
            print(f"   [STATS] Raw scores - Min: {min(raw_scores):.4f}, Max: {max(raw_scores):.4f}")
            print(f"   [STATS] Negative tokens - Min: {min(neg_counts)}, Max: {max(neg_counts)}, Avg: {sum(neg_counts)/len(neg_counts):.1f}")
        
        # Limit pool for SBERT reranking
        accepted_tfidf = accepted_tfidf[:TOP_K_PER_PAPER * 3]
        print(f"   -> Limited to top {len(accepted_tfidf)} for SBERT reranking")
        
        # Initialize accepted list
        accepted = []
        
        # Apply SBERT reranking
        if accepted_tfidf:
            accepted_tfidf = self.sbert_reranker.rerank_figures(accepted_tfidf)
            
            print(f"\n GATE 2: SBERT RERANKING")
            print(f" Input: {len(accepted_tfidf)} figures")

            sbert_scores = [f.sbert_sim for f in accepted_tfidf]
            print(f"   [STATS] SBERT scores - Min: {min(sbert_scores):.4f}, Max: {max(sbert_scores):.4f}, Avg: {sum(sbert_scores)/len(sbert_scores):.4f}")
            
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
                print(f"\n GATE 3: COMBINED SCORE FILTER (threshold={COMBINED_THRESHOLD})")
                print(f"   Weights: TF-IDF={TFIDF_WEIGHT}, SBERT={SBERT_WEIGHT}")
                print(f"   [STATS] Combined scores - Min: {min(combined_scores):.4f}, Max: {max(combined_scores):.4f}, Avg: {sum(combined_scores)/len(combined_scores):.4f}")
            else:
                # Legacy cascade approach: sort by SBERT only
                accepted_tfidf = sorted(
                    accepted_tfidf,
                    key=lambda x: x.sbert_sim,
                    reverse=True
                )
                # Final selection: filter by SBERT threshold
                accepted = [f for f in accepted_tfidf if f.sbert_sim >= SBERT_MIN_SIM]
                
                print(f"\n[GATE 3] SBERT FILTER (threshold={SBERT_MIN_SIM})")
            
            # Common pass/reject logging for both modes
            print(f" Passed: {len(accepted)} figures")
            print(f" Rejected: {len(accepted_tfidf) - len(accepted)} figures")

            accepted = accepted[:TOP_K_PER_PAPER]
            print(f"   Limited to top {TOP_K_PER_PAPER} for final output")
        
        # Mark selected figures
        for f in accepted:
            f.selected = True
        
        # Extract images
        extracted = self.image_extractor.extract_images(tar, accepted, paper_id)
        
        print(f"\n[GATE 4] IMAGE EXTRACTION")
        print(f"   -> Input: {len(accepted)} figures")
        print(f"   [OK] Successfully extracted: {len(extracted)} images")
        print(f"   [FAIL] Failed: {len(accepted) - len(extracted)} images")
        
        if extracted:
            self.stats['papers_with_extracted'] += 1
        
        self.stats['total_saved'] += len(extracted)
        self.all_extracted.extend(extracted)
        
        print(f"\n[STATS] CUMULATIVE STATS")
        print(f"   Total papers processed: {self.stats['papers_checked']}")
        print(f"   Total images saved: {self.stats['total_saved']}/{MAX_IMAGES}")
        
        # Create records for DataFrame (only TF-IDF accepted figures, not all)
        self._create_records(paper_id, accepted_tfidf, extracted)
        
        # Print summary for this paper
        self._print_paper_summary(paper_id, figures, accepted_tfidf, accepted, extracted)
        
        return extracted, figures
    
    def _create_records(self, paper_id: str, figures: List[Figure], extracted: List[ExtractedImage]):
        """Create per-figure records for tabular export/logging.

        Parameters
        ----------
        paper_id : str
            arXiv paper identifier.
        figures : list[Figure]
            Figures considered after filtering.
        extracted : list[ExtractedImage]
            Successfully extracted images.
        """
        extracted_lookup = {e.img_name: e for e in extracted}
        
        for f in figures:
            img_name = os.path.basename(f.img_path)
            e = extracted_lookup.get(img_name)
            
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
                print(f"    [OK] {img.img_name}")
            if len(extracted) > 3:
                print(f"    ... and {len(extracted) - 3} more")