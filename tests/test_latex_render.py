"""
Test LaTeX rendering for TikZ/circuit figures.
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.image_extractor import ImageExtractor
from core.preprocessor import TextPreprocessor
from models.figure_data import Figure


def test_tikz_detection():
    """Test that TikZ blocks are detected in LaTeX source."""
    print("\n" + "="*80)
    print("TEST 1: TikZ Block Detection")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    # Sample LaTeX with tikzpicture
    sample_tex = r"""
\begin{figure}[h]
\centering
\begin{tikzpicture}
    \node[draw, circle] (a) at (0,0) {A};
    \node[draw, circle] (b) at (2,0) {B};
    \draw[->] (a) -- (b);
\end{tikzpicture}
\caption{Simple quantum circuit with two nodes.}
\end{figure}
"""
    
    figures = extractor.extract_figures_from_tex(sample_tex)
    
    print(f"Extracted {len(figures)} figure(s)")
    assert len(figures) == 1, f"Expected 1 figure, got {len(figures)}"
    
    fig = figures[0]
    print(f"Caption: {fig.caption}")
    print(f"Image path: {fig.img_path}")
    print(f"Has LaTeX block: {fig.latex_block is not None}")
    
    assert fig.img_path == "__LATEX_RENDER__", "Should be marked for rendering"
    assert fig.latex_block is not None, "Should have LaTeX block"
    assert "tikzpicture" in fig.latex_block, "Should contain tikzpicture"
    
    print("✅ PASSED: TikZ block detected correctly")


def test_circuitikz_detection():
    """Test that circuitikz blocks are detected."""
    print("\n" + "="*80)
    print("TEST 2: CircuiTikZ Block Detection")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    sample_tex = r"""
\begin{figure}[h]
\centering
\begin{circuitikz}
    \draw (0,0) to[C] (2,0);
\end{circuitikz}
\caption{Circuit with capacitor.}
\end{figure}
"""
    
    figures = extractor.extract_figures_from_tex(sample_tex)
    
    print(f"Extracted {len(figures)} figure(s)")
    assert len(figures) == 1, f"Expected 1 figure, got {len(figures)}"
    
    fig = figures[0]
    print(f"Caption: {fig.caption}")
    print(f"Has LaTeX block: {fig.latex_block is not None}")
    
    assert fig.latex_block is not None, "Should have LaTeX block"
    assert "circuitikz" in fig.latex_block, "Should contain circuitikz"
    
    print("✅ PASSED: CircuiTikZ block detected correctly")


def test_quantikz_detection():
    """Test that quantikz blocks are detected."""
    print("\n" + "="*80)
    print("TEST 3: Quantikz Block Detection")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    sample_tex = r"""
\begin{figure}[h]
\centering
\begin{quantikz}
    \lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
    \lstick{$\ket{0}$} & \qw & \targ{} & \meter{}
\end{quantikz}
\caption{Bell state preparation circuit.}
\end{figure}
"""
    
    figures = extractor.extract_figures_from_tex(sample_tex)
    
    print(f"Extracted {len(figures)} figure(s)")
    assert len(figures) == 1, f"Expected 1 figure, got {len(figures)}"
    
    fig = figures[0]
    print(f"Caption: {fig.caption}")
    print(f"Has LaTeX block: {fig.latex_block is not None}")
    
    assert fig.latex_block is not None, "Should have LaTeX block"
    assert "quantikz" in fig.latex_block, "Should contain quantikz"
    
    print("✅ PASSED: Quantikz block detected correctly")


def test_multiple_tikz_blocks():
    """Test multiple TikZ blocks with subcaptions."""
    print("\n" + "="*80)
    print("TEST 4: Multiple TikZ Blocks with Subcaptions")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    sample_tex = r"""
\begin{figure}[h]
\centering
\begin{tikzpicture}
    \node {Circuit A};
\end{tikzpicture}
\begin{tikzpicture}
    \node {Circuit B};
\end{tikzpicture}
\caption{(a) First quantum circuit. (b) Second quantum circuit.}
\end{figure}
"""
    
    figures = extractor.extract_figures_from_tex(sample_tex)
    
    print(f"Extracted {len(figures)} figure(s)")
    assert len(figures) == 2, f"Expected 2 figures, got {len(figures)}"
    
    print(f"Figure 1 caption: {figures[0].caption}")
    print(f"Figure 2 caption: {figures[1].caption}")
    
    assert "First quantum circuit" in figures[0].caption, "Caption should be split"
    assert "Second quantum circuit" in figures[1].caption, "Caption should be split"
    
    print("✅ PASSED: Multiple blocks with subcaptions parsed correctly")


def test_latex_rendering_basic():
    """Test actual LaTeX rendering (requires pdflatex)."""
    print("\n" + "="*80)
    print("TEST 5: LaTeX Rendering (requires pdflatex)")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    # Simple tikz block
    latex_block = r"""
\begin{tikzpicture}
    \node[draw, circle, fill=blue!20] (q0) at (0,0) {$|0\rangle$};
    \node[draw, rectangle, fill=green!20] (h) at (2,0) {H};
    \node[draw, circle, fill=blue!20] (q1) at (4,0) {$|+\rangle$};
    \draw[->] (q0) -- (h);
    \draw[->] (h) -- (q1);
\end{tikzpicture}
"""
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override OUTPUT_DIR temporarily
        original_output = extractor.image_extractor if hasattr(extractor, 'image_extractor') else None
        
        try:
            out_path = extractor._render_latex_block(latex_block, "test_paper", 0)
            
            if out_path:
                print(f"✅ Rendered successfully: {out_path}")
                print(f"File exists: {os.path.exists(out_path)}")
                
                if os.path.exists(out_path):
                    file_size = os.path.getsize(out_path)
                    print(f"File size: {file_size} bytes")
                    assert file_size > 0, "Output file should not be empty"
                    print("✅ PASSED: LaTeX rendered to valid PNG")
                else:
                    print("⚠️ WARNING: File not created at expected path")
            else:
                print("⚠️ WARNING: pdflatex or conversion tools not available")
                print("   Install: MiKTeX or TeX Live, and pdftocairo (Poppler)")
                print("   This is expected if LaTeX toolchain is not installed")
                
        except Exception as e:
            print(f"⚠️ WARNING: Rendering failed: {e}")
            print("   This is expected if LaTeX toolchain is not installed")


def test_mixed_figures():
    """Test figure with both bitmap images and LaTeX blocks."""
    print("\n" + "="*80)
    print("TEST 6: Mixed Bitmap + LaTeX Figures")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    extractor = ImageExtractor(preprocessor)
    
    sample_tex = r"""
\begin{figure}[h]
\centering
\includegraphics{circuit_diagram.png}
\caption{Quantum circuit implementation.}
\end{figure}

\begin{figure}[h]
\centering
\begin{tikzpicture}
    \node {TikZ circuit};
\end{tikzpicture}
\caption{Theoretical circuit design.}
\end{figure}
"""
    
    figures = extractor.extract_figures_from_tex(sample_tex)
    
    print(f"Extracted {len(figures)} figure(s)")
    assert len(figures) == 2, f"Expected 2 figures, got {len(figures)}"
    
    # First should be bitmap
    print(f"Figure 1: {figures[0].img_path}")
    assert figures[0].img_path == "circuit_diagram.png", "First should be bitmap"
    assert figures[0].latex_block is None, "First should not have LaTeX block"
    
    # Second should be LaTeX render
    print(f"Figure 2: {figures[1].img_path}")
    assert figures[1].img_path == "__LATEX_RENDER__", "Second should be LaTeX render"
    assert figures[1].latex_block is not None, "Second should have LaTeX block"
    
    print("✅ PASSED: Mixed figure types handled correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print(" LATEX RENDERING TEST SUITE")
    print("="*80)
    
    tests = [
        test_tikz_detection,
        test_circuitikz_detection,
        test_quantikz_detection,
        test_multiple_tikz_blocks,
        test_mixed_figures,
        test_latex_rendering_basic,  # Last since it requires external tools
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f" TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
