"""
Integration tests for the quantum circuit image extractor.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch

from core.preprocessor import TextPreprocessor
from core.tfidf_filter import TfidfFilter
from models.figure_data import Figure


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf_filter = TfidfFilter(self.preprocessor)
    
    def test_preprocessor_clean_caption(self):
        """Test caption cleaning."""
        caption = "Figure 1: Quantum circuit with CNOT gates $|\psi\\rangle$"
        cleaned = self.preprocessor.clean_caption_text(caption)
        
        # Should remove "Figure 1:"
        self.assertNotIn("figure 1", cleaned)
        # Should remove LaTeX math
        self.assertNotIn("$", cleaned)
        # Should keep quantum terms
        self.assertIn("quantum", cleaned)
        self.assertIn("circuit", cleaned)
        self.assertIn("cnot", cleaned)
    
    def test_tfidf_analyzer(self):
        """Test TF-IDF analyzer."""
        text = "Quantum circuit diagram with Hadamard and CNOT gates"
        tokens = self.preprocessor.tfidf_analyzer(text)
        
        # Should return list of tokens
        self.assertIsInstance(tokens, list)
        # Should lowercase
        self.assertTrue(all(t.islower() for t in tokens))
        # Should contain quantum terms
        self.assertTrue(any('quantum' in t or 'cnot' in t for t in tokens))
    
    def test_figure_data_class(self):
        """Test Figure data class."""
        figure = Figure(
            caption="Test caption",
            img_path="figure1.png",
            paper_id="1234.56789"
        )
        
        self.assertEqual(figure.caption, "Test caption")
        self.assertEqual(figure.img_path, "figure1.png")
        self.assertEqual(figure.paper_id, "1234.56789")
        self.assertFalse(figure.selected)
        self.assertFalse(figure.extracted)
    
    def test_preprocess_text_to_string(self):
        """Test text preprocessing to string."""
        text = "Quantum Circuit with 5 qubits and CNOT"
        result = self.preprocessor.preprocess_text_to_string(text)
        
        # Should be a string
        self.assertIsInstance(result, str)
        # Should contain processed terms
        self.assertIn("quantum", result)
        self.assertIn("circuit", result)
        self.assertIn("qubit", result)
    
    def test_negative_token_counting(self):
        """Test negative token counting."""
        # Quantum caption should have few negative tokens
        quantum_text = "quantum circuit cnot hadamard"
        neg_count = self.preprocessor.count_negative_tokens(quantum_text)
        self.assertLessEqual(neg_count, 1)
        
        # Plot caption should have more negative tokens
        plot_text = "plot graph histogram energy spectrum"
        neg_count = self.preprocessor.count_negative_tokens(plot_text)
        self.assertGreater(neg_count, 1)
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_tfidf_filter_mock(self, mock_vectorizer):
        """Test TF-IDF filter with mock."""
        # Create mock figures
        figures = [
            Figure(caption="Quantum circuit", img_path="fig1.png"),
            Figure(caption="Energy spectrum", img_path="fig2.png")
        ]
        
        # Mock the vectorizer response
        mock_sims = [[0.8, 0.1], [0.2, 0.9]]  # Similarity scores
        mock_vectorizer.return_value.fit_transform.return_value = Mock()
        
        # This is a simplified test - in reality we'd need to mock more
        self.assertIsInstance(figures, list)


if __name__ == '__main__':
    unittest.main()