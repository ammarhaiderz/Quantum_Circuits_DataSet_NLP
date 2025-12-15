"""
Fast zero-shot classifier for quantum circuit detection.
"""

from transformers import pipeline
import torch
from typing import List, Dict
import time

from config.settings import ZERO_SHOT_MODEL, ZERO_SHOT_THRESHOLD
from models.figure_data import Figure

class FastZeroShotClassifier:
    """Fast zero-shot classifier for pre-filtering."""
    
    def __init__(self, model_name=ZERO_SHOT_MODEL):
        """
        Initialize with a small, fast zero-shot model.
        """
        print(f"📦 Loading zero-shot classifier: {model_name}")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            # Optimized labels for quantum circuit detection
            self.labels = [
                "quantum circuit diagram with gates and wires",
                "scientific plot, chart, or graph",
                "device schematic or hardware diagram", 
                "conceptual diagram or flowchart",
                "mathematical equation or formula",
                "data visualization or results figure",
                "energy level diagram or spectrum",
                "3D rendering or simulation output",
                "microscope image or experimental photo",
                "table of data or parameter list"
            ]
            
            print(f"✅ Zero-shot classifier loaded")
            
        except Exception as e:
            print(f"❌ Failed to load zero-shot classifier: {e}")
            self.classifier = None
    
    def is_quantum_circuit(self, caption: str) -> Dict:
        """
        Predict if caption describes a quantum circuit.
        
        Returns:
            Dict with 'is_circuit' (bool) and 'confidence' (float)
        """
        if not self.classifier or not caption or len(caption.strip()) < 5:
            return {
                'is_circuit': False,
                'confidence': 0.0,
                'top_label': 'invalid',
                'reason': 'no_classifier_or_short_caption'
            }
        
        try:
            # Run classification
            result = self.classifier(
                caption,
                self.labels,
                multi_label=False,
                hypothesis_template="This image shows {}"
            )
            
            # Extract top result
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # Determine if it's a circuit
            is_circuit = (
                top_label == "quantum circuit diagram with gates and wires" and 
                top_score >= ZERO_SHOT_THRESHOLD
            )
            
            return {
                'is_circuit': is_circuit,
                'confidence': top_score,
                'top_label': top_label,
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
            
        except Exception as e:
            print(f"⚠️ Zero-shot classification failed: {e}")
            return {
                'is_circuit': False,
                'confidence': 0.0,
                'top_label': 'error',
                'reason': str(e)
            }
    
    def batch_classify(self, figures: List[Figure]) -> List[Figure]:
        """Classify multiple figures."""
        if not self.classifier:
            return figures
        
        for fig in figures:
            result = self.is_quantum_circuit(fig.caption or "")
            fig.zero_shot_is_circuit = result['is_circuit']
            fig.zero_shot_confidence = result['confidence']
            fig.zero_shot_label = result['top_label']
        
        return figures
    
    def filter_circuits(self, figures: List[Figure]) -> List[Figure]:
        """Filter to only quantum circuit figures."""
        if not self.classifier:
            return figures
        
        circuit_figures = []
        for fig in figures:
            if fig.zero_shot_is_circuit:
                circuit_figures.append(fig)
        
        print(f"🔍 Zero-shot filter: {len(circuit_figures)}/{len(figures)} identified as circuits")
        return circuit_figures