"""
Extract circuit LaTeX from tex files with labels.
"""

import re
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from config import CIRCUIT_PATTERNS, INPUT_TAR_FOLDER

@dataclass
class ExtractedCircuit:
    """Represents an extracted circuit from a paper."""
    paper_id: str
    tex_file: str
    circuit_text: str
    circuit_label: Optional[str]  # NEW: Store the label
    caption_text: Optional[str]   # NEW: Store caption if available
    circuit_type: str
    position: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'paper_id': self.paper_id,
            'tex_file': self.tex_file,
            'circuit_text': self.circuit_text,
            'circuit_label': self.circuit_label,
            'caption_text': self.caption_text,
            'circuit_type': self.circuit_type,
            'position': self.position,
            'length': len(self.circuit_text),
            'metadata': self.metadata,
        }

class CircuitExtractor:
    """Extract quantum circuits from LaTeX files with labels."""
    
    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern, re.DOTALL), name)
            for pattern, name in CIRCUIT_PATTERNS
        ]
    
    def extract_paper_id(self, filename: str) -> str:
        """Extract arXiv paper ID from filename."""
        patterns = [
            r'(\d{4}\.\d{4,5})',  # 1234.56789
            r'([a-z\-]+/\d{7})',  # quant-ph/1234567
            r'arXiv_(\d{4}\.\d{4,5})',  # arXiv_1234.56789
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return Path(filename).stem
    
    def _find_label_for_circuit(self, content: str, circuit_start: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Find label and caption for a circuit.
        Searches backwards from circuit position.
        
        Returns: (label, caption)
        """
        # Search backwards for \label{...}
        search_window = content[max(0, circuit_start - 2000):circuit_start]
        
        # Pattern for \label{...}
        label_pattern = r'\\label\{([^}]+)\}'
        label_match = re.search(label_pattern, search_window)
        label = label_match.group(1) if label_match else None
        
        # Pattern for \caption{...} (might be in figure environment)
        caption_pattern = r'\\caption\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'
        caption_match = re.search(caption_pattern, search_window)
        caption = caption_match.group(1) if caption_match else None
        
        # Also look for equation labels (eq:, equation:)
        if label and not any(prefix in label for prefix in ['eq:', 'equation:', 'fig:', 'figure:']):
            # Try to find better label in wider context
            wider_search = content[max(0, circuit_start - 5000):circuit_start]
            eq_label_pattern = r'\\label\{([^}]*?:[^}]+)\}'
            eq_match = re.search(eq_label_pattern, wider_search)
            if eq_match:
                label = eq_match.group(1)
        
        return label, caption
    
    def _find_context_around_circuit(self, content: str, circuit_start: int, circuit_end: int) -> Dict:
        """Extract context around the circuit."""
        # Get 500 chars before and after
        context_start = max(0, circuit_start - 500)
        context_end = min(len(content), circuit_end + 500)
        
        context = content[context_start:context_end]
        
        # Check if circuit is in figure environment
        in_figure = '\\begin{figure' in context[:context_start - circuit_start + 500]
        
        # Check if circuit is in equation environment
        in_equation = '\\begin{equation' in context[:context_start - circuit_start + 500]
        
        # Find section title if any
        section_pattern = r'\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}'
        section_match = re.search(section_pattern, content[max(0, circuit_start - 2000):circuit_start])
        section_title = section_match.group(1) if section_match else None
        
        return {
            'in_figure': in_figure,
            'in_equation': in_equation,
            'section_title': section_title,
            'context_preview': context,
        }
    
    def _is_valid_circuit(self, text: str) -> bool:
        """Validate if extracted text is a valid circuit."""
        # Must contain circuit elements
        circuit_keywords = ['\\qw', '\\gate', '\\ctrl', '\\targ', '\\meter', 
                           '\\cw', '\\lstick', '\\rstick', '\\qwx']
        if not any(keyword in text for keyword in circuit_keywords):
            return False
        
        # Must have balanced braces
        if text.count('{') != text.count('}'):
            return False
        
        # Must be reasonable length
        if len(text) < 50 or len(text) > 10000:
            return False
        
        return True
    
    def _clean_circuit_text(self, text: str) -> str:
        """Clean and normalize circuit text."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure it starts with \Qcircuit or \begin{qcircuit}
        if not (text.startswith('\\Qcircuit') or text.startswith('\\begin{qcircuit}')):
            # Try to find the actual start
            if '\\Qcircuit' in text:
                start = text.find('\\Qcircuit')
                text = text[start:]
            elif '\\begin{qcircuit}' in text:
                start = text.find('\\begin{qcircuit}')
                text = text[start:]
        
        return text
    
    def extract_from_content(self, content: str, paper_id: str, tex_file: str) -> List[ExtractedCircuit]:
        """Extract circuits from LaTeX content with labels."""
        circuits = []
        
        for pattern, pattern_name in self.compiled_patterns:
            for match in pattern.finditer(content):
                circuit_text = match.group(1)
                
                if self._is_valid_circuit(circuit_text):
                    cleaned_text = self._clean_circuit_text(circuit_text)
                    
                    # Find label and caption
                    label, caption = self._find_label_for_circuit(content, match.start())
                    
                    # Find context
                    context = self._find_context_around_circuit(content, match.start(), match.end())
                    
                    # Analyze circuit
                    metadata = self._analyze_circuit(cleaned_text)
                    metadata.update(context)  # Add context to metadata
                    
                    circuit = ExtractedCircuit(
                        paper_id=paper_id,
                        tex_file=tex_file,
                        circuit_text=cleaned_text,
                        circuit_label=label,
                        caption_text=caption,
                        circuit_type=pattern_name,
                        position=match.start(),
                        metadata=metadata
                    )
                    
                    circuits.append(circuit)
        
        # Remove duplicates (same circuit text in same paper)
        unique_circuits = []
        seen = set()
        
        for circuit in circuits:
            key = f"{circuit.paper_id}_{circuit.circuit_text[:100]}"
            if key not in seen:
                seen.add(key)
                unique_circuits.append(circuit)
        
        return unique_circuits
    
    def _analyze_circuit(self, circuit_text: str) -> Dict:
        """Analyze circuit to extract metadata."""
        analysis = {
            'gates': len(re.findall(r'\\gate\{[^}]*\}', circuit_text)),
            'controls': len(re.findall(r'\\ctrl\{[^}]*\}', circuit_text)),
            'targets': len(re.findall(r'\\targ', circuit_text)),
            'wires': len(re.findall(r'\\qw', circuit_text)),
            'measurements': len(re.findall(r'\\meter', circuit_text)),
            'labeled_qubits': len(re.findall(r'\\lstick\{[^}]*\}', circuit_text)),
            'classical_wires': len(re.findall(r'\\cw', circuit_text)),
            'lines': circuit_text.count('\\\\'),
            'has_parameters': '@C=' in circuit_text or '@R=' in circuit_text,
            'has_label_command': '\\label{' in circuit_text,  # Check if label is inside circuit
        }
        
        # Extract any label inside the circuit itself
        if '\\label{' in circuit_text:
            label_match = re.search(r'\\label\{([^}]+)\}', circuit_text)
            if label_match:
                analysis['internal_label'] = label_match.group(1)
        
        # Estimate qubit count
        ctrl_indices = set()
        for match in re.finditer(r'\\ctrl\{([^}]*)\}', circuit_text):
            try:
                index = int(match.group(1))
                ctrl_indices.add(index)
            except:
                pass
        
        labeled_qubits = analysis['labeled_qubits']
        max_wire = max(ctrl_indices) if ctrl_indices else 0
        analysis['estimated_qubits'] = max(labeled_qubits, max_wire + 1, 1)
        
        # Count unique gate types
        gate_types = defaultdict(int)
        for match in re.finditer(r'\\gate\{([^}]*)\}', circuit_text):
            gate_name = match.group(1)
            # Simplify gate names
            simple_name = gate_name.split('(')[0].strip() if '(' in gate_name else gate_name.strip()
            gate_types[simple_name] += 1
        
        analysis['gate_types'] = dict(gate_types)
        analysis['unique_gate_types'] = len(gate_types)
        
        # Check for specific algorithms
        algorithm_indicators = {
            'qft': any(keyword in circuit_text.lower() for keyword in ['qft', 'quantum fourier']),
            'grover': 'grover' in circuit_text.lower(),
            'shor': 'shor' in circuit_text.lower(),
            'teleportation': any(keyword in circuit_text.lower() for keyword in ['teleport', 'bell']),
            'vqe': any(keyword in circuit_text.lower() for keyword in ['vqe', 'variational']),
            'qaoa': 'qaoa' in circuit_text.lower(),
        }
        analysis['algorithm_indicators'] = {k: v for k, v in algorithm_indicators.items() if v}
        
        return analysis
    
    def extract_from_tar(self, tar_path: Path) -> List[ExtractedCircuit]:
        """Extract circuits from a tar.gz file."""
        all_circuits = []
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find .tex files
                tex_files = [m for m in tar.getmembers() 
                           if m.isfile() and m.name.lower().endswith('.tex')]
                
                for member in tex_files:
                    try:
                        file_obj = tar.extractfile(member)
                        if not file_obj:
                            continue
                        
                        # Read content
                        content = file_obj.read().decode('utf-8', errors='ignore')
                        paper_id = self.extract_paper_id(member.name)
                        
                        # Extract circuits
                        circuits = self.extract_from_content(content, paper_id, member.name)
                        all_circuits.extend(circuits)
                        
                        if circuits:
                            print(f"  Found {len(circuits)} circuits in {paper_id}")
                            for circuit in circuits[:3]:  # Show first 3
                                label_info = f" (label: {circuit.circuit_label})" if circuit.circuit_label else ""
                                print(f"    - {circuit.circuit_type}{label_info}")
                        
                    except Exception as e:
                        # Skip problematic files
                        continue
                        
        except Exception as e:
            print(f"Error processing {tar_path.name}: {e}")
        
        return all_circuits