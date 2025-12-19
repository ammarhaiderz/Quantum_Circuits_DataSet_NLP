"""
Fixed scanner for complete \Qcircuit blocks - handles escape sequences and corrupted files.
"""

import os
import re
import tarfile
import json
import gzip
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

class QCircuitBlockCounter:
    """Fixed scanner that finds and counts complete \Qcircuit blocks."""
    
    def __init__(self, tar_folder: str, output_file: str = "qcircuit_block_counts.json"):
        self.tar_folder = Path(tar_folder)
        self.output_file = output_file
        
        # FIXED: Use raw strings for regex patterns
        self.circuit_patterns = [
            # Complete \Qcircuit blocks with braces
            (r'(\\Qcircuit\s*(?:@[A-Za-z]=[0-9\.]+(?:em|pt))?\s*\{.*?\})', 'basic_qcircuit'),
            
            # Multi-line circuits
            (r'(\\Qcircuit[^{]*\{[^}]*\\\\[^}]*\})', 'multiline_qcircuit'),
            
            # Circuits with lstick commands
            (r'(\\Qcircuit[^{]*\{[^}]*\\lstick\{[^}]*\}[^}]*\})', 'labeled_circuit'),
        ]
        
        self.compiled_patterns = [(re.compile(pattern, re.DOTALL), name) 
                                 for pattern, name in self.circuit_patterns]
        
        # Results storage
        self.results = {
            'scan_summary': {
                'total_tar_files': 0,
                'valid_tar_files': 0,
                'corrupted_tar_files': 0,
                'total_papers_scanned': 0,
                'papers_with_circuits': 0,
                'total_circuit_blocks': 0,
                'average_blocks_per_paper': 0.0,
                'max_blocks_in_paper': 0,
                'papers_by_block_count': defaultdict(int),
            },
            'paper_details': [],
            'corrupted_files': [],
            'tar_file_stats': defaultdict(dict),
        }
    
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
    
    def is_valid_tar(self, tar_path: Path) -> bool:
        """Check if file is a valid tar.gz file."""
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Try to read first member
                members = tar.getmembers()
                return len(members) > 0
        except:
            return False
    
    def find_complete_circuit_blocks(self, content: str) -> List[Dict]:
        """Find complete \Qcircuit blocks in content."""
        blocks = []
        
        # Method 1: Use regex patterns
        for pattern, pattern_name in self.compiled_patterns:
            for match in pattern.finditer(content):
                block_text = match.group(1)
                if self._is_valid_circuit_block(block_text):
                    blocks.append({
                        'text': block_text,
                        'type': pattern_name,
                        'length': len(block_text),
                        'preview': block_text[:200] + '...' if len(block_text) > 200 else block_text
                    })
        
        # Method 2: Manual extraction for complex cases
        qcircuit_positions = [m.start() for m in re.finditer(r'\\Qcircuit', content)]
        
        for pos in qcircuit_positions:
            block = self._extract_complete_block(content, pos)
            if block and block not in [b['text'] for b in blocks]:
                blocks.append({
                    'text': block,
                    'type': 'extracted_block',
                    'length': len(block),
                    'preview': block[:200] + '...' if len(block) > 200 else block
                })
        
        # Remove duplicates (keep first occurrence)
        unique_blocks = []
        seen = set()
        for block in blocks:
            block_hash = hash(block['preview'])
            if block_hash not in seen:
                seen.add(block_hash)
                unique_blocks.append(block)
        
        return unique_blocks
    
    def _is_valid_circuit_block(self, text: str) -> bool:
        """Validate if text is a complete circuit block."""
        # Must contain circuit elements
        circuit_keywords = ['\\qw', '\\gate', '\\ctrl', '\\targ', '\\meter', '\\cw', '\\lstick', '\\rstick']
        if not any(keyword in text for keyword in circuit_keywords):
            return False
        
        # Must have balanced braces
        if text.count('{') != text.count('}'):
            return False
        
        return True
    
    def _extract_complete_block(self, content: str, start_pos: int) -> Optional[str]:
        """Extract complete block starting at position."""
        # Find opening brace
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            return None
        
        # Count braces
        brace_count = 1
        pos = brace_start + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count != 0:
            return None
        
        block_text = content[start_pos:pos]
        
        # Additional validation
        if not self._is_valid_circuit_block(block_text):
            return None
        
        return block_text
    
    def scan_tar_file(self, tar_path: Path) -> Dict:
        """Scan a single tar.gz file."""
        tar_results = {
            'tar_file': str(tar_path.name),
            'status': 'success',
            'papers_scanned': 0,
            'papers_with_circuits': 0,
            'total_blocks_found': 0,
            'paper_details': [],
            'error': None,
        }
        
        try:
            # Check if file exists and has content
            if not tar_path.exists():
                tar_results['status'] = 'missing'
                return tar_results
            
            file_size = tar_path.stat().st_size
            if file_size == 0:
                tar_results['status'] = 'empty'
                return tar_results
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Get .tex files
                tex_files = [m for m in tar.getmembers() 
                           if m.isfile() and m.name.lower().endswith('.tex')]
                
                for member in tex_files:
                    tar_results['papers_scanned'] += 1
                    
                    try:
                        file_obj = tar.extractfile(member)
                        if not file_obj:
                            continue
                        
                        # Read with error handling
                        try:
                            content = file_obj.read().decode('utf-8', errors='ignore')
                        except UnicodeDecodeError:
                            # Try other encodings
                            file_obj.seek(0)
                            try:
                                content = file_obj.read().decode('latin-1', errors='ignore')
                            except:
                                continue
                        
                        paper_id = self.extract_paper_id(member.name)
                        
                        # Find circuit blocks
                        blocks = self.find_complete_circuit_blocks(content)
                        
                        if blocks:
                            tar_results['papers_with_circuits'] += 1
                            tar_results['total_blocks_found'] += len(blocks)
                            
                            paper_detail = {
                                'paper_id': paper_id,
                                'tex_file': member.name,
                                'circuit_blocks_count': len(blocks),
                                'blocks_found': len(blocks),
                                'block_examples': [],
                            }
                            
                            # Store example blocks
                            for i, block in enumerate(blocks[:2]):
                                paper_detail['block_examples'].append({
                                    'index': i,
                                    'preview': block['preview'],
                                    'length': block['length'],
                                })
                            
                            tar_results['paper_details'].append(paper_detail)
                            
                    except Exception as e:
                        # Skip problematic files
                        continue
                        
        except tarfile.ReadError as e:
            tar_results['status'] = 'corrupted'
            tar_results['error'] = str(e)
        except Exception as e:
            tar_results['status'] = 'error'
            tar_results['error'] = str(e)
        
        return tar_results
    
    def scan_all_tar_files(self, max_workers: int = 2, batch_size: int = 100) -> Dict:
        """Scan all tar.gz files with batching to manage memory."""
        # Find all tar.gz files
        tar_files = list(self.tar_folder.glob('*.tar.gz')) + list(self.tar_folder.glob('*.tgz'))
        total_files = len(tar_files)
        self.results['scan_summary']['total_tar_files'] = total_files
        
        print(f"Found {total_files} tar.gz files")
        print("Starting scan for complete \\\\Qcircuit blocks...")
        
        # Process in batches
        valid_files = 0
        corrupted_files = 0
        
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = tar_files[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            print(f"Files {batch_start + 1} to {batch_end}")
            
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.scan_tar_file, tar_file): tar_file 
                          for tar_file in batch_files}
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                 total=len(batch_files), desc="Scanning batch"):
                    tar_file = futures[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        # Track file status
                        if result['status'] == 'success':
                            valid_files += 1
                        else:
                            corrupted_files += 1
                            self.results['corrupted_files'].append({
                                'file': tar_file.name,
                                'status': result['status'],
                                'error': result.get('error'),
                            })
                        
                        # Store stats
                        if result['status'] == 'success':
                            self.results['tar_file_stats'][tar_file.name] = {
                                'papers_scanned': result['papers_scanned'],
                                'papers_with_circuits': result['papers_with_circuits'],
                                'total_blocks': result['total_blocks_found'],
                            }
                        
                    except Exception as e:
                        print(f"Unexpected error with {tar_file.name}: {e}")
            
            # Aggregate batch results
            self._aggregate_batch_results(batch_results)
            
            # Save intermediate results
            self._save_intermediate_results(batch_start)
        
        # Update summary
        self.results['scan_summary'].update({
            'valid_tar_files': valid_files,
            'corrupted_tar_files': corrupted_files,
        })
        
        return self.results
    
    def _aggregate_batch_results(self, batch_results: List[Dict]):
        """Aggregate results from a batch."""
        for result in batch_results:
            if result['status'] != 'success':
                continue
            
            # Update summary
            self.results['scan_summary']['total_papers_scanned'] += result['papers_scanned']
            self.results['scan_summary']['papers_with_circuits'] += result['papers_with_circuits']
            self.results['scan_summary']['total_circuit_blocks'] += result['total_blocks_found']
            
            # Store paper details (limit total)
            if len(self.results['paper_details']) < 10000:  # Keep memory manageable
                self.results['paper_details'].extend(result['paper_details'])
            
            # Update block count distribution
            for paper in result['paper_details']:
                count = paper['circuit_blocks_count']
                self.results['scan_summary']['papers_by_block_count'][count] += 1
    
    def _save_intermediate_results(self, batch_num: int):
        """Save intermediate results periodically."""
        intermediate_file = f"qcircuit_scan_batch_{batch_num}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Print progress
        summary = self.results['scan_summary']
        print(f"\nProgress after {batch_num} files:")
        print(f"  Papers scanned: {summary['total_papers_scanned']}")
        print(f"  Papers with circuits: {summary['papers_with_circuits']}")
        print(f"  Total circuit blocks: {summary['total_circuit_blocks']}")
        if summary['papers_with_circuits'] > 0:
            print(f"  Avg blocks per paper with circuits: {summary['total_circuit_blocks'] / summary['papers_with_circuits']:.1f}")
    
    def save_results(self):
        """Save final results."""
        # Calculate final averages
        summary = self.results['scan_summary']
        if summary['papers_with_circuits'] > 0:
            summary['average_blocks_per_paper'] = summary['total_circuit_blocks'] / summary['papers_with_circuits']
        else:
            summary['average_blocks_per_paper'] = 0
        
        if summary['papers_by_block_count']:
            summary['max_blocks_in_paper'] = max(summary['papers_by_block_count'].keys())
        
        # Save to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nFinal results saved to {self.output_file}")
    
    def print_summary(self):
        """Print summary of findings."""
        summary = self.results['scan_summary']
        
        print("\n" + "="*60)
        print("QCircuit Block Scanner - Final Report")
        print("="*60)
        print(f"Total tar.gz files: {summary['total_tar_files']}")
        print(f"Valid tar.gz files: {summary['valid_tar_files']}")
        print(f"Corrupted/invalid files: {summary['corrupted_tar_files']}")
        print(f"Papers scanned: {summary['total_papers_scanned']:,}")
        print(f"Papers with \\\\Qcircuit blocks: {summary['papers_with_circuits']:,}")
        
        if summary['total_papers_scanned'] > 0:
            percentage = (summary['papers_with_circuits'] / summary['total_papers_scanned']) * 100
            print(f"Percentage with circuits: {percentage:.1f}%")
        
        print(f"Total circuit blocks found: {summary['total_circuit_blocks']:,}")
        
        if summary['papers_with_circuits'] > 0:
            print(f"Average blocks per paper (with circuits): {summary['average_blocks_per_paper']:.1f}")
        
        print(f"\nPapers by block count:")
        sorted_counts = sorted(summary['papers_by_block_count'].items())
        for count, num_papers in sorted_counts[:15]:  # Show top 15
            print(f"  {count:2d} block(s): {num_papers:5d} papers")
        
        if len(sorted_counts) > 15:
            print(f"  ... and {len(sorted_counts) - 15} more block counts")
        
        # Show some examples
        if self.results['paper_details']:
            print(f"\nExample papers with circuits:")
            for i, paper in enumerate(self.results['paper_details'][:3], 1):
                print(f"\nExample {i}:")
                print(f"  Paper ID: {paper['paper_id']}")
                print(f"  Circuit blocks: {paper['circuit_blocks_count']}")
                if paper.get('block_examples'):
                    print(f"  Example circuit preview:")
                    print(f"    {paper['block_examples'][0]['preview'][:150]}...")

# Quick test function
def quick_test():
    """Quick test with a small subset."""
    print("Running quick test...")
    
    scanner = QCircuitBlockCounter("arxiv_cache", "test_results.json")
    
    # Test with just 10 files
    tar_files = list(scanner.tar_folder.glob('*.tar.gz'))[:10]
    
    print(f"Testing with {len(tar_files)} files...")
    
    for tar_file in tar_files:
        print(f"\nTesting {tar_file.name}:")
        result = scanner.scan_tar_file(tar_file)
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Papers: {result['papers_scanned']}")
            print(f"  Papers with circuits: {result['papers_with_circuits']}")
            print(f"  Total blocks: {result['total_blocks_found']}")

# Main execution
if __name__ == "__main__":
    # Configuration
    TAR_FOLDER = "arxiv_cache"  # Your tar file folder
    OUTPUT_FILE = "qcircuit_block_counts_final.json"
    
    # Run scanner
    scanner = QCircuitBlockCounter(TAR_FOLDER, OUTPUT_FILE)
    
    # Quick test first (optional)
    # quick_test()
    
    # Full scan with error handling
    try:
        results = scanner.scan_all_tar_files(max_workers=4, batch_size=50)
        scanner.save_results()
        scanner.print_summary()
    except KeyboardInterrupt:
        print("\nScan interrupted by user. Saving current results...")
        scanner.save_results()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving partial results...")
        scanner.save_results()