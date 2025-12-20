"""
Extract and render quantum circuits from arXiv tar.gz files.
One script that finds \Qcircuit blocks and renders them.
"""

import re
import tarfile
import json
import tempfile
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from pdflatex import PDFLaTeX

def extract_paper_id(filename: str) -> str:
    """Extract arXiv paper ID from filename."""
    match = re.search(r'(\d{4}\.\d{4,5})', filename)
    return match.group(1) if match else Path(filename).stem

def extract_complete_block(content: str, start_pos: int) -> str:
    """Extract complete \Qcircuit{...} block."""
    # Find opening brace
    brace_start = content.find('{', start_pos)
    if brace_start == -1:
        return ""
    
    # Count braces to find matching closing brace
    brace_count = 1
    pos = brace_start + 1
    
    while pos < len(content) and brace_count > 0:
        if content[pos] == '{':
            brace_count += 1
        elif content[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        return ""
    
    return content[start_pos:pos]

def find_label_and_caption(content: str, circuit_pos: int) -> tuple:
    """Find label and caption near circuit position."""
    # Search backwards for label
    search_start = max(0, circuit_pos - 1000)
    search_text = content[search_start:circuit_pos]
    
    # Find the last \label before circuit
    label_match = re.search(r'\\label\{([^}]+)\}(?!.*\\label\{)', search_text)
    label = label_match.group(1) if label_match else None
    
    # Find the last \caption before circuit
    caption_match = re.search(r'\\caption\{([^}]+)\}(?!.*\\caption\{)', search_text)
    caption = caption_match.group(1) if caption_match else None
    
    return label, caption

def extract_circuits_from_tar(tar_path: Path) -> List[Dict]:
    """Extract all circuits from a single tar.gz file."""
    circuits = []
    
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
                    
                    content = file_obj.read().decode('utf-8', errors='ignore')
                    paper_id = extract_paper_id(member.name)
                    
                    # Find all \Qcircuit occurrences
                    for match in re.finditer(r'\\Qcircuit', content):
                        circuit_text = extract_complete_block(content, match.start())
                        if not circuit_text or len(circuit_text) < 50:
                            continue
                        
                        # Find label and caption
                        label, caption = find_label_and_caption(content, match.start())
                        
                        circuits.append({
                            'paper_id': paper_id,
                            'tex_file': member.name,
                            'circuit_text': circuit_text,
                            'label': label,
                            'caption': caption,
                        })
                        
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"Error with {tar_path.name}: {e}")
    
    return circuits

def render_circuit(circuit_data: Dict, output_dir: Path, index: int) -> Dict:
    """Render a single circuit using pdflatex."""
    circuit_text = circuit_data['circuit_text']
    label = circuit_data.get('label', '')
    
    # Create circuit ID
    if label:
        # Clean label for filename
        clean_label = re.sub(r'[^\w\-]', '_', label)
        circuit_id = f"{circuit_data['paper_id']}_{clean_label}_{index}"
    else:
        circuit_id = f"{circuit_data['paper_id']}_circuit_{index}"
    
    # Create LaTeX document (EXACTLY like your example)
    tex_code = f"""
\\documentclass{{standalone}}
\\usepackage{{amsmath}}
\\usepackage{{braket}}
\\usepackage{{qcircuit}}

\\begin{{document}}

\\begin{{equation*}}
{circuit_text}
\\end{{equation*}}

\\end{{document}}
"""
    
    # Write to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        tex_file = temp_dir / "circuit.tex"
        tex_file.write_text(tex_code)
        
        try:
            # Compile with pdflatex (using the library like your example)
            pdfl = PDFLaTeX.from_texfile(str(tex_file))
            pdf_bytes, log, proc = pdfl.create_pdf(
                keep_pdf_file=False,
                keep_log_file=False
            )
            
            if proc.returncode == 0:
                # Save PDF
                pdf_path = output_dir / f"{circuit_id}.pdf"
                pdf_path.write_bytes(pdf_bytes)
                
                # Convert PDF to PNG
                png_path = output_dir / f"{circuit_id}.png"
                try:
                    import subprocess
                    subprocess.run([
                        "pdftoppm", "-png", "-singlefile",
                        str(pdf_path), str(png_path.with_suffix(''))
                    ], capture_output=True)
                except:
                    pass  # PNG conversion is optional
                
                return {
                    **circuit_data,
                    'circuit_id': circuit_id,
                    'pdf_path': str(pdf_path),
                    'png_path': str(png_path) if png_path.exists() else None,
                    'render_success': True,
                    'error': None,
                }
            else:
                return {
                    **circuit_data,
                    'circuit_id': circuit_id,
                    'render_success': False,
                    'error': f"pdflatex failed with code {proc.returncode}",
                }
                
        except Exception as e:
            return {
                **circuit_data,
                'circuit_id': circuit_id,
                'render_success': False,
                'error': str(e),
            }

def main():
    """Main function: Extract and render circuits."""
    # Configuration
    TAR_FOLDER = Path("arxiv_cache")
    OUTPUT_DIR = Path("rendered_circuits")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find tar files
    tar_files = list(TAR_FOLDER.glob('*.tar.gz'))
    print(f"Found {len(tar_files)} tar.gz files")
    
    # Limit to 2 files for testing
    tar_files = tar_files[:2]
    
    all_circuits = []
    all_rendered = []
    
    # Step 1: Extract circuits
    print("\n=== Step 1: Extracting circuits ===")
    for tar_file in tqdm(tar_files, desc="Processing tar files"):
        circuits = extract_circuits_from_tar(tar_file)
        all_circuits.extend(circuits)
        
        if circuits:
            paper_id = circuits[0]['paper_id'] if circuits else "unknown"
            print(f"  {tar_file.name}: Found {len(circuits)} circuits in {paper_id}")
    
    print(f"\nTotal circuits found: {len(all_circuits)}")
    
    # Save extracted circuits
    with open(OUTPUT_DIR / "extracted_circuits.json", 'w', encoding='utf-8') as f:
        json.dump(all_circuits, f, indent=2, ensure_ascii=False)
    print(f"Saved circuits to {OUTPUT_DIR / 'extracted_circuits.json'}")
    
    # Step 2: Render circuits
    print("\n=== Step 2: Rendering circuits ===")
    
    # Limit to first 3 circuits for testing
    circuits_to_render = all_circuits[:3]
    print(f"Rendering {len(circuits_to_render)} circuits...")
    
    for i, circuit in enumerate(tqdm(circuits_to_render, desc="Rendering")):
        rendered = render_circuit(circuit, OUTPUT_DIR, i)
        all_rendered.append(rendered)
        
        if rendered['render_success']:
            label_info = f" (label: {rendered['label']})" if rendered['label'] else ""
            print(f"[OK] {rendered['circuit_id']}{label_info}")
        else:
            print(f"[FAIL] {rendered['circuit_id']}: {rendered['error']}")
    
    # Save rendering results
    with open(OUTPUT_DIR / "rendering_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_rendered, f, indent=2, ensure_ascii=False)
    
    # Print summary
    success_count = sum(1 for r in all_rendered if r['render_success'])
    print(f"\n=== Summary ===")
    print(f"Circuits extracted: {len(all_circuits)}")
    print(f"Circuits rendered: {success_count}/{len(circuits_to_render)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Show some examples
    if all_rendered:
        print("\nFirst 3 rendered circuits:")
        for i, rendered in enumerate(all_rendered[:3]):
            status = "OK" if rendered['render_success'] else "FAIL"
            label = rendered.get('label', 'No label')
            print(f"{i+1}. {status} {rendered['circuit_id']} - {label}")
            if rendered.get('caption'):
                print(f"   Caption: {rendered['caption'][:80]}...")

if __name__ == "__main__":
    main()