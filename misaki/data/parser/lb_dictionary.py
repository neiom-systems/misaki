#!/usr/bin/env python3
"""
Luxembourgish Dictionary Parser (Parallelized with Batching)

Extracts word-to-IPA phoneme mappings from De Lëtzebuerger Dictionnaire (LOD.pdf)

Pattern: WORD [IPA_PHONEME] rest_of_entry
Example: Aangel [ˈaːŋəl] Femininum (Pluriel...)
"""

import re
import json
import pdfplumber
from pathlib import Path
from typing import Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Module-level function for multiprocessing
def extract_page_batch(args: Tuple[int, int, str]) -> Tuple[int, int, Dict[str, str]]:
    """
    Extract word-phoneme pairs from a batch of pages.
    
    Args:
        args: Tuple of (start_page, end_page, pdf_path)
    
    Returns:
        Tuple of (start_page, end_page, dict_of_entries)
    """
    start_page, end_page, pdf_path = args
    
    entries = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(start_page, end_page):
                if page_num >= len(pdf.pages):
                    break
                
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Look for pattern: WORD [IPA]
                    match = re.match(r'^(\w[\w\s\-\']*?)\s*\[([^\]]+)\]', line.strip())
                    
                    if match:
                        word = match.group(1).strip().lower()
                        ipa = match.group(2).strip()
                        
                        # Filter out unwanted entries
                        if len(word) > 1 and not word.startswith('i ') and not word.startswith('beispill'):
                            entries[word] = ipa
    
    except Exception as e:
        print(f"Error processing pages {start_page}-{end_page}: {e}")
    
    return start_page, end_page, entries


class LBDictionaryParser:
    def __init__(self, pdf_path: str, num_workers: int = None, batch_size: int = 100):
        self.pdf_path = pdf_path
        self.entries: Dict[str, str] = {}
        # Use number of CPU cores if not specified
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.batch_size = batch_size  # Pages per worker
    
    def parse_pdf_parallel(self) -> Dict[str, str]:
        """
        Parse the entire PDF in parallel using process pool with batching.
        Each worker processes a batch of pages to reduce PDF open/close overhead.
        
        Returns: Dictionary of {word: ipa_phoneme}
        """
        print(f"Opening PDF: {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
        
        print(f"Using {self.num_workers} workers, batch size: {self.batch_size} pages")
        print(f"Estimated batches: {(total_pages + self.batch_size - 1) // self.batch_size}\n")
        
        # Create batches: (start_page, end_page, pdf_path)
        batches = []
        for start in range(0, total_pages, self.batch_size):
            end = min(start + self.batch_size, total_pages)
            batches.append((start, end, self.pdf_path))
        
        # Process batches in parallel
        processed_batches = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(extract_page_batch, batch): batch for batch in batches}
            
            for future in as_completed(futures):
                start_page, end_page, batch_entries = future.result()
                self.entries.update(batch_entries)
                
                processed_batches += 1
                progress = (processed_batches / len(batches)) * 100
                print(f"✓ Batch {processed_batches}/{len(batches)} (pages {start_page}-{end_page}) - "
                      f"{progress:.1f}% - {len(self.entries)} total entries")
        
        return self.entries
    
    def save_json(self, output_path: str) -> None:
        """Save extracted dictionary to JSON file"""
        print(f"\nSaving {len(self.entries)} entries to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        print(f"✓ Saved {len(self.entries)} word-phoneme pairs")
    
    def get_stats(self) -> None:
        """Print statistics about parsed dictionary"""
        print(f"\n=== Dictionary Statistics ===")
        print(f"Total entries: {len(self.entries)}")
        
        if self.entries:
            sample_words = sorted(list(self.entries.keys()))[:15]
            print(f"\nFirst 15 entries (alphabetical):")
            for word in sample_words:
                print(f"  {word} → {self.entries[word]}")


def main():
    """Main function to parse LOD.pdf and generate lb_gold.json"""
    
    # Paths
    pdf_path = '/Users/vivienhenz/misaki/LOD.pdf'
    output_path = '/Users/vivienhenz/misaki/misaki/data/lb_gold.json'
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        return
    
    # Parse PDF with parallel processing (batch size = 100 pages per worker)
    parser = LBDictionaryParser(pdf_path, batch_size=100)
    entries = parser.parse_pdf_parallel()
    
    # Print statistics
    parser.get_stats()
    
    # Save to JSON
    parser.save_json(output_path)
    
    print(f"\n✓ Dictionary successfully created at {output_path}")


if __name__ == '__main__':
    main()
