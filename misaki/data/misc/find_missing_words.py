#!/usr/bin/env python3
"""
Find Missing Words Script

Compares words in the Luxembourgish TTS corpus CSV with the pronunciation dictionary.
Identifies words that are in the corpus but not in the dictionary.

Output: missing.json - list of missing words that need phoneme mappings
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

def extract_words_from_csv(csv_path: str) -> set:
    """
    Extract all unique words from the Luxembourgish corpus CSV.
    
    CSV format: ID|text (pipe-separated, not comma-separated)
    Only extracts from the text column (second column).
    """
    words = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip header if present
            if '|' not in line:
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            # Get second column (text)
            text = parts[1].strip()
            
            if not text:
                continue
            
            # Extract individual words (split by spaces and punctuation)
            # Keep Luxembourgish characters (ä, ë, ö, ü, etc.)
            word_list = re.findall(r"[a-zäëöüïâêôûœàèéìòùA-Z0-9\-']+", text)
            words.update(w.lower() for w in word_list if len(w) > 1)
    
    return words

def load_dictionary(json_path: str) -> set:
    """Load all word keys from the pronunciation dictionary."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return set(data.keys())

def find_missing_words(corpus_words: set, dictionary_words: set) -> set:
    """Find words in corpus that are not in dictionary."""
    return corpus_words - dictionary_words

def save_missing_words(missing_words: set, output_path: str) -> None:
    """Save missing words to JSON file with metadata."""
    output_data = {
        "total_missing": len(missing_words),
        "missing_words": sorted(list(missing_words))
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def main():
    """Main function."""
    
    # Paths
    csv_path = '/Users/vivienhenz/misaki/misaki/luxembourgish_male_corpus/artifacts/luxembourgish_male_tts/LOD-male.csv'
    dictionary_path = '/Users/vivienhenz/misaki/misaki/data/lb_gold.json'
    output_path = '/Users/vivienhenz/misaki/misaki/data/misc/missing.json'
    
    # Check if files exist
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    if not Path(dictionary_path).exists():
        print(f"Error: Dictionary file not found at {dictionary_path}")
        return
    
    print(f"Loading corpus from: {csv_path}")
    corpus_words = extract_words_from_csv(csv_path)
    print(f"✓ Found {len(corpus_words):,} unique words in corpus")
    
    print(f"\nLoading dictionary from: {dictionary_path}")
    dictionary_words = load_dictionary(dictionary_path)
    print(f"✓ Found {len(dictionary_words):,} words in dictionary")
    
    print(f"\nFinding missing words...")
    missing_words = find_missing_words(corpus_words, dictionary_words)
    print(f"✓ Found {len(missing_words):,} missing words")
    
    # Coverage statistics
    coverage = (len(corpus_words) - len(missing_words)) / len(corpus_words) * 100
    print(f"\nDictionary coverage: {coverage:.1f}%")
    
    # Show some examples
    if missing_words:
        print(f"\nSample of missing words (first 20):")
        for word in sorted(list(missing_words))[:20]:
            print(f"  - {word}")
    
    # Save to file
    print(f"\nSaving missing words to: {output_path}")
    save_missing_words(missing_words, output_path)
    print(f"✓ Saved {len(missing_words):,} missing words")

if __name__ == '__main__':
    main()
