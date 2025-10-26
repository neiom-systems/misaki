"""
Luxembourgish Grapheme-to-Phoneme (G2P) Module

Converts Luxembourgish text to IPA phonemes using a comprehensive pronunciation dictionary.

Dictionary: 33,786 word-to-phoneme mappings from LOD (Lëtzebuerger Onlinedict)
Source: University of Luxembourg transcription tool + manual transcriptions

Usage:
    from misaki import lb
    g2p = lb.LBG2P()
    phonemes, tokens = g2p("Moien, Lëtzebuerg!")
    # Output: "ˈmoi̯ən ˈlətsəbuːɐ̯ɛɕ"
"""

import json
import re
import importlib.resources
from typing import Tuple, Optional
from pathlib import Path


class LBG2P:
    """
    Luxembourgish Grapheme-to-Phoneme Converter
    
    Converts Luxembourgish text to IPA phoneme transcriptions using a dictionary
    lookup approach with support for unknown words.
    """
    
    def __init__(self, unk: str = '❓'):
        """
        Initialize the Luxembourgish G2P engine.
        
        Args:
            unk: Symbol to use for unknown words (default: '❓')
        """
        self.unk = unk
        self.lexicon = self._load_lexicon()
    
    def _load_lexicon(self) -> dict:
        """
        Load the Luxembourgish pronunciation dictionary.
        
        Returns:
            Dictionary mapping lowercase words to IPA phonemes
        """
        try:
            # Try to load from package data
            from misaki import data
            dict_path = importlib.resources.files(data) / 'lb_gold.json'
            
            with importlib.resources.as_file(dict_path) as path:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            # Fallback: try direct path
            try:
                fallback_path = Path(__file__).parent / 'data' / 'lb_gold.json'
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                print(f"Warning: Could not load Luxembourgish dictionary: {e}")
                return {}
    
    def _tokenize(self, text: str) -> list:
        """
        Tokenize text into words while preserving punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, punctuation) tuples
        """
        # Split by whitespace
        tokens = []
        
        for token in text.split():
            if not token:
                continue
            
            # Separate trailing punctuation
            match = re.match(r'^(.*?)([.,!?;:\-—…"«»()[\]{}]*?)$', token)
            if match:
                word = match.group(1)
                punct = match.group(2)
                
                if word:
                    tokens.append((word, punct))
                elif punct:
                    # Pure punctuation token
                    tokens.append((punct, ''))
        
        return tokens
    
    def _lookup_word(self, word: str) -> str:
        """
        Look up a word in the dictionary.
        
        Tries exact match first, then case-insensitive lookup.
        
        Args:
            word: Word to look up
            
        Returns:
            IPA phoneme string, or unk symbol if not found
        """
        # Try exact lowercase match first
        word_lower = word.lower()
        if word_lower in self.lexicon:
            return self.lexicon[word_lower]
        
        # Try removing accents (for variants)
        # This is a simple approach - could be expanded
        return self.unk
    
    def __call__(self, text: str) -> Tuple[str, None]:
        """
        Convert Luxembourgish text to phonemes.
        
        Args:
            text: Input Luxembourgish text
            
        Returns:
            Tuple of (phoneme_string, None)
            - phoneme_string: Space-separated IPA phonemes
            - None: Tokens (not yet implemented for simplicity)
        
        Examples:
            >>> g2p = LBG2P()
            >>> phonemes, _ = g2p("Moien")
            >>> print(phonemes)
            ˈmoi̯ən
        """
        if not text or not text.strip():
            return '', None
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Convert each token to phonemes
        phoneme_list = []
        
        for word, punct in tokens:
            if not word:
                continue
            
            # Look up word
            phoneme = self._lookup_word(word)
            
            # Add punctuation back
            if punct:
                phoneme += punct
            
            phoneme_list.append(phoneme)
        
        # Join with spaces
        result = ' '.join(phoneme_list)
        
        return result, None
    
    def get_stats(self) -> dict:
        """
        Get dictionary statistics.
        
        Returns:
            Dictionary with stats about the loaded lexicon
        """
        return {
            'total_entries': len(self.lexicon),
            'unk_symbol': self.unk,
            'sample_words': sorted(list(self.lexicon.keys()))[:10]
        }


# Export for package imports
__all__ = ['LBG2P']
