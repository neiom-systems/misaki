import importlib.resources
import json
import numpy as np
import re
import spacy
import unicodedata
from functools import lru_cache

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from . import data
from .token import MToken


def merge_tokens(tokens: List[MToken], unk: Optional[str] = None) -> MToken:
    stress = {tk._.stress for tk in tokens if tk._.stress is not None}
    currency = {tk._.currency for tk in tokens if tk._.currency is not None}
    rating = {tk._.rating for tk in tokens}
    if unk is None:
        phonemes = None
    else:
        phonemes = ''
        for tk in tokens:
            if tk._.prespace and phonemes and not phonemes[-1].isspace() and tk.phonemes:
                phonemes += ' '
            phonemes += unk if tk.phonemes is None else tk.phonemes
    return MToken(
        text=''.join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text,
        tag=max(tokens, key=lambda tk: sum(1 if c == c.lower() else 2 for c in tk.text)).tag,
        whitespace=tokens[-1].whitespace,
        phonemes=phonemes,
        start_ts=tokens[0].start_ts,
        end_ts=tokens[-1].end_ts,
        _=MToken.Underscore(
            is_head=tokens[0]._.is_head,
            alias=None,
            stress=list(stress)[0] if len(stress) == 1 else None,
            currency=max(currency) if currency else None,
            num_flags=''.join(sorted({c for tk in tokens for c in tk._.num_flags})),
            prespace=tokens[0]._.prespace,
            rating=None if None in rating else min(rating),
        ),
    )


LB_DIPHTHONGS = frozenset('AEIOUYäëéöüyœʀ')


def stress_weight(ps: Optional[str]) -> int:
    return sum(2 if c in LB_DIPHTHONGS else 1 for c in ps) if ps else 0


@dataclass
class TokenContext:
    future_vowel: Optional[bool] = None
    future_zu: bool = False


try:
    import regex
except ImportError as exc:
    raise ImportError("The 'regex' module is required for misaki.lb.") from exc


def make_subtokenize_once():
    subtoken_pattern = regex.compile(
        r"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?"
        r"\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$"
    )
    return lambda word: regex.findall(subtoken_pattern, word)


subtokenize = make_subtokenize_once()
del make_subtokenize_once

LINK_REGEX = re.compile(r'\[([^\]]+)\]\(([^\)]*)\)')

SUBTOKEN_JUNKS = frozenset("',-._‘’/")
PUNCTS = frozenset(';:,.!?—…"“”')
NON_QUOTE_PUNCTS = frozenset(p for p in PUNCTS if p not in '"“”')

PUNCT_TAGS = frozenset([".", ",", "-LRB-", "-RRB-", "``", '""', "''", ":", "$", "#", 'NFP'])
PUNCT_TAG_PHONEMES = {'-LRB-': '(', '-RRB-': ')', '``': chr(8220), '""': chr(8221), "''": chr(8221)}

PRIMARY_STRESS = 'ˈ'
SECONDARY_STRESS = 'ˌ'
STRESSES = PRIMARY_STRESS + SECONDARY_STRESS
LB_VOWELS = frozenset('AEIOUYaeiouyäëéöüœɐɑɔəɛɪʊɜœ̃ɛ̃ãôâîôûŷ')


def apply_stress(ps: Optional[str], stress: Optional[float]) -> Optional[str]:
    if ps is None:
        return None

    def restress(seq: str) -> str:
        indexed = list(enumerate(seq))
        stresses = {i: next(j for j, v in indexed[i:] if v in LB_VOWELS) for i, c in indexed if c in STRESSES}
        for i, j in stresses.items():
            _, c = indexed[i]
            indexed[i] = (j - 0.5, c)
        return ''.join(p for _, p in sorted(indexed))

    if stress is None:
        return ps
    if stress < -1:
        return ps.replace(PRIMARY_STRESS, '').replace(SECONDARY_STRESS, '')
    if stress == -1 or (stress in (0, -0.5) and PRIMARY_STRESS in ps):
        return ps.replace(SECONDARY_STRESS, '').replace(PRIMARY_STRESS, SECONDARY_STRESS)
    if stress in (0, 0.5, 1) and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in LB_VOWELS):
            return ps
        return restress(SECONDARY_STRESS + ps)
    if stress >= 1 and PRIMARY_STRESS not in ps and SECONDARY_STRESS in ps:
        return ps.replace(SECONDARY_STRESS, PRIMARY_STRESS)
    if stress > 1 and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in LB_VOWELS):
            return ps
        return restress(PRIMARY_STRESS + ps)
    return ps


def is_digit(text: str) -> bool:
    return bool(re.fullmatch(r'[0-9]+', text))


LETTER_PHONEMES = {
    'A': 'aː',
    'B': 'beː',
    'C': 'tseː',
    'D': 'deː',
    'E': 'eː',
    'F': 'ɛf',
    'G': 'ɡeː',
    'H': 'haː',
    'I': 'iː',
    'J': 'jɔt',
    'K': 'kaː',
    'L': 'ɛl',
    'M': 'ɛm',
    'N': 'ɛn',
    'O': 'oː',
    'P': 'peː',
    'Q': 'kuː',
    'R': 'ɛʀ',
    'S': 'ɛs',
    'T': 'teː',
    'U': 'uː',
    'V': 'faʊ',
    'W': 'veː',
    'X': 'ɪks',
    'Y': 'ʏpsilɔn',
    'Z': 'tsɛt',
}


DIGIT_WORDS = {
    0: 'null',
    1: 'een',
    2: 'zwee',
    3: 'dräi',
    4: 'véier',
    5: 'fënnef',
    6: 'sechs',
    7: 'siwen',
    8: 'aacht',
    9: 'néng',
}

TEEN_WORDS = {
    10: 'zéng',
    11: 'eelef',
    12: 'zwielef',
    13: 'dräizéng',
    14: 'véierzéng',
    15: 'fofzéng',
    16: 'sechzéng',
    17: 'siwwenzéng',
    18: 'uechtzéng',
    19: 'nonzéng',
}

TENS_WORDS = {
    20: 'zwanzeg',
    30: 'drësseg',
    40: 'véierzeg',
    50: 'fofzeg',
    60: 'sechzeg',
    70: 'siwwenzeg',
    80: 'achtzeg',
    90: 'nonzeg',
}

SCALE_WORDS = [
    (1_000_000_000, 'milliard'),
    (1_000_000, 'millioun'),
    (1_000, 'dausend'),
    (100, 'honnert'),
]

ORDINAL_WORDS = {
    1: 'éischt',
    2: 'zweet',
    3: 'drëtt',
    4: 'véiert',
    5: 'fënneft',
    6: 'sechste',
    7: 'siwent',
    8: 'aachten',
    9: 'néngte',
    10: 'zéngte',
}

CURRENCIES = {
    '€': ('euro', 'cent'),
    'EUR': ('euro', 'cent'),
}

ADD_SYMBOLS = {'.': 'punkt', ',': 'komma', '/': 'slash'}
SYMBOLS = {'%': 'prozent', '&': 'an', '+': 'plus', '-': 'minus'}

ORDINAL_SUFFIXES = ('ten', 'ter', 'te', 't', 'sten')

DIMINUTIVE_SUFFIXES = ('chen', 'che', 'ercher', 'cher', 'schen')
COMPOUND_LINKERS = ('', 's', 'es', 'er', 'en', 'n')
IRREGULAR_INFLECTIONS = {
    'kanner': 'kand',
    'kandern': 'kand',
    'männer': 'mann',
    'leit': 'leit',
    'haiser': 'haus',
    'häiser': 'haus',
    'haisercher': 'haus',
    'héiser': 'hees',
    'geet': 'goen',
    'geetn': 'goen',
    'gaang': 'goen',
    'gaangen': 'goen',
    'gesäit': 'gesinn',
    'ass': 'sinn',
    'sidd': 'sinn',
    'hat': 'hunn',
    'hatten': 'hunn',
    'kënnt': 'kommen',
}
UMLAUT_REVERSALS = (
    ('ä', 'a'),
    ('ë', 'e'),
    ('ö', 'o'),
    ('ü', 'u'),
)


def lowercase_first(word: str) -> str:
    return word[:1].lower() + word[1:]


class Lexicon:
    def __init__(self):
        self.cap_stresses = (0.5, 1.5)
        with importlib.resources.as_file(importlib.resources.files(data) / 'lb_gold.json') as path:
            with open(path, 'r', encoding='utf-8') as stream:
                self.golds: Dict[str, str] = json.load(stream)
        self._known_cache: Dict[str, bool] = {}

    @lru_cache(maxsize=4096)
    def has_entry(self, word: str) -> bool:
        if word in self.golds or word in SYMBOLS or word in ADD_SYMBOLS:
            return True
        if word in LETTER_PHONEMES or word.upper() in LETTER_PHONEMES:
            return True
        return False

    def is_known(self, word: str) -> bool:
        if self.has_entry(word):
            return True
        lower = word.lower()
        if lower != word and self.has_entry(lower):
            return True
        if len(word) == 1:
            return self.has_entry(word.upper())
        for variant in self.generate_variants(lower):
            if self.has_entry(variant):
                return True
        parts = self.split_compound(lower)
        if parts:
            return all(self.has_entry(part) or any(self.has_entry(v) for v in self.generate_variants(part)) for part in parts)
        return False

    def lookup(self, word: str, stress: Optional[float]) -> Tuple[Optional[str], Optional[int]]:
        if word in LETTER_PHONEMES:
            return apply_stress(LETTER_PHONEMES[word], stress), 3
        phon = self.golds.get(word)
        if phon is None and word.lower() != word:
            phon = self.golds.get(lowercase_first(word))
        if phon is None and word.upper() == word:
            phon = self.get_acronym(word)
        if phon is None:
            return None, None
        return apply_stress(phon, stress), 4

    def get_acronym(self, word: str) -> Optional[str]:
        phones = []
        for char in word:
            if char in LETTER_PHONEMES:
                phones.append(LETTER_PHONEMES[char])
            else:
                return None
        phoneme = ''.join(phones)
        return apply_stress(phoneme, 0)

    @staticmethod
    def _strip_apostrophe(word: str) -> Tuple[str, Optional[str]]:
        if not word.endswith("'"):
            return word, None
        stripped = word[:-1]
        if stripped.lower() in ('d', 'n', 't'):
            mapping = {'d': 'de', 'n': 'en', 't': 'et'}
            base = mapping.get(stripped.lower())
            if base and stripped.isupper():
                base = base.upper()
            elif base and stripped[0].isupper():
                base = base.capitalize()
            return base, stripped
        return stripped, stripped

    def get_special_case(
        self,
        word: str,
        stress: Optional[float],
        ctx: TokenContext,
    ) -> Tuple[Optional[str], Optional[int]]:
        lowered = word.lower()
        if word in ADD_SYMBOLS:
            return self.lookup(ADD_SYMBOLS[word], stress)
        if word in SYMBOLS:
            return self.lookup(SYMBOLS[word], stress)
        if lowered in ('d\'', 'd’'):
            return self.lookup('de', stress)
        if lowered in ('n\'', 'n’'):
            return self.lookup('en', stress)
        if lowered == 't\'':  # contraction of "et"
            return self.lookup('et', stress)
        if lowered == 'd':
            return self.lookup('de', stress)
        if lowered == 'n':
            return self.lookup('en', stress)
        if lowered == 't':
            return self.lookup('et', stress)
        if lowered in ('am', 'an') and ctx.future_vowel is False:
            return self.lookup(lowered, stress)
        common_map = {
            'ze': 'zu',
            'zu': 'zu',
            'mat': 'mat',
            'op': 'op',
            'vum': 'vun',
            'vumm': 'vun',
            'vunn': 'vun',
            'vumme': 'vun',
            'fir': 'fir',
            'vir': 'vir',
            'ouni': 'ouni',
            'beim': 'bei',
        }
        if lowered in common_map:
            return self.lookup(common_map[lowered], stress)
        return None, None

    def part_known(self, part: str) -> bool:
        if self.has_entry(part):
            return True
        for variant in self.generate_variants(part):
            if self.has_entry(variant):
                return True
        return False

    def generate_variants(self, word: str) -> Iterable[str]:
        variants = set()
        override = IRREGULAR_INFLECTIONS.get(word)
        if override:
            variants.add(override)
        for suffix in DIMINUTIVE_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[:-len(suffix)]
                variants.add(stem)
                if suffix.endswith('chen'):
                    variants.add(stem + 'e')
        if word.endswith('en') and len(word) > 4:
            variants.add(word[:-2])
            variants.add(word[:-2] + 'e')
        if word.endswith('er') and len(word) > 4:
            variants.add(word[:-2])
        if word.endswith('ger') and len(word) > 5:
            variants.add(word[:-3] + 'g')
        if word.endswith('ter') and len(word) > 5:
            variants.add(word[:-3] + 't')
        if word.endswith('ewer') and len(word) > 6:
            variants.add(word[:-4] + 'ew')
        for old, new in UMLAUT_REVERSALS:
            if old in word:
                idx = word.rfind(old)
                variants.add(word[:idx] + new + word[idx + len(old):])
        if word.endswith('ësch') and len(word) > 5:
            variants.add(word[:-4])
        if word.endswith('esch') and len(word) > 5:
            variants.add(word[:-4])
        if word.endswith('elt') and len(word) > 4:
            variants.add(word[:-3])
        if word.endswith('ëlt') and len(word) > 4:
            variants.add(word[:-3])
        if word.endswith('ten') and len(word) > 4:
            variants.add(word[:-3])
        return [v for v in variants if v and v != word]

    def split_compound(self, word: str) -> Optional[List[str]]:
        lower = word.lower()

        @lru_cache(maxsize=2048)
        def helper(segment: str, depth: int = 0) -> List[List[str]]:
            sequences: List[List[str]] = []
            if self.part_known(segment):
                sequences.append([segment])
            if len(segment) < 5 or depth > 3:
                return sequences
            length = len(segment)
            for i in range(3, length - 2):
                left = segment[:i]
                right = segment[i:]
                for extend in range(0, min(3, len(right))):
                    left_candidate = left + right[:extend]
                    remainder = right[extend:]
                    if len(remainder) < 2:
                        continue
                    if not self.part_known(left_candidate):
                        continue
                    for tail in helper(remainder, depth + 1):
                        sequences.append([left_candidate] + tail)
                for linker in COMPOUND_LINKERS[1:]:
                    if right.startswith(linker) and len(right) > len(linker) + 2 and self.part_known(left):
                        trimmed = right[len(linker):]
                        for tail in helper(trimmed, depth + 1):
                            if self.part_known(linker):
                                sequences.append([left, linker] + tail)
                            else:
                                sequences.append([left] + tail)
            return sequences

        combos = helper(lower, 0)
        if not combos:
            return None
        combos.sort(key=lambda parts: (len(parts), -sum(len(part) for part in parts)))
        return combos[0]

    def _suffix_candidates(self, word: str) -> Iterable[str]:
        length = len(word)
        if length > 4 and word.endswith('en'):
            yield word[:-2]
            yield word[:-2] + 'e'
        if length > 3 and word.endswith('er'):
            yield word[:-2]
        if length > 3 and word.endswith('em'):
            yield word[:-2]
            yield word[:-2] + 'e'
        if length > 4 and word.endswith('ten'):
            yield word[:-3]
        if length > 4 and word.endswith('ter'):
            yield word[:-3]
        if length > 3 and word.endswith('te'):
            yield word[:-2]
        if length > 3 and word.endswith('ten'):
            yield word[:-3] + 't'
        if length > 3 and word.endswith('t'):
            yield word[:-1]
        if length > 3 and word.endswith('n'):
            yield word[:-1]
        if length > 3 and word.endswith('s'):
            yield word[:-1]

    @staticmethod
    def _prefix_candidates(word: str) -> Iterable[str]:
        if word.startswith('ge') and len(word) > 4:
            yield word[2:]
        if word.startswith('ver') and len(word) > 5:
            yield word[3:]
        if word.startswith('zer') and len(word) > 5:
            yield word[3:]

    def get_word(self, word: str, stress: Optional[float], ctx: TokenContext) -> Tuple[Optional[str], Optional[int]]:
        ps, rating = self.get_special_case(word, stress, ctx)
        if ps is not None:
            return ps, rating
        base, _ = self._strip_apostrophe(word)
        ps, rating = self.lookup(base, stress)
        if ps is not None:
            return ps, rating
        lowered = base.lower()
        ps, rating = self.lookup(lowered, stress)
        if ps is not None:
            return ps, rating
        for variant in self.generate_variants(lowered):
            ps, rating = self.lookup(variant, stress)
            if ps is not None:
                return ps, rating
        for candidate in self._suffix_candidates(lowered):
            ps, rating = self.lookup(candidate, stress)
            if ps is not None:
                return ps, rating
        for candidate in self._prefix_candidates(lowered):
            ps, rating = self.lookup(candidate, stress)
            if ps is not None:
                return ps, rating
        compound_parts = self.split_compound(base)
        if compound_parts:
            combined = []
            ratings = []
            for idx, part in enumerate(compound_parts):
                part_candidates = [part, *self.generate_variants(part)]
                part_ps = None
                part_rating = None
                for cand in part_candidates:
                    part_ps, part_rating = self.lookup(cand, stress if idx == 0 else None)
                    if part_ps is not None:
                        break
                if part_ps is None:
                    combined = []
                    break
                if idx > 0:
                    part_ps = apply_stress(part_ps, -0.5)
                combined.append(part_ps)
                ratings.append(4 if part_rating is None else part_rating)
            if combined:
                return ' '.join(combined), min(ratings) if ratings else 4
        return None, None

    @staticmethod
    def is_currency(text: str) -> bool:
        if '.' not in text and ',' not in text:
            return True
        separators = text.replace(',', '.').split('.')
        if len(separators) > 2:
            return False
        if len(separators) == 2 and len(separators[1]) > 2:
            return False
        return all(part.isdigit() for part in separators if part)

    def _cardinal_under_hundred(self, num: int) -> List[str]:
        if num < 10:
            return [DIGIT_WORDS[num]]
        if num < 20:
            return [TEEN_WORDS[num]]
        tens = num // 10 * 10
        unit = num % 10
        if unit == 0:
            return [TENS_WORDS[tens]]
        words = [DIGIT_WORDS[unit], 'an', TENS_WORDS[tens]]
        return words

    def cardinal_words(self, num: int) -> List[str]:
        if num == 0:
            return [DIGIT_WORDS[0]]
        words: List[str] = []
        for scale, label in SCALE_WORDS:
            if num >= scale:
                count = num // scale
                num %= scale
                words.extend(self.cardinal_words(count))
                words.append(label)
        if num >= 100:
            hundreds = num // 100
            num %= 100
            if hundreds == 1:
                words.append('honnert')
            else:
                words.extend([DIGIT_WORDS[hundreds], 'honnert'])
        if num:
            words.extend(self._cardinal_under_hundred(num))
        return words

    def ordinal_words(self, num: int) -> Optional[List[str]]:
        if num in ORDINAL_WORDS:
            return [ORDINAL_WORDS[num]]
        if num < 100:
            tens = num // 10 * 10
            unit = num % 10
            if unit and tens:
                base = self._cardinal_under_hundred(num)
                base[-1] = base[-1] + 'ten'
                return base
        return None

    def get_number(
        self,
        word: str,
        currency: Optional[str],
        is_head: bool,
        num_flags: str,
    ) -> Tuple[Optional[str], Optional[int]]:
        match = re.search(r"[A-Za-zäöüéë’']+$", word)
        suffix = match.group() if match else ''
        number_part = word[:-len(suffix)] if suffix else word
        number_part = number_part.replace("’", "'")
        is_ordinal = suffix.lower() in ORDINAL_SUFFIXES
        number_text = number_part.replace("'", '')
        number_text = number_text.replace(',', '.')
        result: List[Tuple[str, int]] = []

        def append_words(words: Iterable[str]) -> None:
            for w in words:
                lookup, rating = self.lookup(w, None)
                if lookup is None:
                    # fallback to raw letters if lookup fails
                    lookup = w
                    rating = 3
                result.append((lookup, rating or 4))

        if number_text.startswith('-'):
            minus, rating = self.lookup('minus', None)
            if minus:
                result.append((minus, rating or 4))
            number_text = number_text[1:]

        if not number_text:
            return None, None

        raw_number = number_part
        comma_decimal = ',' in raw_number
        dot_sections = [section for section in raw_number.split('.') if section]
        has_dot = '.' in raw_number
        thousand_group = (
            has_dot
            and not comma_decimal
            and len(dot_sections) > 1
            and all(len(section) == 3 and section.isdigit() for section in dot_sections[1:])
        )
        dot_decimal = has_dot and not thousand_group and not comma_decimal

        if currency and Lexicon.is_currency(number_part):
            integer_part = raw_number
            cents_part = ''
            if ',' in raw_number:
                integer_part, cents_part = raw_number.split(',', 1)
            elif raw_number.count('.') == 1:
                integer_part, cents_part = raw_number.split('.', 1)
            integer_digits = re.sub(r"[^\d]", '', integer_part)
            if integer_digits:
                append_words(self.cardinal_words(int(integer_digits)))
            unit, subunit = CURRENCIES.get(currency, ('euro', 'cent'))
            append_words([unit])
            cents_digits = re.sub(r"[^\d]", '', cents_part)[:2]
            if cents_digits:
                cents_value = int(cents_digits)
                if cents_value:
                    append_words(self.cardinal_words(cents_value))
                    append_words([subunit])
        elif comma_decimal or dot_decimal:
            sep = ',' if comma_decimal else '.'
            integer_part, decimal_part = raw_number.split(sep, 1)
            integer_digits = re.sub(r"[^\d]", '', integer_part)
            if integer_digits:
                append_words(self.cardinal_words(int(integer_digits)))
            point, rating = self.lookup('komma', None)
            if point:
                result.append((point, rating or 4))
            for digit in re.sub(r"[^\d]", '', decimal_part):
                append_words([DIGIT_WORDS[int(digit)]])
        elif thousand_group and ''.join(dot_sections).isdigit():
            append_words(self.cardinal_words(int(''.join(dot_sections))))
        elif is_ordinal and number_text.isdigit():
            ord_words = self.ordinal_words(int(number_text))
            if ord_words:
                append_words(ord_words)
        elif number_text.isdigit():
            num = int(number_text)
            if num >= 1000 and not currency:
                append_words(self.cardinal_words(num))
            elif is_head and num < 100:
                append_words(self._cardinal_under_hundred(num))
            else:
                append_words([DIGIT_WORDS[int(d)] for d in number_text])
        else:
            return None, None

        if not result:
            return None, None
        phonemes = ' '.join(p for p, _ in result)
        rating = min(r for _, r in result)
        return phonemes, rating

    def append_currency(self, ps: Optional[str], currency: Optional[str]) -> Optional[str]:
        if ps is None or not currency:
            return ps
        units = CURRENCIES.get(currency)
        if not units:
            return ps
        unit_word = units[0]
        unit_phoneme, _ = self.lookup(unit_word, None)
        if not unit_phoneme:
            return ps
        return f'{ps} {unit_phoneme}'

    @staticmethod
    def numeric_if_needed(char: str) -> str:
        if not char.isdigit():
            numeric = unicodedata.numeric(char, None)
            if numeric is None:
                return char
            if numeric == int(numeric):
                return str(int(numeric))
            return char
        return char

    @staticmethod
    def is_number(word: str, is_head: bool) -> bool:
        if not any(c.isdigit() for c in word):
            return False
        value = word
        for suffix in (*ORDINAL_SUFFIXES, 'en', 'er', 't', 's'):
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                break
        if not value:
            return False
        return all(
            c.isdigit() or c in ',.-\'' or (is_head and i == 0 and c == '-')
            for i, c in enumerate(value)
        )

    def __call__(self, tk: MToken, ctx: TokenContext) -> Tuple[Optional[str], Optional[int]]:
        word = tk.text if tk._.alias is None else tk._.alias
        word = word.replace(chr(8216), "'").replace(chr(8217), "'")
        word = unicodedata.normalize('NFKC', word)
        word = ''.join(Lexicon.numeric_if_needed(c) for c in word)
        stress = None if word == word.lower() else self.cap_stresses[int(word == word.upper())]
        ps, rating = self.get_word(word, stress, ctx)
        if ps is not None:
            return apply_stress(self.append_currency(ps, tk._.currency), tk._.stress), rating
        if Lexicon.is_number(word, tk._.is_head):
            ps, rating = self.get_number(word, tk._.currency, tk._.is_head, tk._.num_flags)
            if ps is not None:
                return apply_stress(ps, tk._.stress), rating
        return None, None


class LBG2P:
    def __init__(self, version: Optional[str] = None, trf: bool = False, fallback=None, unk: str = '❓'):
        self.version = version
        self.trf = trf
        self.unk = unk
        self.lexicon = Lexicon()
        self.nlp = self._load_spacy(trf=trf)
        self.fallback = fallback

    @staticmethod
    def _load_spacy(trf: bool) -> spacy.language.Language:
        preferred = ['lb_core_news_trf', 'lb_core_news_sm', 'xx_sent_ud_sm']
        for name in preferred:
            try:
                if spacy.util.is_package(name):
                    return spacy.load(name)
                return spacy.load(name)
            except (OSError, IOError):
                continue
        return spacy.blank('lb')

    @staticmethod
    def preprocess(text: str) -> Tuple[str, List[str], Dict[int, Union[str, int, float]]]:
        result = ''
        tokens: List[str] = []
        features: Dict[int, Union[str, int, float]] = {}
        last_end = 0
        text = text.lstrip()
        for match in LINK_REGEX.finditer(text):
            result += text[last_end:match.start()]
            tokens.extend(text[last_end:match.start()].split())
            feature = match.group(2)
            if is_digit(feature[1 if feature[:1] in ('-', '+') else 0:]):
                feature_value: Union[int, float, str] = int(feature)
            elif feature in ('0.5', '+0.5'):
                feature_value = 0.5
            elif feature == '-0.5':
                feature_value = -0.5
            elif len(feature) > 1 and feature[0] == '/' and feature[-1] == '/':
                feature_value = feature[0] + feature[1:].rstrip('/')
            elif len(feature) > 1 and feature[0] == '#' and feature[-1] == '#':
                feature_value = feature[0] + feature[1:].rstrip('#')
            else:
                feature_value = None
            if feature_value is not None:
                features[len(tokens)] = feature_value
            result += match.group(1)
            tokens.append(match.group(1))
            last_end = match.end()
        if last_end < len(text):
            result += text[last_end:]
            tokens.extend(text[last_end:].split())
        return result, tokens, features

    def tokenize(
        self,
        text: str,
        tokens: List[str],
        features: Dict[int, Union[str, int, float]],
    ) -> List[MToken]:
        doc = self.nlp(text)
        mutable_tokens: List[MToken] = []
        for tk in doc:
            tag = tk.tag_
            if not tag:
                if tk.is_punct:
                    tag = 'PUNCT'
                elif tk.like_num:
                    tag = 'CD'
                elif getattr(tk, 'is_currency', False):
                    tag = '$'
                elif tk.pos_:
                    tag = tk.pos_
            mutable_tokens.append(
                MToken(
                    text=tk.text,
                    tag=tag,
                    whitespace=tk.whitespace_,
                    _=MToken.Underscore(is_head=True, num_flags='', prespace=False),
                )
            )
        if not features:
            return mutable_tokens
        align = spacy.training.Alignment.from_strings(tokens, [tk.text for tk in mutable_tokens])
        for index, value in features.items():
            if not isinstance(value, (str, int, float)):
                continue
            for i, j in enumerate(np.where(align.y2x.data == index)[0]):
                if j >= len(mutable_tokens):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mutable_tokens[j]._.stress = value
                elif isinstance(value, str):
                    if value.startswith('/'):
                        mutable_tokens[j]._.is_head = i == 0
                        mutable_tokens[j].phonemes = value.lstrip('/') if i == 0 else ''
                        mutable_tokens[j]._.rating = 5
                    elif value.startswith('#'):
                        mutable_tokens[j]._.num_flags = value.lstrip('#')
        return mutable_tokens

    def fold_left(self, tokens: List[MToken]) -> List[MToken]:
        result: List[MToken] = []
        for token in tokens:
            token = merge_tokens([result.pop(), token], unk=self.unk) if result and not token._.is_head else token
            result.append(token)
        return result

    @staticmethod
    def retokenize(tokens: List[MToken]) -> List[Union[MToken, List[MToken]]]:
        words: List[Union[MToken, List[MToken]]] = []
        currency = None
        for i, token in enumerate(tokens):
            if token._.alias is None and token.phonemes is None:
                pieces = [
                    replace(
                        token,
                        text=sub,
                        whitespace='',
                        _=MToken.Underscore(
                            is_head=True,
                            num_flags=token._.num_flags,
                            stress=token._.stress,
                            prespace=False,
                        ),
                    )
                    for sub in subtokenize(token.text)
                ]
            else:
                pieces = [token]
            pieces[-1].whitespace = token.whitespace
            for j, piece in enumerate(pieces):
                if piece._.alias is not None or piece.phonemes is not None:
                    pass
                elif piece.text in CURRENCIES:
                    currency = piece.text
                    piece.phonemes = ''
                    piece._.rating = 4
                    if words:
                        prev = words[-1]
                        target = prev if isinstance(prev, MToken) else prev[-1]
                        if target and target._.currency is None:
                            target_text = target.text.replace(',', '').replace('.', '').replace("’", '').replace("'", '')
                            if target.tag == 'CD' or target_text.isdigit():
                                target._.currency = currency
                                currency = None
                elif piece.tag == ':' and piece.text in ('-', '–'):
                    piece.phonemes = '—'
                    piece._.rating = 3
                elif piece.tag in PUNCT_TAGS or re.fullmatch(r'\W+', piece.text):
                    piece.phonemes = PUNCT_TAG_PHONEMES.get(piece.tag, ''.join(c for c in piece.text if c in PUNCTS))
                    piece._.rating = 3
                elif currency is not None:
                    if not (piece.tag == 'CD' or is_digit(piece.text) or piece.text.replace(',', '').replace('.', '').isdigit()):
                        currency = None
                    elif j + 1 == len(pieces) and (i + 1 == len(tokens) or tokens[i + 1].tag != 'CD'):
                        piece._.currency = currency
                        currency = None
                if piece._.alias is not None or piece.phonemes is not None:
                    words.append(piece)
                elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
                    piece._.is_head = False
                    words[-1].append(piece)
                else:
                    words.append(piece if piece.whitespace else [piece])
        return [w[0] if isinstance(w, list) and len(w) == 1 else w for w in words]

    @staticmethod
    def token_context(ctx: TokenContext, phonemes: Optional[str], token: MToken) -> TokenContext:
        vowel = ctx.future_vowel
        if phonemes:
            for char in phonemes:
                if char in NON_QUOTE_PUNCTS:
                    vowel = None
                    break
                if char in LB_VOWELS:
                    vowel = True
                    break
                if char.isalpha():
                    vowel = False
        future_zu = token.text.lower() in ('zu', 'zuem', 'ze')
        return TokenContext(future_vowel=vowel, future_zu=future_zu)

    @staticmethod
    def resolve_tokens(group: List[MToken]) -> None:
        text = ''.join(tk.text + tk.whitespace for tk in group[:-1]) + group[-1].text
        prespace = (
            ' ' in text
            or '/' in text
            or len({0 if c.isalpha() else (1 if is_digit(c) else 2) for c in text if c not in SUBTOKEN_JUNKS}) > 1
        )
        for index, token in enumerate(group):
            if token.phonemes is None:
                if index == len(group) - 1 and token.text in NON_QUOTE_PUNCTS:
                    token.phonemes = token.text
                    token._.rating = 3
                elif all(char in SUBTOKEN_JUNKS for char in token.text):
                    token.phonemes = ''
                    token._.rating = 3
            elif index > 0:
                token._.prespace = prespace
        if prespace:
            return
        indices = [(PRIMARY_STRESS in tk.phonemes, stress_weight(tk.phonemes), i) for i, tk in enumerate(group) if tk.phonemes]
        if len(indices) == 2 and len(group[indices[0][2]].text) == 1:
            target_index = indices[1][2]
            group[target_index].phonemes = apply_stress(group[target_index].phonemes, -0.5)
            return
        if len(indices) < 2 or sum(flag for flag, _, _ in indices) <= (len(indices) + 1) // 2:
            return
        for _, _, idx in sorted(indices)[: len(indices) // 2]:
            group[idx].phonemes = apply_stress(group[idx].phonemes, -0.5)

    def __call__(
        self,
        text: str,
        preprocess: Union[bool, Callable[[str], Tuple[str, List[str], Dict[int, Union[str, int, float]]]]] = True,
    ) -> Tuple[str, List[MToken]]:
        pre = LBG2P.preprocess if preprocess is True else preprocess
        processed_text, raw_tokens, features = pre(text) if pre else (text, [], {})
        tokens = self.tokenize(processed_text, raw_tokens, features)
        tokens = self.fold_left(tokens)
        tokens = LBG2P.retokenize(tokens)
        ctx = TokenContext()
        for index, word in reversed(list(enumerate(tokens))):
            if not isinstance(word, list):
                if word.phonemes is None:
                    phon, rating = self.lexicon(replace(word, _=word._), ctx)
                    if phon is not None:
                        word.phonemes, word._.rating = phon, rating
                if word.phonemes is None and self.fallback is not None:
                    phon, rating = self.fallback(replace(word, _=word._))
                    word.phonemes, word._.rating = phon, rating
                ctx = LBG2P.token_context(ctx, word.phonemes, word)
                continue
            left, right = 0, len(word)
            should_fallback = False
            while left < right:
                merged = None
                if not any(piece._.alias is not None or piece.phonemes is not None for piece in word[left:right]):
                    merged = merge_tokens(word[left:right])
                phon, rating = (None, None) if merged is None else self.lexicon(merged, ctx)
                if phon is not None:
                    word[left].phonemes = phon
                    word[left]._.rating = rating
                    for fragment in word[left + 1:right]:
                        fragment.phonemes = ''
                        fragment._.rating = rating
                    ctx = LBG2P.token_context(ctx, phon, merged)
                    right = left
                    left = 0
                elif left + 1 < right:
                    left += 1
                else:
                    right -= 1
                    token = word[right]
                    if token.phonemes is None:
                        if all(c in SUBTOKEN_JUNKS for c in token.text):
                            token.phonemes = ''
                            token._.rating = 3
                        elif self.fallback is not None:
                            should_fallback = True
                            break
                    left = 0
            if should_fallback and self.fallback is not None:
                merged = merge_tokens(word)
                phon, rating = self.fallback(merged)
                word[0].phonemes, word[0]._.rating = phon, rating
                for idx in range(1, len(word)):
                    word[idx].phonemes = ''
                    word[idx]._.rating = rating
            else:
                LBG2P.resolve_tokens(word)
        tokens = [merge_tokens(token, unk=self.unk) if isinstance(token, list) else token for token in tokens]
        result = ''.join((self.unk if tk.phonemes is None else tk.phonemes) + tk.whitespace for tk in tokens)
        return result, tokens


__all__ = ['LBG2P']
