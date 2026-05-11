"""Unit tests for src/preprocessor.py."""

import pytest
from src.preprocessor import tokenize, strip_suffix, preprocess, preprocess_query


# ── Shared fixtures ──────────────────────────────────────────────────────────

MOCK_SUFFIX_RULES = [
    # longest first (sorted by len desc)
    ("ciones", ""),
    ("ción", ""),
    ("mente", ""),
    ("ando", ""),
    ("iendo", ""),
    ("ados", ""),
    ("adas", ""),
    ("ado", ""),
    ("ada", ""),
    ("idos", ""),
    ("idas", ""),
    ("ido", ""),
    ("ida", ""),
    ("ista", ""),
    ("ismo", ""),
    ("idad", ""),
    ("ble", ""),
    ("dor", ""),
    ("nte", ""),
]

MOCK_STOP_WORDS = {
    "de", "la", "el", "en", "y", "a", "que", "se", "los", "las",
    "un", "una", "por", "con", "del", "es", "para", "como",
}


# ── tokenize tests ──────────────────────────────────────────────────────────

def test_tokenize_basic():
    result = tokenize("Hola mundo")
    assert result == ["hola", "mundo"]


def test_tokenize_punctuation():
    result = tokenize("El trabajador, según el art. 3°")
    assert "trabajador" in result
    assert "según" in result
    # Single-char tokens like 'e' from "el" split should be excluded (len >= 2)
    # "el" is 2 chars so it IS included by tokenize (stop word removal is later)
    assert "el" in result


def test_tokenize_spanish_chars():
    # "raciónación" is a single string — it should tokenize as one token
    # Actually the request says ['ración', 'ación'] — but that's only if
    # there's a non-letter separator. "raciónación" is all Spanish letters.
    # Let me re-read: the spec says r'[^a-záéíóúüñ]+' — so 'ó' IS a letter.
    # "raciónación" would be one token. The test title says "spanish_chars"
    # to verify accented characters are preserved.
    result = tokenize("raciónación")
    # This is one continuous Spanish-letter token
    assert "raciónación" in result


# ── strip_suffix tests ──────────────────────────────────────────────────────

def test_strip_suffix_cion():
    # "ración" → strip "ción" → "ra" which is < 3 chars → no stripping
    result = strip_suffix("ración", MOCK_SUFFIX_RULES)
    assert result == "ración"

    # "contratación" → strip "ción" (4 chars) → "contrata" (8 chars ≥ 3) → OK
    result = strip_suffix("contratación", MOCK_SUFFIX_RULES)
    assert result == "contrata"


def test_strip_suffix_cion_correct_result():
    # Corrected test for contratación
    result = strip_suffix("contratación", MOCK_SUFFIX_RULES)
    assert result == "contrata"


def test_strip_suffix_min_stem():
    # "ido" → strip "ido" → "" which is < 3 chars → no stripping
    result = strip_suffix("ido", MOCK_SUFFIX_RULES)
    assert result == "ido"

    # "vida" → strip "ida" → "v" which is < 3 → no stripping
    result = strip_suffix("vida", MOCK_SUFFIX_RULES)
    assert result == "vida"


def test_strip_suffix_longest_first():
    # "obligaciones" ends with both "ciones" (6 chars) and "nes" (not in rules)
    # but also ends with "es". The longest matching suffix "ciones" should be used.
    result = strip_suffix("obligaciones", MOCK_SUFFIX_RULES)
    # "obligaciones"[:-6] + "" = "obliga" (6 chars ≥ 3)
    assert result == "obliga"


# ── preprocess tests ─────────────────────────────────────────────────────────

def test_preprocess_removes_stopwords():
    text = "trabajador de la empresa en contrato"
    result = preprocess(text, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
    assert "de" not in result
    assert "la" not in result
    assert "en" not in result


def test_preprocess_returns_stems():
    text = "trabajador obligaciones contratación"
    result = preprocess(text, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
    # "trabajador" → strip "dor" → "trabaja" (7 ≥ 3)
    assert "trabaja" in result
    # "obligaciones" → strip "ciones" → "obliga" (6 ≥ 3)
    assert "obliga" in result
    # "contratación" → strip "ción" → "contrata" (8 ≥ 3)
    assert "contrata" in result


def test_preprocess_query_identical_to_preprocess():
    text = "salario del trabajador"
    result1 = preprocess(text, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
    result2 = preprocess_query(text, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
    assert result1 == result2
