"""Text preprocessing pipeline: tokenization, stop-word removal, and stemming.

Loads stop words from the STOP_WORD table and suffix rules from the SUFFIX table,
then applies them to produce normalized term lists for each document.
"""

import re


def load_stop_words(conn) -> set[str]:
    """Query the STOP_WORD table and return a set of all words (lowercased)."""
    cursor = conn.cursor()
    cursor.execute("SELECT word FROM STOP_WORD")
    words = {row[0].lower() for row in cursor.fetchall()}
    cursor.close()
    return words


def load_suffix_rules(conn) -> list[tuple[str, str]]:
    """Query the SUFFIX table ordered by suffix length DESC (longest first).

    Returns a list of (suffix, replacement) tuples.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT suffix, replacement FROM SUFFIX ORDER BY LENGTH(suffix) DESC"
    )
    rules = [(row[0], row[1] if row[1] else "") for row in cursor.fetchall()]
    cursor.close()
    return rules


def strip_suffix(token: str, rules: list[tuple[str, str]]) -> str:
    """Apply the first matching suffix rule to *token*.

    Only applies a rule if the remaining stem after stripping is >= 3 characters.
    Returns the modified token, or the original if no rule matches.
    """
    for suffix, replacement in rules:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + replacement
            if len(stem) >= 3:
                return stem
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase *text* and split on non-Spanish-letter characters.

    Returns tokens with length >= 2.
    """
    text = text.lower()
    tokens = re.split(r"[^a-záéíóúüñ]+", text)
    return [t for t in tokens if len(t) >= 2]


def _remove_accents(text: str) -> str:
    accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u'}
    for k, v in accents.items():
        text = text.replace(k, v)
    return text

def preprocess(text: str, stop_words: set, suffix_rules: list) -> list[str]:
    """Full preprocessing pipeline.

    Steps:
    1. Tokenize the text.
    2. Remove stop words (accent-insensitive to match MySQL collation).
    3. Apply suffix stripping.
    4. Remove stop words again (stemming might produce stop words like 'delito' -> 'del').
    5. Remove tokens shorter than 3 characters.
    """
    # Create an unaccented set of stop words for robust matching
    sw_unaccented = {_remove_accents(w) for w in stop_words}
    
    tokens = tokenize(text)
    # 1st pass stop word removal
    tokens = [t for t in tokens if _remove_accents(t) not in sw_unaccented]
    
    # Stemming
    tokens = [strip_suffix(t, suffix_rules) for t in tokens]
    
    # 2nd pass stop word removal
    tokens = [t for t in tokens if _remove_accents(t) not in sw_unaccented]
    
    # Length filter
    tokens = [t for t in tokens if len(t) >= 3]
    return tokens


def preprocess_query(query: str, stop_words: set, suffix_rules: list) -> list[str]:
    """Preprocess a query string identically to document preprocessing.

    Queries must go through the exact same pipeline as documents for LSI
    correctness.
    """
    return preprocess(query, stop_words, suffix_rules)
