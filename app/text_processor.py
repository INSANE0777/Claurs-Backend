import re
import string
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

from app.config import get_settings

try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Minimal fallback stopword list if NLTK data is not downloaded
    _STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
        "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "himself", "him", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "itself", "let", "me", "more", "most", "my", "myself", "nor", "of",
        "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
        "your", "yours", "yourself", "yourselves",
    }

_STEMMER = PorterStemmer()


def ensure_nltk_data() -> None:
    """Download required NLTK data if missing."""
    try:
        import nltk
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def strip_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if len(t) > 1]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [_STEMMER.stem(t) for t in tokens]


def process_text(raw_html_or_text: str, title: str = "") -> Dict[str, Any]:
    ensure_nltk_data()
    text = strip_html(raw_html_or_text)
    full_text = f"{title} {text}".strip()
    normalized = normalize_text(full_text)
    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    stemmed = stem_tokens(tokens)
    return {
        "content_text": text,
        "content_processed": " ".join(tokens),
        "tokens": tokens,
        "stemmed": stemmed,
    }


def build_snippet(text: str, query_terms: List[str], max_len: int = 160) -> str:
    if not text:
        return ""
    text = strip_html(text)
    lower = text.lower()
    best_pos = 0
    for term in query_terms:
        idx = lower.find(term)
        if idx != -1:
            best_pos = max(0, idx - 30)
            break
    snippet = text[best_pos : best_pos + max_len]
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_len < len(text):
        snippet = snippet + "..."
    return snippet
