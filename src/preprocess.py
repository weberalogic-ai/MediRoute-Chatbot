"""
Text cleaning utilities used both at training time and at chatbot inference time.
Keeping this in one shared module guarantees the chatbot cleans live user queries
exactly the same way the training data was cleaned.
"""

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers, remove stopwords and short tokens."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)
