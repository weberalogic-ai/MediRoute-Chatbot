"""
MediRoute chatbot engine. Loads the trained NLU model and TF-IDF vectorizer
saved by train.py, and exposes a chatbot_reply() function that turns a raw
user query into a natural-language routing response - with a confidence
threshold that falls back to a clarifying question, and a retrieval-based
evidence layer that shows the most similar real training record.

Usage (as a library):
    from chatbot import chatbot_reply
    print(chatbot_reply("I have chest pain and shortness of breath"))

Usage (interactive CLI):
    python src/chatbot.py
"""

import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH, TRAIN_TEXT_PATH,
    CONFIDENCE_THRESHOLD,
)
from preprocess import clean_text


def _load_artifacts():
    missing = [p for p in [MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH, TRAIN_TEXT_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing trained model artifacts: "
            + ", ".join(missing)
            + "\nRun `python src/train.py` first to train and save the model."
        )

    model = load_model(MODEL_PATH)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(TRAIN_TEXT_PATH, "rb") as f:
        train_text = pickle.load(f)
    return model, tfidf, le, train_text


# Loaded once, on import
_model, _tfidf, _le, _train_text = _load_artifacts()
_train_tfidf_matrix = _tfidf.transform(_train_text)


def find_similar_case(user_query: str, top_k: int = 1):
    """Returns the most similar real training record(s) for a given query."""
    cleaned = clean_text(user_query)
    vec = _tfidf.transform([cleaned])
    sims = cosine_similarity(vec, _train_tfidf_matrix)[0]
    best_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in best_idx:
        results.append({
            "similarity": float(sims[i]),
            "excerpt": _train_text.iloc[i][:200],
        })
    return results


def classify(user_query: str):
    """Returns (ranked_specialties, probabilities) for a raw user query."""
    cleaned = clean_text(user_query)
    vec = _tfidf.transform([cleaned]).toarray()
    probs = _model.predict(vec, verbose=0)[0]
    ranked = np.argsort(probs)[::-1]
    return [_le.classes_[i] for i in ranked], probs[ranked]


def chatbot_reply(user_query: str, threshold: float = CONFIDENCE_THRESHOLD, with_evidence: bool = True) -> str:
    specialties, probs = classify(user_query)
    top_specialty, top_conf = specialties[0], probs[0]

    if top_conf < threshold:
        alternatives = ", ".join(specialties[:3])
        reply = (
            f"I'm not fully certain about that one. It could relate to a few different "
            f"departments: {alternatives}. Could you share a bit more detail (e.g. main "
            f"symptom, how long it's lasted) so I can route you more accurately?"
        )
    else:
        reply = (
            f"Based on what you've described, this looks like it falls under the "
            f"{top_specialty} department (confidence: {top_conf * 100:.1f}%). "
            f"I'd recommend booking a consultation there. If this is a medical emergency, "
            f"please seek immediate in-person care or call emergency services."
        )

    if with_evidence:
        match = find_similar_case(user_query)[0]
        reply += (
            f"\n(Reference: most similar training record, similarity "
            f"{match['similarity'] * 100:.1f}% - \"{match['excerpt']}...\")"
        )

    return reply


def _run_cli():
    print("MediRoute Healthcare Query Routing Chatbot  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit"):
            print("Bot: Take care! Goodbye.")
            break
        print("Bot:", chatbot_reply(user_input))
        print("-" * 60)


if __name__ == "__main__":
    _run_cli()
