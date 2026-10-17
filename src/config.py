"""
Shared configuration for the MediRoute healthcare chatbot project.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mtsamples.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dense_tfidf_model.keras")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
TRAIN_TEXT_PATH = os.path.join(MODEL_DIR, "train_text.pkl")  # needed for retrieval evidence

# --- Data / preprocessing settings ---
TOP_N_CLASSES = 10          # keep the N most frequent medical specialties
MAX_FEATURES = 5000         # TF-IDF vocabulary size
RANDOM_SEED = 42

# --- Chatbot settings ---
CONFIDENCE_THRESHOLD = 0.25  # below this, the bot asks a clarifying question instead of routing
