"""
Trains the MediRoute NLU engine (Dense TF-IDF neural network) on the MTSamples
dataset and saves all artifacts needed for inference: the trained Keras model,
the fitted TF-IDF vectorizer, the label encoder, and the cleaned training text
(used later by the chatbot's retrieval-based evidence layer).

Usage:
    python src/train.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from config import (
    DATA_PATH, MODEL_DIR, MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH,
    TRAIN_TEXT_PATH, TOP_N_CLASSES, MAX_FEATURES, RANDOM_SEED,
)
from preprocess import clean_text

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}.\n"
            f"Download mtsamples.csv from "
            f"https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions "
            f"and place it in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.dropna(subset=["transcription", "medical_specialty"]).reset_index(drop=True)

    top_classes = df["medical_specialty"].value_counts().head(TOP_N_CLASSES).index.tolist()
    df_top = df[df["medical_specialty"].isin(top_classes)].reset_index(drop=True)

    print(f"Loaded {len(df)} rows -> {len(df_top)} rows after top-{TOP_N_CLASSES} filtering")

    df_top["clean_text"] = df_top["transcription"].apply(clean_text)
    return df_top


def build_model(input_dim: int, num_classes: int) -> Sequential:
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_top = load_and_prepare_data()

    le = LabelEncoder()
    y = le.fit_transform(df_top["medical_specialty"])
    y_cat = to_categorical(y)

    X_train_text, X_test_text, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
        df_top["clean_text"], y, y_cat,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()

    model = build_model(input_dim=MAX_FEATURES, num_classes=len(le.classes_))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train_tfidf, y_train_cat,
        validation_split=0.15,
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test_tfidf, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}   Test Loss: {test_loss:.4f}\n")

    y_pred = np.argmax(model.predict(X_test_tfidf), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # --- Save all artifacts needed for inference ---
    model.save(MODEL_PATH)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(TRAIN_TEXT_PATH, "wb") as f:
        pickle.dump(X_train_text.reset_index(drop=True), f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to {TFIDF_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved training text (for retrieval) to {TRAIN_TEXT_PATH}")


if __name__ == "__main__":
    main()
