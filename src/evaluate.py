"""
Re-evaluates the saved MediRoute model on a fresh train/test split and prints
accuracy, classification report, and a confusion matrix plot.

Usage:
    python src/evaluate.py
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from config import MODEL_PATH, TFIDF_PATH, LABEL_ENCODER_PATH, TRAIN_TEXT_PATH
from train import load_and_prepare_data
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
from tensorflow.keras.utils import to_categorical


def main():
    model = load_model(MODEL_PATH)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    df_top = load_and_prepare_data()
    y = le.transform(df_top["medical_specialty"])
    y_cat = to_categorical(y, num_classes=len(le.classes_))

    _, X_test_text, _, y_test, _, y_test_cat = train_test_split(
        df_top["clean_text"], y, y_cat,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    X_test_tfidf = tfidf.transform(X_test_text).toarray()
    y_pred = np.argmax(model.predict(X_test_tfidf), axis=1)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
