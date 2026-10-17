# MediRoute: A Neural Network-Based Healthcare Chatbot for Medical Query Classification and Specialist Routing

MediRoute is a Medical NLP healthcare chatbot that classifies a patient's free-text query
into the correct medical specialty and responds conversationally — either routing the user
to a department or asking a clarifying question when uncertain.

At its core is an NLU (Natural Language Understanding) engine: a feedforward neural network
trained on the [MTSamples (Medical Transcriptions)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
dataset, which classifies free text into one of the ten most common medical specialties.
A lightweight response-generation layer wraps this classifier in a confidence-aware,
conversational interface, and a retrieval-based evidence layer shows the most similar real
training record behind every answer for transparency.

## Features

- **NLU Engine** — a Dense feedforward neural network (Keras) trained on TF-IDF features, achieving 43.19% test accuracy across 10 medical specialties (see `/reports` for the full comparison against a bonus LSTM model and published literature).
- **Confidence-aware routing** — high-confidence predictions route the user to a department; low-confidence predictions trigger a clarifying follow-up instead of a guess.
- **Retrieval-based evidence** — every reply is backed by the single most similar real record from the training data, for transparency and debuggability.
- **Interactive CLI chatbot** — talk to MediRoute directly from the terminal.

## Project Structure

```
mediroute-project/
├── data/                   # place mtsamples.csv here (not included — see Setup)
├── models/                 # saved model artifacts (created by train.py)
├── notebooks/
│   └── medical_nlp_healthcare_chatbot.ipynb   # full exploratory notebook (EDA, training, both models, demo)
├── reports/
│   └── MediRoute_Healthcare_Chatbot_Report.docx
├── src/
│   ├── config.py            # paths & hyperparameters
│   ├── preprocess.py         # shared text-cleaning function
│   ├── train.py               # trains & saves the NLU model
│   ├── chatbot.py             # loads the model, serves chatbot replies, interactive CLI
│   └── evaluate.py            # re-evaluates the saved model, plots confusion matrix
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo and install dependencies:**
   ```bash
   git clone <your-repo-url>
   cd mediroute-project
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   Download `mtsamples.csv` from
   [Kaggle: Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
   and place it in the `data/` folder.

3. **Train the model:**
   ```bash
   python src/train.py
   ```
   This cleans the data, trains the Dense TF-IDF neural network, prints evaluation metrics,
   and saves the model + vectorizer + label encoder into `models/`.

4. **Chat with MediRoute:**
   ```bash
   python src/chatbot.py
   ```
   Type a medical query and press Enter. Type `quit` to exit.

   Or use it as a library:
   ```python
   from src.chatbot import chatbot_reply
   print(chatbot_reply("I have chest pain and shortness of breath"))
   ```

5. **Re-evaluate / regenerate the confusion matrix:**
   ```bash
   python src/evaluate.py
   ```

## Dataset

[MTSamples (Medical Transcriptions)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions),
T. Boyle, Kaggle, 2019 — 4,999 real, de-identified medical transcription records across
~40 medical specialties, CC0 Public Domain license. This project restricts the classification
task to the 10 most frequent specialties to address severe class imbalance in the raw data.

## Results Summary

| Model                         | Test Accuracy |
|--------------------------------|:-------------:|
| Dense Neural Network + TF-IDF (primary) | 43.19% |
| Embedding + LSTM (extension)    | 39.34% |

Full methodology, literature survey, architecture details, and discussion of limitations
are documented in `reports/MediRoute_Healthcare_Chatbot_Report.docx`.

## Team

| Name | Roll No. |
|---|---|
| Jit Nayak | 23CS2011050 |
| Ankush Sarkhel | 23CS2011020 |
| Abhisek Tudu | 23CS2011002 |
| Debajit Patra | 23CS2011040 |
| Antarik De | 23CS2011021 |

## License

This project is for academic coursework purposes. The MTSamples dataset is used under its
CC0 Public Domain license.
