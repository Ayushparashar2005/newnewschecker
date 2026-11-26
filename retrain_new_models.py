import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_pipeline
from src.features import SequenceTokenizer
from src.evaluation import (
    train_test_split_scratch, confusion_matrix_scratch,
    accuracy_score_scratch, precision_score_scratch, recall_score_scratch, f1_score_scratch
)
from src.models.lstm_rnn import LSTMClassifier
from src.models.bert_classifier import BERTClassifier, DistilBertFeatureExtractor

def main():
    print("=== Training LSTM RNN & DistilBERT Head ===")
    true_path = os.path.join("dataset", "True.csv")
    fake_path = os.path.join("dataset", "Fake.csv")
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("Error: Dataset files not found.")
        sys.exit(1)

    # 1. Load balanced sample (2000 samples, 1000 real & 1000 fake)
    sample_size = 2000
    half = sample_size // 2
    df_t = pd.read_csv(true_path).sample(half, random_state=42)
    df_f = pd.read_csv(fake_path).sample(half, random_state=42)
    df_t['label'] = 0
    df_f['label'] = 1

    df_combined = pd.concat([df_t, df_f]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    raw_texts = (df_combined['title'].fillna('') + ' ' + df_combined['text'].fillna('')).tolist()
    y = df_combined['label'].values

    # Train/Test split (80% train, 20% test)
    indices = np.arange(len(raw_texts))
    idx_train, idx_test, y_train, y_test = train_test_split_scratch(indices, y, test_size=0.2, random_state=42)

    texts_train = [raw_texts[i] for i in idx_train]
    texts_test = [raw_texts[i] for i in idx_test]

    print(f"Loaded {len(raw_texts)} articles: {len(texts_train)} train, {len(texts_test)} test.")

    # 2. Train LSTM RNN
    print("\n--- 1. Training LSTM RNN (Pure NumPy) ---")
    t0_lstm = time.time()
    clean_train = [preprocess_pipeline(t) for t in texts_train]
    clean_test = [preprocess_pipeline(t) for t in texts_test]

    seq_tokenizer = SequenceTokenizer(max_vocab=3000, max_len=120)
    X_train_seq = seq_tokenizer.fit_transform(clean_train)
    X_test_seq = seq_tokenizer.transform(clean_test)

    lstm = LSTMClassifier(
        vocab_size=seq_tokenizer.vocab_size,
        embed_dim=48,
        hidden_dim=64,
        lr=0.005,
        epochs=15,
        batch_size=32,
        bidirectional=True,
        lambda_reg=0.0001,
        verbose=True
    )
    lstm.fit(X_train_seq, y_train)
    t_lstm_train = time.time() - t0_lstm

    train_preds_lstm = lstm.predict(X_train_seq)
    test_preds_lstm = lstm.predict(X_test_seq)

    lstm_train_acc = accuracy_score_scratch(y_train, train_preds_lstm)
    lstm_test_acc = accuracy_score_scratch(y_test, test_preds_lstm)
    lstm_prec = precision_score_scratch(y_test, test_preds_lstm)
    lstm_rec = recall_score_scratch(y_test, test_preds_lstm)
    lstm_f1 = f1_score_scratch(y_test, test_preds_lstm)
    lstm_tp, lstm_fp, lstm_tn, lstm_fn, lstm_matrix = confusion_matrix_scratch(y_test, test_preds_lstm)

    print(f"LSTM Test Accuracy: {lstm_test_acc:.4f} | F1: {lstm_f1:.4f} | Train Time: {t_lstm_train:.2f}s")

    # Save LSTM & Tokenizer immediately
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/sequence_tokenizer.pkl", "wb") as f:
        pickle.dump(seq_tokenizer, f)
    with open("saved_models/lstm_rnn.pkl", "wb") as f:
        pickle.dump(lstm, f)
    print("Saved LSTM and SequenceTokenizer to saved_models/.")

    # 3. Train DistilBERT Head
    print("\n--- 2. Extracting DistilBERT Embeddings & Training Head ---")
    t0_bert = time.time()
    print("Extracting 768-dim embeddings via DistilBERT...")
    emb_train = DistilBertFeatureExtractor.extract_embeddings(texts_train, max_length=128, batch_size=32)
    emb_test = DistilBertFeatureExtractor.extract_embeddings(texts_test, max_length=128, batch_size=32)

    bert = BERTClassifier(hidden_dim=128, lr=0.002, epochs=35, batch_size=32, verbose=True)
    bert.fit_from_embeddings(emb_train, y_train)
    t_bert_train = time.time() - t0_bert

    train_preds_bert = bert.predict(emb_train)
    test_preds_bert = bert.predict(emb_test)

    bert_train_acc = accuracy_score_scratch(y_train, train_preds_bert)
    bert_test_acc = accuracy_score_scratch(y_test, test_preds_bert)
    bert_prec = precision_score_scratch(y_test, test_preds_bert)
    bert_rec = recall_score_scratch(y_test, test_preds_bert)
    bert_f1 = f1_score_scratch(y_test, test_preds_bert)
    bert_tp, bert_fp, bert_tn, bert_fn, bert_matrix = confusion_matrix_scratch(y_test, test_preds_bert)

    print(f"DistilBERT Test Accuracy: {bert_test_acc:.4f} | F1: {bert_f1:.4f} | Train Time: {t_bert_train:.2f}s")

    bert.save_head("saved_models/bert_head.pkl")

    # 5. Update saved_models/metrics.pkl
    metrics_path = "saved_models/metrics.pkl"
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
    else:
        metrics = {}

    metrics["LSTM Recurrent NN"] = {
        "train_accuracy": lstm_train_acc,
        "accuracy": lstm_test_acc,
        "precision": lstm_prec,
        "recall": lstm_rec,
        "f1": lstm_f1,
        "train_time": t_lstm_train,
        "confusion_matrix": lstm_matrix,
        "tp": lstm_tp, "fp": lstm_fp,
        "tn": lstm_tn, "fn": lstm_fn
    }

    metrics["DistilBERT Transformer"] = {
        "train_accuracy": bert_train_acc,
        "accuracy": bert_test_acc,
        "precision": bert_prec,
        "recall": bert_rec,
        "f1": bert_f1,
        "train_time": t_bert_train,
        "confusion_matrix": bert_matrix,
        "tp": bert_tp, "fp": bert_fp,
        "tn": bert_tn, "fn": bert_fn
    }

    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    print("\nSuccessfully updated saved_models/ with LSTM and DistilBERT!")

if __name__ == "__main__":
    main()
