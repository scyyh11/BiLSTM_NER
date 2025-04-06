## 📘 BiLSTM Named Entity Recognition (NER)

This project implements a **Named Entity Recognition (NER)** system using **Bidirectional LSTM (BiLSTM)** models on the CoNLL-2003 dataset. The system supports both randomly initialized embeddings and pretrained **GloVe embeddings** with case-awareness features.

---

### 🧠 Highlights

- **Two model variants**:
  - `BiLSTM`: trained with randomly initialized word embeddings.
  - `BiLSTM + Caps + GloVe`: uses pretrained GloVe embeddings and includes capitalization features.
- Modular codebase organized for reusability and clarity.
- Uses `torchtext` for GloVe loading and `collate_fn` for padded batching.

---

### 🏗️ Project Structure

```
BiLSTM_NER/
├── config.py                     # Global configuration (e.g., hyperparams)
├── main.py                       # Entrypoint for training/evaluation
│
├── dataset/
│   ├── conll2003_dataset.py      # Standard dataset loader
│   └── conll2003_caps_dataset.py # Dataset with capitalization features
│
├── models/
│   ├── bilstm.py                 # Basic BiLSTM model
│   └── bilstm_caps.py            # BiLSTM model with capitalization support
│
├── training/
│   ├── train.py                  # Training loop
│   ├── predict.py                # Prediction/inference
│   └── evaluate.py               # Evaluation metrics (precision, recall, F1)
│
├── utils/
│   ├── caps_helper.py            # Capitalization case encoding
│   ├── collate_fn.py             # Custom collate function for padding
│   └── vocab.py                  # Vocabulary + GloVe loading via torchtext
│
└── requirements.txt              # Dependencies
```

---

### ✅ Key Features

- **GloVe Integration**: Uses `torchtext.vocab.GloVe` for automatic download and vector retrieval.
- **Capitalization Awareness**: Adds case-based features (upper/lower/title/mixed) as input signals.
- **Custom Collation**: Dynamic padding using PyTorch's `pad_sequence`.
- **Separation of Concerns**: Cleanly separates datasets, models, training logic, and utilities.

---

### 📎 Notable Files

- `bilstm.py`: Implements the baseline BiLSTM model.
- `bilstm_caps.py`: Enhanced BiLSTM model that adds capitalization case vectors.
- `vocab.py`: Loads word indices and creates an embedding matrix aligned with GloVe.
- `predict.py`: Produces CoNLL-formatted output predictions from trained models.

---