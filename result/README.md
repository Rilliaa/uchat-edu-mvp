# EN

---

## 📂 Model Artifacts & Training Results

This directory contains the trained model artifacts, serialized objects (Pickle), and references to external model repositories used in the UChat NLU Engine.

The models are categorized into two evolutionary stages:
1.  **Focus Mode (v2.0)**: Optimized for the 16-intent MVP (Recommended).
2.  **Legacy Mode (v1.0)**: Experimental baseline covering 114 intents.

---

## 🤖 Model Locations

### 1. Transformer Models (IndoBERTweet)
Due to file size limits, the fine-tuned Transformer models are hosted on **Hugging Face Hub**.

| Version | Model Name | Hugging Face Repository | Status |
| :--- | :--- | :--- | :--- |
| **v2.0 (Focus Mode)** | `FM_IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/FM_IndoBERTweet_Intent_Classifier) | ✅ Focused on more common intents |
| **v1.0 (Legacy)** | `UChat-IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/UChat-IndoBERTweet) | ⚠️ Wider Intents Scope |

### 2. Lightweight Models (TF-IDF + Logistic Regression)
These models are stored locally within this directory as serialized `.pkl` files.

| Version | Local Path | Description |
| :--- | :--- | :--- |
| **v2.0 (Focus Mode)** | [`fm-tf-idf/`](./fm-tf-idf/) | Contains vectorizer & classifier for the **16-intent** optimized scope. |
| **v1.0 (Legacy)** | [`tf-idf/`](./tf-idf/) | Contains vectorizer & classifier for the broad **114-intent** scope. |

---

## 📁 Directory Structure

```text
result/
├── evaluation/
│   ├── classification-reports
│   └── confusion-matrixs
│    
├── fm-tf-idf/                     # v2.0 Focus Mode Artifacts
│   └── fm_intent_classifier_tfidf_lr.pkl
│
├── tf-idf/                        # v1.0 Legacy Mode Artifacts
│   └── intent_classifier_tfidf_lr.pkl
│
└── README.md                      # This documentation
````

-----

## 💻 How to Load Models

### Loading TF-IDF (Local)

```python
import joblib

# Load Focus Mode Model
model_fm = joblib.load("result/fm-tf-idf/fm_intent_classifier_tfidf_lr.pkl")

# Predict
prediction = model_fm.predict(["Cek nilai saya dong"])
```

### Loading IndoBERTweet (Hugging Face)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Focus Mode Model
model_name = "rilliaa/FM_IndoBERTweet_Intent_Classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

-----

## ID

Direktori ini berisi artefak model yang telah dilatih, objek terserialisasi (Pickle), dan referensi ke repositori model eksternal yang digunakan dalam UChat NLU Engine.

Model dikategorikan ke dalam dua tahap evolusi:

1.  **Focus Mode (v2.0)**: Dioptimalkan untuk MVP 16-intent (Direkomendasikan).
2.  **Legacy Mode (v1.0)**: Baseline eksperimental mencakup 114 intent.

-----

## 🤖 Lokasi Model

### 1\. Model Transformers (IndoBERTweet)

Karena ukuran file yang besar, model Transformer disimpan di **Hugging Face Hub**.

| Versi | Nama Model | Repositori Hugging Face | Status |
| :--- | :--- | :--- | :--- |
| **v2.0 (Focus Mode)** | `FM_IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/FM_IndoBERTweet_Intent_Classifier) | ✅ Lebih Fokus ke Intent Utama |
| **v1.0 (Legacy)** | `UChat-IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/UChat-IndoBERTweet)  | ⚠️ Cakupan Intent lebih luas |

### 2\. Model Yang Lebih Ringan (TF-IDF + Logistic Regression)

Model ini disimpan secara lokal di dalam direktori ini sebagai file `.pkl`.

| Versi | Path Lokal | Deskripsi |
| :--- | :--- | :--- |
| **v2.0 (Focus Mode)** | [`fm-tf-idf/`](https://www.google.com/search?q=./fm-tf-idf/) | Berisi vectorizer & classifier untuk cakupan **16-intent**. |
| **v1.0 (Legacy)** | [`tf-idf/`](https://www.google.com/search?q=./tf-idf/) | Berisi vectorizer & classifier untuk cakupan luas **114-intent**. |

## 📁 Struktur Direktori

```text
result/
├── evaluation/
│   ├── classification-reports
│   └── confusion-matrixs
│    
├── fm-tf-idf/                    
│   └── fm_intent_classifier_tfidf_lr.pkl
│
├── tf-idf/                      
│   └── intent_classifier_tfidf_lr.pkl
│
└── README.md                      
````
