## EN
## 📓 Training Notebooks & Experiments

This directory contains the Jupyter Notebooks used for the end-to-end machine learning pipeline: from data loading and preprocessing to model training and evaluation.

The notebooks are organized into two distinct development tracks, reflecting the project's evolution from a broad experimental scope to a refined MVP product.

---

## 🧬 Notebook Index

### 1. v2.0 Focus Mode (The MVP Standard)
*Optimized training pipelines for the 16-intent scope with augmented datasets.*

| Notebook File | Description | Model Architecture |
| :--- | :--- | :--- |
| ![`fm_indobertweet_intent_classification.ipynb`](fm_indobertweet_intent_classification.ipynb) | Fine-tuning the Transformer model on the curated dataset. Achieves **100% Accuracy**. | `IndoBERTweet` |
| ![`fm_tfidf_intent_classification.ipynb`](fm_tfidf_intent_classification.ipynb) | Training the lightweight baseline model. Used for low-latency inference. Achieves **97% Accuracy**. | `TF-IDF + Logistic Regression` |

### 2. v1.0 Legacy Mode (Experimental)
*Initial experiments covering the broad 114-intent scope.*

| Notebook File | Description | Model Architecture |
| :--- | :--- | :--- |
| ![`indobertweet_intent_classification.ipynb`](indobertweet_intent_classification.ipynb) | Experimental fine-tuning on the massive intent space. Shows good performance but high resource usage. | `IndoBERTweet` |
| ![`tfidf_intent_classification.ipynb`](tfidf_intent_classification.ipynb) | Baseline training on 114 intents. Highlights the limitations of simple models on complex scopes (65% Accuracy). | `TF-IDF + Logistic Regression` |

---

## ⚙️ Pipeline Overview

All notebooks generally follow this standardized workflow:

1.  **Data Ingestion**: Loading the intent dataset (CSV).
2.  **Preprocessing**:
    * Text cleaning (lowercase, regex removal).
    * Label encoding.
    * *For BERT*: Tokenization using `AutoTokenizer`.
    * *For TF-IDF*: Vectorization using `TfidfVectorizer`.
3.  **Training**:
    * *For BERT*: Using Hugging Face `Trainer` API with GPU acceleration.
    * *For TF-IDF*: Using `scikit-learn`.
4.  **Evaluation**: Generating Classification Reports and Confusion Matrices (saved to `result/evaluation/`).
5.  **Artifact Export**: Saving the model to `result/` (pickle) or pushing to Hugging Face Hub.

---

## 🚀 How to Run

These notebooks are designed to be run in **Google Colab** or a local Jupyter environment with GPU support (specifically for the BERT models).

**Requirements:**
* Python 3.10+
* `transformers`, `scikit-learn`, `pandas`, `numpy`
* GPU (T4 or better recommended for IndoBERTweet training)

---
---

## 🇮🇩 ID 

Direktori ini berisi Jupyter Notebooks yang digunakan untuk pipeline machine learning dari ujung ke ujung: mulai dari pemuatan data, preprocessing, hingga pelatihan dan evaluasi model.

Notebook diatur ke dalam dua jalur pengembangan yang berbeda, mencerminkan evolusi proyek dari cakupan eksperimental yang luas menjadi produk MVP yang terarah.

---

## 🧬 Indeks Notebook

### 1. v2.0 Focus Mode (Standar MVP)
*Pipeline pelatihan yang dioptimalkan untuk cakupan 16-intent dengan dataset yang diaugmentasi.*

| File Notebook | Deskripsi | Arsitektur Model |
| :--- | :--- | :--- |
| ![`fm_indobertweet_intent_classification.ipynb`](fm_indobertweet_intent_classification.ipynb) | Fine-tuning model Transformer pada dataset terkurasi. Mencapai **Akurasi 100%**. | `IndoBERTweet` |
| ![`fm_tfidf_intent_classification.ipynb`](fm_tfidf_intent_classification.ipynb) | Pelatihan model baseline ringan. Digunakan untuk inferensi latensi rendah. Mencapai **Akurasi 97%**. | `TF-IDF + Logistic Regression` |

### 2. v1.0 Legacy Mode (Eksperimental)
*Eksperimen awal yang mencakup 114-intent.*

| File Notebook | Deskripsi | Arsitektur Model |
| :--- | :--- | :--- |
| ![`indobertweet_intent_classification.ipynb`](indobertweet_intent_classification.ipynb) | Eksperimen fine-tuning pada ruang intent yang masif. Menunjukkan performa baik namun boros sumber daya. | `IndoBERTweet` |
| ![`tfidf_intent_classification.ipynb`](tfidf_intent_classification.ipynb) | Pelatihan baseline pada 114 intent. Menyoroti keterbatasan model sederhana pada cakupan yang kompleks (Akurasi 65%). | `TF-IDF + Logistic Regression` |

---

## ⚙️ Ringkasan Pipeline

Semua notebook umumnya mengikuti alur kerja standar berikut:

1.  **Data Ingestion**: Memuat dataset intent.
2.  **Preprocessing**:
    * Pembersihan teks (lowercase, hapus regex).
    * Label encoding.
    * *Untuk BERT*: Tokenisasi menggunakan `AutoTokenizer`.
    * *Untuk TF-IDF*: Vektorisasi menggunakan `TfidfVectorizer`.
3.  **Training**:
    * *Untuk BERT*: Menggunakan Hugging Face `Trainer` API dengan akselerasi GPU.
    * *Untuk TF-IDF*: Menggunakan `scikit-learn`.
4.  **Evaluation**: Menghasilkan Laporan Klasifikasi dan Confusion Matrix (disimpan ke `result/evaluation/`).
5.  **Artifact Export**: Menyimpan model ke `result/` (pickle) atau push ke Hugging Face Hub.
