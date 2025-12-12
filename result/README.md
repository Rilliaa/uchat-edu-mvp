# 📂 Model Artifacts & Training Results

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
| **v2.0 (Focus Mode)** | `FM_IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/FM_IndoBERTweet_Intent_Classifier) | ✅ **Production Ready** |
| **v1.0 (Legacy)** | `UChat-IndoBERTweet` | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open-yellow)](https://huggingface.co/rilliaa/UChat-IndoBERTweet) | ⚠️ Experimental |

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
├── fm-tf-idf/                     # v2.0 Focus Mode Artifacts
│   └── fm_intent_classifier_tfidf_lr.pkl
│
├── tf-idf/                        # v1.0 Legacy Mode Artifacts
│   └── intent_classifier_tfidf_lr.pkl
│
└── README.md                      # This documentation
