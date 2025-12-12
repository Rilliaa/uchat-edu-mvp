# 🎓 UChat — Intent Classification Engine (NLU Core MVP)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/HF-Transformers-yellow.svg)]()

## 🇬🇧 EN

## 🎯 Project Description

This project presents the **NLU Intent Classification Engine** that forms the core reasoning capability of UChat.  
As an MVP, the scope is intentionally narrowed to a single essential objective:  
**mapping user utterances into consistent, high-quality intent labels.**

Two evolutionary tracks were developed:

1. **v1.0 Legacy Mode — Broad 114-Class Intent Space**  
   Exploratory stage to understand domain coverage and data sparsity limitations.

2. **v2.0 Focus Mode — Optimized 16-Class Intent Space**  
   A refined, production-ready approach built on curated datasets and linguistic augmentation.

---

## 🚀 Key Features (NLU-Only Focus)

- **Intent Classification as the Core Logic**  
  Every utterance is converted into a precise intent label, serving as the system’s decision layer.

- **Dual Model Architectures**  
  - **TF-IDF + Logistic Regression** — lightweight and fast baseline.  
  - **Fine-tuned IndoBERTweet** — deep contextual understanding and robust performance.

- **Data-Centric Improvements**  
  100 utterances per intent, enhanced with slang, typos, formal variants, and paraphrase augmentation.

- **Reduced Error Through Focused Intent Space**  
  Consolidating the scope from 114 → 16 intents dramatically improves stability and accuracy.

---

## 📊 Model Evaluation

### **1. Focus Mode (v2.0) — Recommended for Production**

| Model | Accuracy | Macro F1 | Status |
|-------|----------|----------|---------|
| **IndoBERTweet (Fine-tuned)** | **100%** | **1.00** | ✅ Stable |
| **TF-IDF + LogReg** | **97%** | **0.97** | ⚡ Strong Baseline |

**Insight:**  
Refined intent scope and improved dataset quality produce near-perfect model performance.  
This confirms that **data quality and class selection** are more impactful than expanding the number of classes.

---

### **2. Legacy Mode (v1.0) — Experimental**

| Model | Accuracy | Macro F1 | Status |
|--------|----------|----------|---------|
| **IndoBERTweet** | **94%** | **0.94** | ⚠️ Overfitted |
| **TF-IDF + LogReg** | **65%** | **0.64** | ❌ Unstable |

**Observation:**  
Intent overlap and sparse samples severely impact classical models.  
Even BERT exhibits stress in the 114-class scenario, reinforcing the decision to move to **Focus Mode** for a reliable MVP.

---

## 📂 Repository Structure

- `app.py` — Streamlit interface for model comparison.
- `notebooks/` — Training logs and evaluation reports.
- `result/` — Serialized TF-IDF model artifacts.
- `src/` — Preprocessing utilities and intent classifier modules.

---

## 🚀 Live Demo

Try our interactive demo on **Streamlit Cloud**:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uchat-edu-mvp.streamlit.app/)


---

# 🇮🇩 ID

## 🎯 Deskripsi Proyek

Proyek ini menghadirkan **Mesin Klasifikasi Intent** sebagai inti dari sistem NLU UChat.  
Pada tahap MVP, ruang lingkup difokuskan sepenuhnya pada kemampuan model dalam mengonversi kalimat pengguna menjadi **label intent** yang akurat dan stabil.

Dikembangkan melalui dua pendekatan:

1. **v1.0 Legacy Mode — 114 Intent**  
   Eksperimen awal untuk memahami cakupan domain dan tantangan kelangkaan data.

2. **v2.0 Focus Mode — 16 Intent Utama**  
   Versi yang teroptimasi, berbasis dataset terstruktur dengan kualitas yang ditingkatkan.

---

## 🚀 Fitur Utama (Fokus Intent Classification)

- **Klasifikasi Intent sebagai Otak Sistem**  
  Setiap kalimat pengguna dipetakan ke intent yang paling relevan.

- **Dua Arsitektur Model**  
  - **TF-IDF + Logistic Regression** — cepat dan ringan.  
  - **Fine-tuned IndoBERTweet** — kemampuan memahami konteks yang lebih mendalam.

- **Data-Centric Enhancements**  
  Dataset 100 data per intent dengan augmentasi bahasa gaul, typo, formal, dan parafrase.

- **Minimasi Error melalui Pengurangan Cakupan**  
  Mengurangi kelas dari 114 → 16 terbukti meningkatkan performa secara signifikan.

---

## 📊 Evaluasi Model

### **1. Focus Mode (v2.0) — Direkomendasikan**

| Model | Akurasi | Macro F1 | Status |
|--------|----------|----------|---------|
| **IndoBERTweet (Fine-tuned)** | **100%** | **1.00** | ✅ Stabil |
| **TF-IDF + LogReg** | **97%** | **0.97** | ⚡ Baseline Kuat |

**Insight:**  
Pembatasan ruang lingkup dan peningkatan kualitas data menghasilkan performa yang sangat konsisten.  
Pendekatan ini menegaskan bahwa kualitas dataset lebih penting daripada sekadar memperbanyak jumlah intent.

---

### **2. Legacy Mode (v1.0) — Eksperimental**

| Model | Akurasi | Macro F1 | Status |
|--------|----------|----------|---------|
| **IndoBERTweet** | **94%** | **0.94** | ⚠️ Cenderung Overfit |
| **TF-IDF + LogReg** | **65%** | **0.64** | ❌ Tidak Stabil |

**Observasi:**  
Tumpang tindih antar-intent dan distribusi data yang jarang menyebabkan degradasi performa, terutama pada model klasik.  
Hal ini memperkuat keputusan strategis untuk beralih ke **Focus Mode** pada MVP.

---

## 📁 Struktur Repository

- `app.py` — Antarmuka Streamlit.
- `notebooks/` — Catatan pelatihan dan evaluasi.
- `result/` — Artefak model TF-IDF.
- `src/` — Modul preprocessing dan classifier.

---

## 🚀 Demo Langung

Coba demo kami secara interaktif langsung di **Streamlit Cloud**:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uchat-edu-mvp.streamlit.app/)



