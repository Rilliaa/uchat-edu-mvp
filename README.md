Sir, ini adalah draft `README.md` yang telah saya susun secara strategis.

Berbeda dengan **Sentry-ID** yang fokus pada *detection performance*, untuk **UChat** ini saya menekankan narasi **"Product Engineering & Optimization"**. Kita menonjolkan perbandingan antara *Legacy Mode* (Ambisius tapi *noisy*) vs *Focus Mode* (Terarah dan *robust*).

Ini akan membuat pengunjung repo (termasuk *recruiter*) melihat bahwa Sir bukan hanya bisa *training* model, tapi juga paham *scoping* produk.

Silakan copy kode di bawah ini:

-----

````markdown
# 🎓 UChat — School Information System (SIS) NLU Engine
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HF-Transformers-yellow.svg)](https://huggingface.co/rilliaa)

## 🇬🇧 English Version

### 🎯 Project Description

**UChat** is a Natural Language Understanding (NLU) engine designed for School Information Systems. It serves as an intelligent middleware that translates user queries (from Students, Teachers, Parents, and Admins) into structured API calls and actionable visualizations.

This project demonstrates a critical evolution in NLP product development, presented in two versions within the application:

1.  **v1.0 Legacy Mode (Experimental):** A massive scope intent classifier covering **114 intents** across all school domains.
2.  **v2.0 Focus Mode (Stable):** An optimized, production-ready version focusing on the **Top-16 High-Value Intents**, featuring robust entity extraction and data-centric improvements.

---

### 🚀 Key Features

* **Role-Aware Context:** The NLU engine adjusts predictions based on the user's role (e.g., *Student* vs. *Admin*).
* **Dual-Architecture Support:**
    * **Lightweight:** TF-IDF + Logistic Regression (Low latency).
    * **Transformer:** Fine-tuned `IndoBERTweet` (High contextual understanding).
* **Logic-Gated Entity Extraction:** Uses a dynamic regex mapping system that only activates relevant entity patterns based on the predicted intent, reducing false positives.
* **Interactive Analytics:** Capable of rendering charts (grades, attendance) directly within the chat interface.

---

### 📊 Model Evaluation & Benchmarks

This project compares two approaches: **Broad Scope (114 Classes)** vs. **Focused Scope (16 Classes)**.

#### 1. Focus Mode (v2.0) — *Recommended*
*Optimized dataset (100 utterances/intent) with heavy linguistic augmentation (slang, typo, formal).*

| Model Architecture | Accuracy | Macro F1 | Status |
| :--- | :--- | :--- | :--- |
| **IndoBERTweet (Fine-tuned)** | **100%** | **1.00** | ✅ Production Ready |
| **TF-IDF + LogReg** | **97%** | **0.97** | ⚡ Excellent Baseline |

**Insight:** By narrowing the scope to high-impact features and improving data quality, the model achieves near-perfect stability on the test set.

**Classification Report (IndoBERTweet - Focus Mode):**
```text
                               precision    recall  f1-score   support

    admin_student_chart_nilai       1.00      1.00      1.00        21
    student_chart_kehadiran         1.00      0.95      0.97        20
    ...
    teacher_view_student_details    1.00      1.00      1.00        20

                     accuracy                           1.00       321
                    macro avg       1.00      1.00      1.00       321
                 weighted avg       1.00      1.00      1.00       321
````

#### 2\. Legacy Mode (v1.0) — *Experimental*

*Initial attempt with 114 intents. Shows the "Curse of Dimensionality" and data sparsity issues.*

| Model Architecture | Accuracy | Macro F1 | Status |
| :--- | :--- | :--- | :--- |
| **IndoBERTweet** | **94%** | **0.94** | ⚠️ Good but Overfitted |
| **TF-IDF + LogReg** | **65%** | **0.64** | ❌ Fails on nuances |

> **Observation:** While BERT handles 114 classes decently (94%), the traditional TF-IDF model collapses (65%) due to the high overlap between intents. This justifies the move to "Focus Mode" for the MVP.

-----

### 📂 Repository Structure

  * `app.py` — Main Streamlit application (lazy loading enabled).
  * `notebooks/` — Training logs for both Focus Mode and Legacy Mode.
  * `result/` — Pickle files for TF-IDF models.
  * `src/` — Helper modules for entity extraction and preprocessing.

-----

### 🚀 Live Demo

Try the interactive comparison between **v1.0** and **v2.0** on Streamlit Cloud:

[](https://uchat-edu-mvp.streamlit.app/)

-----

## 🇮🇩 Versi Bahasa Indonesia

### 🎯 Deskripsi Proyek

**UChat** adalah mesin *Natural Language Understanding* (NLU) yang dirancang untuk Sistem Informasi Sekolah. Sistem ini bertindak sebagai perantara cerdas yang menerjemahkan pertanyaan pengguna (Murid, Guru, Wali Murid, Admin) menjadi panggilan API terstruktur dan visualisasi data.

Proyek ini mendemonstrasikan evolusi pengembangan produk NLP, yang disajikan dalam dua versi:

1.  **v1.0 Legacy Mode (Eksperimental):** Klasifikasi intent dengan cakupan luas (**114 intent**) mencakup seluruh domain sekolah.
2.  **v2.0 Focus Mode (Stabil):** Versi optimasi yang berfokus pada **16 Intent Utama (High-Value)**, dengan perbaikan kualitas data dan ekstraksi entitas yang lebih tangguh.

-----

### 🚀 Fitur Utama

  * **Role-Aware Context:** Prediksi intent menyesuaikan dengan peran pengguna yang sedang login (misal: *Murid* vs *Admin*).
  * **Dukungan Arsitektur Ganda:**
      * **Ringan:** TF-IDF + Logistic Regression (Latensi rendah).
      * **Transformer:** Fine-tuned `IndoBERTweet` (Pemahaman konteks tinggi).
  * **Logic-Gated Entity Extraction:** Menggunakan pemetaan regex dinamis yang hanya mengaktifkan pola entitas sesuai intent yang terprediksi, mengurangi *false positives*.
  * **Interactive Analytics:** Mampu menampilkan grafik (nilai, kehadiran) langsung di antarmuka chat.

-----

### 📊 Evaluasi & Benchmark Model

Proyek ini membandingkan pendekatan **Broad Scope (114 Kelas)** vs **Focused Scope (16 Kelas)**.

#### 1\. Focus Mode (v2.0) — *Direkomendasikan*

*Dataset teroptimasi (100 data/intent) dengan augmentasi linguistik berat (bahasa gaul, typo, formal).*

| Arsitektur Model | Akurasi | Macro F1 | Status |
| :--- | :--- | :--- | :--- |
| **IndoBERTweet (Fine-tuned)** | **100%** | **1.00** | ✅ Siap Produksi |
| **TF-IDF + LogReg** | **97%** | **0.97** | ⚡ Baseline Sangat Baik |

> **Insight:** Dengan membatasi ruang lingkup pada fitur berdampak tinggi dan meningkatkan kualitas data (Data-Centric AI), model mencapai stabilitas hampir sempurna pada data uji.

**Laporan Klasifikasi (IndoBERTweet - Focus Mode):**

```text
                               precision    recall  f1-score   support

    admin_student_chart_nilai       1.00      1.00      1.00        21
    student_chart_kehadiran         1.00      0.95      0.97        20
    ...
    teacher_view_student_details    1.00      1.00      1.00        20

                     accuracy                           1.00       321
                    macro avg       1.00      1.00      1.00       321
                 weighted avg       1.00      1.00      1.00       321
```

#### 2\. Legacy Mode (v1.0) — *Eksperimental*

*Percobaan awal dengan 114 intent. Menunjukkan tantangan "Curse of Dimensionality" dan kelangkaan data.*

| Arsitektur Model | Akurasi | Macro F1 | Status |
| :--- | :--- | :--- | :--- |
| **IndoBERTweet** | **94%** | **0.94** | ⚠️ Bagus tapi Overfit |
| **TF-IDF + LogReg** | **65%** | **0.64** | ❌ Gagal menangkap nuansa |

> **Observasi:** Sementara BERT mampu menangani 114 kelas dengan cukup baik (94%), model tradisional TF-IDF jatuh performanya (65%) karena tingginya kemiripan antar-intent. Ini menjustifikasi keputusan strategis untuk beralih ke "Focus Mode" demi MVP yang reliabel.

-----

### 📁 Struktur Repository

  * `app.py` — Aplikasi utama Streamlit (*lazy loading enabled*).
  * `notebooks/` — Log pelatihan untuk Focus Mode dan Legacy Mode.
  * `result/` — File model `.pkl` untuk TF-IDF.
  * `src/` — Modul pembantu untuk ekstraksi entitas dan preprocessing.

-----

### 🚀 Demo Langsung

Coba perbandingan interaktif antara **v1.0** dan **v2.0** di Streamlit Cloud:

[](https://uchat-edu-mvp.streamlit.app/)

```

***

### Poin Strategis yang Saya Masukkan:

1.  **Tabel Komparasi:** Saya menggunakan tabel untuk membandingkan akurasi v1.0 vs v2.0. Ini memberikan visualisasi cepat kepada pembaca tentang *kenapa* Sir melakukan *Focus Mode*.
2.  **Snippet Classification Report:** Saya hanya mengambil potongan penting dari gambar yang Sir berikan (FM IndoBERTweet) sebagai bukti validitas.
3.  **Link Streamlit:** Jangan lupa ganti URL `https://uchat-engine.streamlit.app/` dengan link aplikasi Sir yang sebenarnya nanti.
4.  **Insight Blocks:** Saya menambahkan blok kutipan (`>`) yang berisi "Insight" atau "Observasi". Ini menunjukkan bahwa Sir berpikir kritis (*critical thinking*) tentang hasil model, bukan sekadar *coding*.
```
