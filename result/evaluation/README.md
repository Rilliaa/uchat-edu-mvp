
# 📈 Evaluation Metrics & Visualizations

This directory contains the **forensic evidence** of the model's performance. It visualizes the transition from a broad, noisy intent scope (v1.0) to a focused, high-precision MVP scope (v2.0).

The artifacts here are generated automatically using `scikit-learn` and `matplotlib` during the training phase.

---

## 🏆 Benchmark Summary

The visual reports below confirm the hypothesis: **Reducing intent scope significantly improves model stability.**

### 1. v2.0 Focus Mode (The MVP Standard)
*Optimized 16-Intent Scope with Augmented Data.*

| Metric Image | Model | Performance Highlights |
| :--- | :--- | :--- |
| [`classification-report-fm-indobertweet.png`](./classification-report-fm-indobertweet.png) | **IndoBERTweet** | **Perfect Score (1.00)**. Precision and Recall are balanced across all classes. |
| [`classification-report-fm-tf-idf.png`](./classification-report-fm-tf-idf.png) | **TF-IDF + LR** | **Excellent (0.97)**. Even the lightweight model performs near-perfectly with cleaner data. |

> **Key Takeaway:** In Focus Mode, the gap between the complex model (BERT) and the simple model (TF-IDF) is very small (3%), proving that the dataset is high quality and distinct.

### 2. v1.0 Legacy Mode (The Baseline)
*Broad 114-Intent Scope (Experimental).*

| Metric Image | Model | Performance Highlights |
| :--- | :--- | :--- |
| [`classification-report-indobertweet.png`](classification-report-indobertweet.png) | **IndoBERTweet** | **Good (0.94)**, but prone to overfitting due to data sparsity per intent. |
| [`classification-report-tf-idf.png`](classification-report-tf-idf.png) | **TF-IDF + LR** | **Poor (0.65)**. The model fails to distinguish between overlapping intents (e.g., `teacher_score_update` vs `teacher_score_view`). |

---

## 📂 File Naming Convention

The files follow this naming pattern for easy identification:

* **Prefix `fm-`**: Indicates **Focus Mode** (v2.0 / 16 Intents).
* **No Prefix**: Indicates **Legacy Mode** (v1.0 / 114 Intents).
* **`classification-report-*`**: Shows Precision, Recall, F1-Score, and Support.
* **`conf-matrix-*`**: Heatmap visualization of True Labels vs. Predicted Labels.

---

## 📊 Detailed Analysis

### Why did Legacy TF-IDF fail?
As seen in [`classification-report-tf-idf.png`](classification-report-tf-idf.png), the traditional model struggled with semantic overlap. For example, intents like `teacher_score_update` (Recall 0.20) were often confused with similar administrative commands.

### Why is Focus Mode superior?
As seen in [`classification-report-fm-tf-idf.png`](classification-report-fm-tf-idf.png), reducing the scope allowed the decision boundaries to become clear. Even classes that usually confuse models (like `student_chart_kehadiran` vs `student_chart_nilai`) achieved >0.95 F1-Score because the keywords were distinct enough for the optimized dataset.

---
---

# 🇮🇩 ID (Versi Bahasa Indonesia)

Direktori ini berisi **bukti forensik** dari performa model. Folder ini memvisualisasikan transisi dari cakupan intent yang luas dan *noisy* (v1.0) ke cakupan MVP yang fokus dan presisi tinggi (v2.0).

Artefak di sini dihasilkan secara otomatis menggunakan `scikit-learn` dan `matplotlib` selama fase pelatihan.

---

## 🏆 Ringkasan Benchmark

Laporan visual di bawah ini mengonfirmasi hipotesis: **Mengurangi cakupan intent secara signifikan meningkatkan stabilitas model.**

### 1. v2.0 Focus Mode (Standar MVP)
*Cakupan 16-Intent yang Dioptimalkan dengan Data Augmentasi.*

| File Metrik | Model | Highlight Performa |
| :--- | :--- | :--- |
| [`classification-report-fm-indobertweet.png`](classification-report-fm-indobertweet.png) | **IndoBERTweet** | **Skor Sempurna (1.00)**. Precision dan Recall seimbang di seluruh kelas. |
| [`classification-report-fm-tf-idf.png`](./classification-report-fm-tf-idf.png) | **TF-IDF + LR** | **Sangat Baik (0.97)**. Bahkan model ringan pun berkinerja hampir sempurna dengan data yang lebih bersih. |

> **Poin Penting:** Dalam Focus Mode, celah antara model kompleks (BERT) dan model sederhana (TF-IDF) sangat kecil (3%), membuktikan bahwa dataset memiliki kualitas tinggi dan *distinct*.

### 2. v1.0 Legacy Mode (Baseline)
*Cakupan Luas 114-Intent (Eksperimental).*

| File Metrik | Model | Highlight Performa |
| :--- | :--- | :--- |
| [`classification-report-indobertweet.png`](classification-report-indobertweet.png) | **IndoBERTweet** | **Bagus (0.94)**, namun rentan *overfitting* karena kelangkaan data per intent. |
| [`classification-report-tf-idf.png`](classification-report-tf-idf.png) | **TF-IDF + LR** | **Buruk (0.65)**. Model gagal membedakan intent yang tumpang tindih (misalnya: `teacher_score_update` vs `teacher_score_view`). |

---

## 📂 Konvensi Penamaan File

File mengikuti pola penamaan ini untuk identifikasi yang mudah:

* **Awalan `fm-`**: Menandakan **Focus Mode** (v2.0 / 16 Intent).
* **Tanpa Awalan**: Menandakan **Legacy Mode** (v1.0 / 114 Intent).
* **`classification-report-*`**: Menampilkan Precision, Recall, F1-Score, dan Support.
* **`conf-matrix-*`**: Visualisasi Heatmap dari Label Asli vs Label Prediksi.
