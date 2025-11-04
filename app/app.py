import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import re

# ============================================================
# 1️⃣ Load Models (TF-IDF + IndoBERTweet)
# ============================================================
@st.cache_resource

def load_tfidf():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../result/tf-idf/intent_classifier_tfidf_lr.pkl")
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_resource
def load_indobertweet():
    model_name = "rilliaa/UChat-IndoBERTweet"  # ganti ke repo HF kamu nanti
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# ============================================================
# 2️⃣ Rule-based Entity Extraction (Versi Revisi)
# ============================================================

ENTITY_PATTERNS = {
    # Tahun Ajaran dan Tahun Akademik
    "tahun_ajaran": r"(20\d{2})[/\-–](20\d{2})",
    "kode_ta": r"TA[_\-]?(20\d{2})",
    "tahun_mulai": r"\b(20\d{2})\b(?=.*(mulai|awal))",
    "tahun_selesai": r"\b(20\d{2})\b(?=.*(akhir|selesai))",

    # Tanggal
    "tanggal": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",

    # Nama-nama (guru, murid, wali)
    "nama_murid": r"(?<=murid\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_murid(anak)": r"(?<=anak\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_guru": r"(?<=guru\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_guru_pengampu": r"(?<=pengampu\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_wali_kelas": r"(?<=wali\s(kelas)\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_wali_murid": r"(?<=wali\s(murid)\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

    # Kelas, Mapel, Nilai
    "kelas": r"\b(X|XI|XII)[\s\-]?[A-Z]?\b",
    "kode_mapel": r"MAPEL[_\-]?[A-Z0-9]+",
    "nama_mapel": r"(?<=mata pelajaran\s)([A-Za-z\s]+)",
    "mata_pelajaran": r"(?<=pelajaran\s)([A-Za-z\s]+)",
    "nilai": r"\b\d{1,3}\b",

    # Hari dan jam
    "hari": r"\b(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\b",
    "jam_mulai": r"(?<=mulai\s)(\d{2}:\d{2})",
    "jam_selesai": r"(?<=selesai\s)(\d{2}:\d{2})",
    "jam_ke": r"(?<=jam ke\s)(\d{1,2})",

    # Lokasi dan alamat
    "lokasi": r"(?<=di\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "alamat": r"(?<=alamat\s)(.*)",

    # Identitas
    "nip": r"\b\d{18}\b",
    "nisn": r"\b\d{10}\b",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

    # Pelanggaran & Prestasi
    "nama_pelanggaran": r"(?<=pelanggaran\s)([A-Za-z\s]+)",
    "poin_pelanggaran": r"(?<=poin\s(pelanggaran)\s)(\d+)",
    "nama_prestasi": r"(?<=prestasi\s)([A-Za-z\s]+)",
    "poin_prestasi": r"(?<=poin\s(prestasi)\s)(\d+)",
    "keterangan": r"(?<=keterangan\s)([A-Za-z\s]+)",

    # Status Kehadiran
    "status_kehadiran": r"\b(Hadir|Alpa|Sakit|Izin)\b",
}


def extract_entities(intent: str, text: str):
    entities = {}
    for name, pattern in ENTITY_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities[name] = match.group(1) if len(match.groups()) == 1 else match.group(0)
    return entities


# ============================================================
# 3️⃣ Prediction Functions
# ============================================================

def predict_tfidf(text, model):
    probs = model.predict_proba([text])[0]
    pred = model.classes_[np.argmax(probs)]
    probs_percent = {cls: f"{prob*100:.2f}%" for cls, prob in zip(model.classes_, probs)}
    return pred, probs_percent


def predict_indobertweet(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = np.argmax(probs)
    labels = list(model.config.id2label.values())
    probs_percent = {label: f"{prob*100:.2f}%" for label, prob in zip(labels, probs)}
    return labels[pred_id], probs_percent


# ============================================================
# 4️⃣ Confidence Threshold
# ============================================================
CONF_THRESHOLD = 0.35
def is_confident(probs):
    return max(probs.values()) > CONF_THRESHOLD * 100


# ============================================================
# 5️⃣ Streamlit UI
# ============================================================

st.set_page_config(page_title="UChat NLU MVP", page_icon="🎓", layout="centered")

st.title("🎓 UChat NLU MVP — Intent & Entity Demo")

role = st.selectbox("Login sebagai:", ["admin", "teacher", "parent", "student"])
text = st.text_area("Masukkan perintah (Bahasa Indonesia):", height=100)
model_choice = st.radio("Pilih Model:", ["TF-IDF + Logistic Regression", "IndoBERTweet (Fine-tuned)"])

if st.button("Prediksi Intent"):
    if not text.strip():
        st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    else:
        if model_choice == "TF-IDF + Logistic Regression":
            model = load_tfidf()
            pred, probs = predict_tfidf(text, model)
        else:
            tokenizer, model = load_indobertweet()
            pred, probs = predict_indobertweet(text, tokenizer, model)

        # Confidence Check
        max_conf = float(max([float(p.strip('%')) for p in probs.values()]))
        confident = max_conf >= CONF_THRESHOLD * 100

        if confident:
            entities = extract_entities(pred, text)
            st.success(f"🎯 Intent: **{pred}**")
            st.json({
                "confidence (%)": round(max_conf, 2),
                "role": role,
                "entities": entities
            })
        else:
            st.error("❌ Sistem tidak yakin dengan prediksi (possible OOD). Permintaan diarahkan ke admin manusia.")
            st.json({"predicted_intent": pred, "confidence (%)": round(max_conf, 2)})

st.markdown("---")
st.caption("💡 Dibangun dengan IndoBERTweet & TF-IDF baseline — Bahasa Indonesia Intent Classification.")
