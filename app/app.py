import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re

# ============================================================
# 1️⃣ Load Models (TF-IDF + IndoBERTweet)
# ============================================================
@st.cache_resource
def load_tfidf():
    model = joblib.load("result/intent_classifier_tfidf_lr.pkl")
    return model

@st.cache_resource
def load_indobertweet():
    model_name = "rilliaa/UChat-IndoBERTweet"  # ganti ke repo HF kamu nanti
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# ============================================================
# 2️⃣ Rule-based Entity Extraction
# ============================================================
def extract_entities(intent, text):
    entities = {}
    # regex rules (contoh, bisa kamu kembangkan)
    if "tahun" in text:
        match = re.findall(r"20\d{2}", text)
        if match:
            entities["tahun_ajaran"] = match[0]
    if "nama" in text or "murid" in text:
        match = re.findall(r"(?<=murid\s)(\w+)", text)
        if match:
            entities["nama_murid"] = match[0]
    if "tanggal" in text:
        match = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            entities["tanggal"] = match[0]
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
# 4️⃣ Confidence Threshold (Dummy Example)
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
            st.json({"confidence (%)": max_conf, "entities": entities})
        else:
            st.error("❌ Sistem tidak yakin dengan prediksi (possible OOD). Permintaan diarahkan ke admin manusia.")
            st.json({"predicted_intent": pred, "confidence (%)": max_conf})

st.markdown("---")
st.caption("💡 Dibangun dengan IndoBERTweet & TF-IDF baseline — Bahasa Indonesia Intent Classification.")
