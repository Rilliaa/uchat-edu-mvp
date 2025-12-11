import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import re
import time

# ============================================================
# 0️⃣ PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="UChat NLU Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI yang lebih clean
st.markdown("""
    <style>
    .stTextArea textarea {font-size: 16px !important;}
    .reportview-container {background: #f5f5f5;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 1️⃣ LAZY LOADING MODELS (Memory Efficient)
# ============================================================
# Kita pisahkan loader agar model hanya dimuat jika DIPILIH user

@st.cache_resource
def load_fm_tfidf():
    """Load Focus Mode TF-IDF (16 Intents)"""
    path = "result/fm-tf-idf/fm_intent_classifier_tfidf_lr.pkl"
    try:
        if not os.path.exists(path):
            return None, f"File not found: {path}"
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_fm_indobertweet():
    """Load Focus Mode IndoBERTweet (16 Intents)"""
    model_name = "rilliaa/FM_IndoBERTweet_Intent_Classifier"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return (tokenizer, model), None
    except Exception as e:
        return (None, None), str(e)

@st.cache_resource
def load_legacy_tfidf():
    """Load Legacy TF-IDF (114 Intents)"""
    path = "result/tf-idf/intent_classifier_tfidf_lr.pkl"
    try:
        if not os.path.exists(path):
            return None, f"File not found: {path}"
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_legacy_indobertweet():
    """Load Legacy IndoBERTweet (114 Intents)"""
    model_name = "rilliaa/UChat-IndoBERTweet"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return (tokenizer, model), None
    except Exception as e:
        return (None, None), str(e)

# ============================================================
# 2️⃣ LOGIC GATE & ENTITY EXTRACTION
# ============================================================

# Map Intent ke Entity yang WAJIB dicari (Logic Gate)
# Ini memastikan bot tidak mencari 'Nilai' saat intent-nya 'Cek SPP'

INTENT_ENTITY_MAP = {
    # --- FOCUS MODE (16 Selected Intents) ---
    
    # [ID 1, 2, 7, 8, 9, 15, 16] Role: ADMIN
    "admin_student_chart_nilai": ["nama_murid", "tahun_ajaran"],
    "admin_student_cek_kehadiran": ["nama_murid", "tanggal"],
    "admin_redirect_home": [],
    "admin_redirect_student_view": [],
    "admin_redirect_teacher_view": [],
    "admin_log_activity": [],
    "admin_settings": [],

    # [ID 3, 4, 10, 11, 14] Role: MURID
    "student_chart_nilai": ["nama_murid", "tahun_ajaran"],
    "student_chart_kehadiran": ["nama_murid", "tahun_ajaran"], 
    "student_scores_view": [],
    "student_attendants_view": [],
    "student_achivement": ["nama_murid", "lokasi", "tanggal_prestasi", "nama_prestasi"],

    # [ID 5] Role: ORTU
    "parents_student_chart_nilai": ["nama_murid", "tahun_ajaran"],

    # [ID 6, 12, 13] Role: GURU
    "teacher_view_student_details": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
    "teacher_schedule_view": [],
    "teacher_score_view": [],

    # --- LEGACY INTENTS (Fallback / Optional) ---
    "admin_teacher_add": ["nama_guru", "nip", "alamat", "email"],
    "admin_student_add": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
    "admin_score_add": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
}

# ============================================================
# 3️⃣ Robust Regex Patterns
# ============================================================
ENTITY_PATTERNS = {
    # --- TIME ---
    # Menangkap: 2023/2024, 2023-2024, 2023 2024
    "tahun_ajaran": r"\b(20\d{2})[\/\-–\s](20\d{2})\b",
    
    # Menangkap: 2023-12-01, 01/12/2023, 1 Desember 2023
    "tanggal": r"(\d{4}[-/]\d{2}[-/]\d{2})|(\d{2}[-/]\d{2}[-/]\d{4})|(\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|mei|jun|jul|agu|sep|okt|nov|des)[a-z]*\s+\d{4})",
    
    # Sama dengan tanggal, tapi kuncinya beda untuk Intent Prestasi
    "tanggal_prestasi": r"(\d{4}[-/]\d{2}[-/]\d{2})|(\d{2}[-/]\d{2}[-/]\d{4})|(\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+\d{4})",

    # --- PEOPLE & ID ---
    # Menangkap: "Murid Budi", "Siswa bernama Siti", atau "Budi" (jika diawali kata kunci)
    "nama_murid": r"(?:murid|siswa|anak|bernama|si)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_guru": r"(?:guru|pengajar|bapak|ibu)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    
    "nip": r"\b\d{18}\b",
    "nisn": r"\b\d{10}\b",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

    # --- ACADEMIC ATTRIBUTES ---
    "nilai": r"(?:nilai|skor|angka)\s+(\d{1,3})",
    "nama_mapel": r"(?:mapel|pelajaran|matkul)\s+([A-Za-z\s]+)",
    
    # Menangkap: "Juara 1 Lomba Lari", "Prestasi Olimpiade Math"
    "nama_prestasi": r"(?:juara|menang|lomba|prestasi|medali)\s+([A-Za-z0-9\s]+)",
    
    # Menangkap: "di Jakarta", "Lokasi Aula", "Bertempat di Lapangan"
    "lokasi": r"(?:di|lokasi|tempat|bertempat)\s+([A-Za-z0-9\s]+)",
    
    # Menangkap: "X IPA 1", "Kelas 10", "XII-A"
    "kelas": r"(?:kelas|rombel)\s+([X|XI|XII|10|11|12]+(?:[\s-][A-Z]+)?(?:[\s-]\d+)?)",
    
    # Menangkap alamat (generic)
    "alamat": r"(?:alamat|jalan|jln|tinggal di)\s+(.*)"
}

def extract_entities_optimized(predicted_intent, text):
    """
    Hanya mengeksekusi regex yang relevan dengan intent yang terdeteksi.
    Meningkatkan performa dan mengurangi False Positive.
    """
    extracted = {}
    
    # 1. Cek apakah intent terdaftar di Map
    target_entities = INTENT_ENTITY_MAP.get(predicted_intent, [])
    
    # 2. Loop hanya pada regex yang dibutuhkan
    for entity_key in target_entities:
        pattern = ENTITY_PATTERNS.get(entity_key)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Ambil group terakhir yang valid atau full match
                val = match.group(match.lastindex) if match.lastindex else match.group(0)
                extracted[entity_key] = val.strip()
    
    return extracted

# ============================================================
# 3️⃣ PREDICTION ENGINE
# ============================================================
def predict_nlu(text, model_data, model_type):
    """Unified prediction function handling both TF-IDF and BERT"""
    start_time = time.time()
    
    try:
        if model_type == "tfidf":
            model = model_data
            probs = model.predict_proba([text])[0]
            pred_idx = np.argmax(probs)
            pred_label = model.classes_[pred_idx]
            confidence = probs[pred_idx]
            
        elif model_type == "bert":
            tokenizer, model = model_data
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
            pred_idx = np.argmax(probs)
            pred_label = model.config.id2label[pred_idx]
            confidence = probs[pred_idx]
            
        inference_time = time.time() - start_time
        return pred_label, confidence, inference_time
        
    except Exception as e:
        return None, 0.0, 0.0

# ============================================================
# 4️⃣ UI MAIN EXECUTION
# ============================================================

# --- SIDEBAR: Model Selection ---
st.sidebar.title("🎛️ Control Panel")

st.sidebar.caption("Pilih versi model untuk demonstrasi:")
model_version = st.sidebar.radio(
    "Model Version:",
    ("v2.0 Focus Mode (Recommended)", "v1.0 Legacy (Experimental)"),
    index=0
)

# Logic pemilihan model berdasarkan versi
selected_model_key = ""
if "v2.0" in model_version:
    model_architecture = st.sidebar.selectbox(
        "Architecture (Focus Mode):",
        ("TF-IDF + LogReg (Lightweight)", "IndoBERTweet (High Accuracy)")
    )
    if "TF-IDF" in model_architecture: selected_model_key = "fm_tfidf"
    else: selected_model_key = "fm_bert"
else:
    model_architecture = st.sidebar.selectbox(
        "Architecture (Legacy 114 Intents):",
        ("TF-IDF + LogReg", "IndoBERTweet")
    )
    if "TF-IDF" in model_architecture: selected_model_key = "legacy_tfidf"
    else: selected_model_key = "legacy_bert"

# Context settings
st.sidebar.markdown("---")
role = st.sidebar.selectbox("Simulasi User Role:", ["admin", "student", "teacher", "parent"])
use_role_injection = st.sidebar.checkbox("Inject Role to Prompt?", value=True, help="Menambahkan [ROLE:...] ke teks input")

# --- MAIN AREA ---
st.title("🤖 UChat NLU Engine")
st.markdown(f"**Active Model:** `{model_version}` | **Arch:** `{model_architecture}`")

# Input Area
user_input = st.text_area("User Utterance (Input Text):", height=100, placeholder="Contoh: Tolong tampilkan grafik nilai saya tahun ini...")

# Tombol Eksekusi
if st.button("Analyze Intent & Entities", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Text input is empty.")
    else:
        # 1. Prepare Data
        final_input = f"[ROLE:{role}] {user_input}" if use_role_injection else user_input
        
        # 2. Load Model (Lazy Loading happens here!)
        model_data = None
        error_msg = None
        
        with st.spinner("Loading Model & Inferencing..."):
            if selected_model_key == "fm_tfidf":
                model_data, error_msg = load_fm_tfidf()
                m_type = "tfidf"
            elif selected_model_key == "fm_bert":
                model_data, error_msg = load_fm_indobertweet()
                m_type = "bert"
            elif selected_model_key == "legacy_tfidf":
                model_data, error_msg = load_legacy_tfidf()
                m_type = "tfidf"
            elif selected_model_key == "legacy_bert":
                model_data, error_msg = load_legacy_indobertweet()
                m_type = "bert"

            # 3. Predict if model loaded
            if model_data:
                pred_intent, conf, t_time = predict_nlu(final_input, model_data, m_type)
                
                # 4. Extract Entities
                extracted_entities = extract_entities_optimized(pred_intent, user_input)
                
                # 5. Display Results
                st.markdown("### 📊 Analysis Result")
                
                # Layout Kolom
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Intent", pred_intent)
                with col2:
                    st.metric("Confidence Score", f"{conf*100:.2f}%")
                with col3:
                    st.metric("Inference Time", f"{t_time:.4f}s")

                # Visualisasi Confidence
                if conf < 0.5:
                    st.error("⚠️ Low Confidence: Model is unsure.")
                elif conf < 0.8:
                    st.warning("⚠️ Medium Confidence: Possible ambiguity.")
                else:
                    st.success("✅ High Confidence: Solid prediction.")

                # Entity Section
                st.markdown("#### 🧩 Extracted Entities")
                if extracted_entities:
                    st.json(extracted_entities)
                else:
                    st.info("No relevant entities found for this intent.")
                
                # Debug Info
                with st.expander("🛠️ Debug Information"):
                    st.write(f"**Raw Input Model:** `{final_input}`")
                    st.write(f"**Loaded Model Key:** `{selected_model_key}`")
                    
            else:
                st.error(f"❌ Failed to load model: {error_msg}")
                if "File not found" in error_msg:
                    st.warning("💡 Tips: Pastikan file `.pkl` sudah ada di folder `result/` dan sudah di-push ke GitHub/Hugging Face.")

# Footer
st.markdown("---")
st.caption("Developed for S2 Scholarship Portfolio | Comparisons between Legacy (114 Intents) vs Focus Mode (16 Intents)")
