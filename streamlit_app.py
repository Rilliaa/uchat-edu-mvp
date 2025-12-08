import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import re

# ============================================================
# 1️⃣ Load Models (Safety & Caching)
# ============================================================
@st.cache_resource
def load_tfidf():
    try:
        # Pastikan path ini sesuai dengan struktur folder Anda
        path = "result/tf-idf/intent_classifier_tfidf_lr.pkl"
        if not os.path.exists(path):
            st.error(f"❌ File model tidak ditemukan di: {path}")
            return None
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat TF-IDF: {e}")
        return None

@st.cache_resource
def load_indobertweet():
    try:
        model_name = "rilliaa/UChat-IndoBERTweet" 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Gagal memuat IndoBERTweet: {e}")
        return None, None

# ============================================================
# 2️⃣ Logic Gate: Intent to Entity Mapping
# ============================================================
# Daftar ini membatasi entity apa saja yang boleh diambil untuk intent tertentu.
INTENT_ENTITY_MAP = {
    # --- ADMIN TEACHER/STUDENT/PARENTS MANAGEMENT ---
    "admin_teacher_add": ["nama_guru", "nip", "alamat", "email"],
    "admin_teacher_update": ["nama_guru", "nip", "alamat", "email"],
    "admin_teacher_delete": ["nama_guru", "nip", "alamat", "email"],
    "admin_student_add": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
    "admin_student_update": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
    "admin_student_delete": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
    "admin_student_chart_kehadiran": ["nama_murid", "tahun_ajaran"],
    "admin_student_grafik_kehadiran": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    "admin_student_chart_nilai": ["nama_murid", "tahun_ajaran"],
    "admin_student_grafik_nilai": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    "admin_student_cek_kehadiran": ["nama_murid", "tanggal"],
    "admin_parents_add": ["nama_wali_murid", "nama_murid(anak)", "alamat", "email"],
    "admin_parents_update": ["nama_wali_murid", "nama_murid(anak)", "alamat", "email"],
    "admin_parents_delete": ["nama_wali_murid", "nama_murid(anak)", "alamat", "email"],
    "admin_parents_view_detail": ["nama_wali_murid", "nama_murid(anak)", "alamat", "email"],
    
    # --- ADMIN ACADEMIC & SIS ---
    "admin_academic_year_add": ["tahun_mulai", "tahun_selesai", "kode_ta"],
    "admin_academic_year_update": ["tahun_mulai", "tahun_selesai", "kode_ta"],
    "admin_academic_year_delete": ["kode_ta"],
    "admin_academic_year_transition": ["kode_ta_awal", "kode_ta_akhir"],
    "admin_class_add": ["kode_ta", "nama_kelas", "nama_wali_kelas"],
    "admin_class_update": ["kode_ta", "nama_kelas", "nama_wali_kelas"],
    "admin_class_delete": ["kode_ta", "nama_kelas", "nama_wali_kelas"],
    "admin_subject_add": ["kode_mapel", "nama_mapel", "nama_guru_pengampu"],
    "admin_subject_update": ["kode_mapel", "nama_mapel", "nama_guru_pengampu"],
    "admin_subject_delete": ["kode_mapel", "nama_mapel", "nama_guru_pengampu"],
    "admin_lesson_hour_add": ["hari", "jam_ke", "jam_mulai", "jam_selesai", "keterangan"],
    "admin_lesson_hour_update": ["hari", "jam_ke", "jam_mulai", "jam_selesai", "keterangan"],
    "admin_lesson_hour_delete": ["hari", "jam_ke", "jam_mulai", "jam_selesai", "keterangan"],
    "admin_class_schedule_add": ["kelas", "tahun_ajaran", "hari", "mata_pelajaran", "guru_pengampu"],
    "admin_class_schedule_update": ["kelas", "tahun_ajaran", "hari", "mata_pelajaran", "guru_pengampu"],
    "admin_class_schedule_delete": ["kelas", "tahun_ajaran", "hari", "mata_pelajaran", "guru_pengampu"],
    "admin_score_add": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "admin_score_update": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "admin_score_delete": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "admin_session_add": ["tanggal", "hari", "tahun_ajaran"],
    "admin_session_update": ["tanggal", "hari", "tahun_ajaran"],
    "admin_session_delete": ["tanggal", "hari", "tahun_ajaran"],
    "admin_attendance_add": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "admin_attendance_update": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "admin_attendance_delete": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "admin_violence_add": ["nama_pelanggaran", "poin_pelanggaran"],
    "admin_violence_update": ["nama_pelanggaran", "poin_pelanggaran"],
    "admin_violence_delete": ["nama_pelanggaran", "poin_pelanggaran"],
    "admin_achivement_add": ["nama_prestasi", "poin_prestasi"],
    "admin_achivement_update": ["nama_prestasi", "poin_prestasi"],
    "admin_achivement_delete": ["nama_prestasi", "poin_prestasi"],
    "admin_stud_violations_add": ["nama_murid", "nama_pelanggaran", "lokasi", "tanggal"],
    "admin_stud_violations_update": ["nama_murid", "nama_pelanggaran", "lokasi", "tanggal"],
    "admin_stud_violations_delete": ["nama_murid", "nama_pelanggaran", "lokasi", "tanggal"],
    "admin_stud_violations_detail": ["nama_murid", "nama_pelanggaran", "lokasi", "tanggal"],
    "admin_stud_achivements_add": ["nama_murid", "nama_prestasi", "lokasi", "tanggal"],
    "admin_stud_achivements_update": ["nama_murid", "nama_prestasi", "lokasi", "tanggal"],
    "admin_stud_achivements_delete": ["nama_murid", "nama_prestasi", "lokasi", "tanggal"],
    "admin_stud_achivements_detail": ["nama_murid", "nama_prestasi", "lokasi", "tanggal"],
    
    # --- STUDENT INTENTS ---
    "student_achivement": ["nama_murid", "lokasi", "tanggal", "nama_prestasi"],
    "student_violations": ["nama_murid", "lokasi", "tanggal", "nama_pelanggaran"],
    "student_chart_kehadiran": ["nama_murid", "tahun_ajaran"],
    "student_grafik_kehadiran": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    "student_cek_kehadiran": ["nama_murid", "tanggal"],
    "student_chart_nilai": ["nama_murid", "tahun_ajaran"],
    "student_grafik_nilai": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    
    # --- PARENTS INTENTS ---
    "parents_student_achivement": ["nama_murid", "lokasi", "tanggal", "nama_prestasi"],
    "parents_student_violations": ["nama_murid", "lokasi", "tanggal", "nama_pelanggaran"],
    "parents_student_chart_kehadiran": ["nama_murid", "tahun_ajaran"],
    "parents_student_grafik_kehadiran": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    "parents_student_cek_kehadiran": ["nama_murid", "tanggal"],
    "parents_student_chart_nilai": ["nama_murid", "tahun_ajaran"],
    "parents_student_grafik_nilai": ["nama_murid", "tahun_ajaran_awal", "tahun_ajaran_akhir"],
    
    # --- TEACHER INTENTS ---
    "teacher_score_add": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "teacher_score_update": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "teacher_score_delete": ["nama_murid", "nama_mapel", "nilai", "tahun_ajaran"],
    "teacher_attendance_add": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "teacher_attendance_update": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "teacher_attendance_delete": ["nama_murid", "nama_kelas", "tanggal", "status_kehadiran", "keterangan"],
    "teacher_view_student_details": ["nama_murid", "tahun_ajaran", "nisn", "kelas"],
}

# ============================================================
# 3️⃣ Robust Regex Patterns (Flexible Capturing Groups)
# ============================================================
ENTITY_PATTERNS = {
    # --- TIME & ACADEMIC YEAR ---
    "tahun_ajaran": r"\b(20\d{2})[\/\-–](20\d{2})\b",
    "kode_ta": r"(?:kode[\s:\-]+ta|TA)[_\-]?(20\d{2})",
    "tahun_mulai": r"(?:tahun[\s:\-]+mulai|mulai[\s:\-]+tahun)[\s:\-]+(20\d{2})\b",
    "tahun_selesai": r"(?:tahun[\s:\-]+selesai|selesai[\s:\-]+tahun)[\s:\-]+(20\d{2})\b",
    "tahun_ajaran_awal": r"(?:tahun[\s:\-]+ajaran[\s:\-]+awal)[\s:\-]+(20\d{2})\b",
    "tahun_ajaran_akhir": r"(?:tahun[\s:\-]+ajaran[\s:\-]+akhir)[\s:\-]+(20\d{2})\b",
    "tanggal": r"(\d{4}[-/]\d{2}[-/]\d{2})|(\d{2}[-/]\d{2}[-/]\d{4})",

    # --- NAMES & ROLES ---
    "nama_murid": r"(?:murid|siswa|anaku|anak)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_murid(anak)": r"(?:anak)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_guru": r"(?:guru)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_guru_pengampu": r"(?:guru[\s:\-]+pengampu)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_wali_kelas": r"(?:wali[\s:\-]+kelas)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "nama_wali_murid": r"(?:wali[\s:\-]+murid)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    "guru_pengampu": r"(?:pengampu)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

    # --- ACADEMIC OBJECTS & IDENTIFIERS ---
    "kelas": r"\b(X|XI|XII|10|11|12)[\s\-]?[A-Z]{0,2}\b",
    "nama_kelas": r"(?:nama[\s:\-]+kelas)[\s:\-]+([A-Z][a-z0-9\s]+)",
    "kode_mapel": r"(?:kode[\s:\-]+mapel|MAPEL)[_\-]?[A-Z0-9]+",
    "nama_mapel": r"(?:mata[\s:\-]+pelajaran|mapel)[\s:\-]+([A-Za-z\s]+)",
    "mata_pelajaran": r"(?:pelajaran)[\s:\-]+([A-Za-z\s]+)",
    "nilai": r"(?:nilai)[\s:\-]+(\d{1,3})",
    
    # --- IDs & CONTACT ---
    "nip": r"\b\d{18}\b",
    "nisn": r"\b\d{10}\b",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "alamat": r"(?:alamat)[\s:\-]+(.*)",
    "lokasi": r"(?:di|lokasi)[\s:\-]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    
    # --- TIME & STATUS ---
    "hari": r"\b(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\b",
    "jam_mulai": r"(?:jam[\s:\-]+mulai)[\s:\-]+(\d{2}:\d{2})",
    "jam_selesai": r"(?:jam[\s:\-]+selesai)[\s:\-]+(\d{2}:\d{2})",
    "jam_ke": r"(?:jam[\s:\-]+ke)[\s:\-]+(\d{1,2})",
    "status_kehadiran": r"\b(Hadir|Alpa|Sakit|Izin)\b",
    "keterangan": r"(?:keterangan)[\s:\-]+(.*)",

    # --- VIOLATIONS & ACHIEVEMENTS ---
    "nama_pelanggaran": r"(?:pelanggaran)[\s:\-]+([A-Za-z\s]+)",
    "poin_pelanggaran": r"(?:poin[\s:\-]+pelanggaran)[\s:\-]+(\d+)",
    "nama_prestasi": r"(?:prestasi)[\s:\-]+([A-Za-z\s]+)",
    "poin_prestasi": r"(?:poin[\s:\-]+prestasi)[\s:\-]+(\d+)",
    "kode_ta_awal": r"(?:ta[\s:\-]+awal)[\s:\-]+(20\d{2})\b",
    "kode_ta_akhir": r"(?:ta[\s:\-]+akhir)[\s:\-]+(20\d{2})\b",
}

# ============================================================
# 4️⃣ Conditional Extraction Function
# ============================================================
def extract_entities(predicted_intent: str, text: str):
    """
    Ekstraksi entity yang SANGAT KONDISIONAL.
    Hanya mengeksekusi regex yang relevan dengan intent yang diprediksi.
    """
    extracted_data = {}
    
    # Logic Gate: Ambil daftar entity yang DIIZINKAN untuk intent ini
    allowed_keys = INTENT_ENTITY_MAP.get(predicted_intent, [])
    
    # Hanya loop regex pattern yang ada di daftar izin tersebut
    for key in allowed_keys:
        pattern = ENTITY_PATTERNS.get(key)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Menggunakan logic capturing group (Grup terakhir atau Full Match)
                if match.groups():
                    extracted_data[key] = match.group(match.lastindex).strip()
                else:
                    extracted_data[key] = match.group(0).strip()
                    
    return extracted_data


# ============================================================
# 5️⃣ Prediction Functions
# ============================================================
def predict_tfidf(text, model):
    if model is None: return "Error Loading Model", {}
    probs = model.predict_proba([text])[0]
    pred = model.classes_[np.argmax(probs)]
    probs_percent = {cls: f"{prob*100:.2f}%" for cls, prob in zip(model.classes_, probs)}
    return pred, probs_percent

def predict_indobertweet(text, tokenizer, model):
    if model is None: return "Error Loading Model", {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = np.argmax(probs)
    labels = list(model.config.id2label.values())
    probs_percent = {label: f"{prob*100:.2f}%" for label, prob in zip(labels, probs)}
    return labels[pred_id], probs_percent


# ============================================================
# 6️⃣ Confidence Threshold & Context
# ============================================================
CONF_THRESHOLD = 0.35

def add_role_context(role: str, text: str):
    return f"[ROLE:{role}] {text.strip()}"


# ============================================================
# 7️⃣ Streamlit UI
# ============================================================
st.set_page_config(page_title="UChat NLU MVP", page_icon="🎓", layout="centered")

st.title("🎓 UChat NLU MVP — Intent & Entity Demo")

role = st.selectbox("Login sebagai:", ["admin", "teacher", "parent", "student"])
user_text = st.text_area("Masukkan perintah (Bahasa Indonesia):", height=100)
model_choice = st.radio("Pilih Model:", ["TF-IDF + Logistic Regression", "IndoBERTweet (Fine-tuned)"])

if st.button("Prediksi Intent"):
    if not user_text.strip():
        st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    else:
        # Inject role context
        augmented_text = add_role_context(role, user_text)
        st.info(f"📤 **Input ke model:** `{augmented_text}`")

        # Prediction
        pred = "Error"
        probs = {}
        
        if model_choice == "TF-IDF + Logistic Regression":
            model = load_tfidf()
            if model:
                pred, probs = predict_tfidf(augmented_text, model)
        else:
            tokenizer, model = load_indobertweet()
            if model:
                pred, probs = predict_indobertweet(augmented_text, tokenizer, model)

        # Output Logic
        if pred == "Error Loading Model" or not probs:
             st.error("❌ Gagal memuat model. Periksa path file atau koneksi internet.")
        else:
            # Confidence check
            max_conf = float(max([float(p.strip('%')) for p in probs.values()]))
            confident = max_conf >= CONF_THRESHOLD * 100

            if confident:
                # 🔥 PENTING: Passing 'pred' (intent) ke ekstraktor entity
                entities = extract_entities(pred, user_text)
                
                st.success(f"🎯 Intent: **{pred}**")
                st.json({
                    "confidence (%)": round(max_conf, 2),
                    "role_context": role,
                    "entities_extracted": entities  # Hanya menampilkan entity yang relevan
                })
            else:
                st.error("❌ Sistem tidak yakin dengan prediksi.")
                st.json({
                    "predicted_intent": "human_handsoff",
                    "closest_intent": pred,
                    "confidence (%)": round(max_conf, 2)
                })

st.markdown("---")
st.caption("💡 Dibangun dengan IndoBERTweet & TF-IDF baseline — Bahasa Indonesia Intent Classification.")
