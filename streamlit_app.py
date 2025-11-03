import os
import sys
import subprocess
import streamlit as st
import traceback

# ✅ Pastikan joblib (dan scikit-learn) terinstal di runtime
try:
    import joblib
except ImportError:
    with st.spinner("Menginstal dependensi yang hilang..."):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib", "scikit-learn"])
    import joblib

try:
    from app.app import *
except Exception as e:
    st.error("🚨 Terjadi error saat memuat modul utama:")
    st.code(traceback.format_exc())
