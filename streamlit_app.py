import streamlit as st
import traceback

try:
    from app.app import *
except Exception as e:
    st.error("🚨 Terjadi error saat memuat modul utama:")
    st.code(traceback.format_exc())
