"""PARKNEROO dashboard. Run: streamlit run app.py"""
import json
import sqlite3
import tempfile
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from features import FEATURES, extract_features

st.set_page_config(page_title="PARKNEROO", page_icon="🧠", layout="wide")

# Custom CSS for a cleaner, medical-grade look
st.markdown("""
<style>
    .stMetric { background-color: #f7f9fa; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e1e4e8; }
    h1, h2, h3 { color: #1a202c; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 PARKNEROO")
st.caption("AI-powered, non-invasive multimodal health monitoring platform for tracking digital patterns associated with Parkinson’s disease. **Research Purpose Only. Not a diagnosis.**")
st.divider()

@st.cache_resource
def load():
    return joblib.load("outputs/model.joblib")

@st.cache_resource
def db():
    con = sqlite3.connect("sessions.db", check_same_thread=False)
    # Voice table
    con.execute("CREATE TABLE IF NOT EXISTS s (user TEXT, ts TEXT, prob REAL, feats TEXT)")
    # Handwriting table
    con.execute("CREATE TABLE IF NOT EXISTS spiral_s (user TEXT, ts TEXT, tremor REAL, variance REAL, feats TEXT)")
    return con

bundle, con = load(), db()
model, background = bundle["model"], bundle["background"]
predict = lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1]

st.sidebar.header("Patient Profile")
user = st.sidebar.text_input("User ID", "demo_user")
st.sidebar.markdown("---")
st.sidebar.write(f"Active Voice Model: **{bundle['name']}**")
st.sidebar.info("💡 Make sure to use the same User ID to track changes over time!")

tab1, tab2 = st.tabs(["🎤 Voice Analysis", "✍️ Handwriting & Movement"])

with tab1:
    st.header("Voice Pattern Analysis")
    st.write("Upload a 3-5 second audio recording of a sustained 'aaah' sound.")
    
    up = st.file_uploader("Upload today's recording (.wav)", type=["wav"], key="voice_upload")
    if up and st.button("Analyse & save session", key="voice_analyze", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(up.read())
        try:
            with st.spinner("Analyzing voice patterns..."):
                feats = extract_features(f.name)
                x = pd.DataFrame([feats])[FEATURES]
                prob = float(predict(x)[0])
                con.execute("INSERT INTO s VALUES (?,?,?,?)", (user, datetime.now().isoformat(timespec="seconds"), prob, json.dumps(feats)))
                con.commit()
                
            st.success("Voice analysis complete!")
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.metric("Parkinson's-associated score", f"{prob:.2f}", help="Probability based on the ML model.")
            
            with col_m2:
                st.write("**Top contributing features (SHAP)**")
                ex = shap.Explainer(predict, background)(x)
                fig, ax = plt.subplots(figsize=(6, 3))
                shap.plots.waterfall(ex[0], max_display=6, show=False)
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    st.markdown("### 📈 Longitudinal Monitoring (Voice)")
    hist = pd.read_sql("SELECT ts, prob, feats FROM s WHERE user=? ORDER BY ts", con, params=(user,))
    if hist.empty:
        st.info("No voice sessions recorded yet.")
    else:
        hist["ts"] = pd.to_datetime(hist["ts"])
        base_n = 3
        st.line_chart(hist.set_index("ts")["prob"], use_container_width=True)
        if len(hist) <= base_n:
            st.warning(f"⏳ Baseline building: {len(hist)}/{base_n} sessions recorded.")
        else:
            base = hist["prob"].iloc[:base_n]
            recent = hist["prob"].iloc[-3:].mean()
            threshold = base.mean() + max(2 * base.std(ddof=0), 0.10)
            if recent > threshold:
                st.error(f"⚠️ Persistent upward shift vs. your baseline ({base.mean():.2f} → {recent:.2f}). Consider discussing with a healthcare professional.")
            else:
                st.success(f"✅ Stable relative to baseline ({base.mean():.2f} → {recent:.2f}).")
            
            with st.expander("View Feature Z-Scores"):
                F = pd.DataFrame([json.loads(s) for s in hist["feats"]])
                z = (F.iloc[-3:].mean() - F.iloc[:base_n].mean()) / (F.iloc[:base_n].std(ddof=0) + 1e-9)
                st.bar_chart(z.reindex(z.abs().sort_values(ascending=False).index[:6]))

with tab2:
    st.header("Handwriting Kinematics")
    st.write("Upload an image of a drawn spiral. The computer vision system will extract digital biomarkers and track them over time.")
    
    col_img1, col_img2 = st.columns(2)
    
    uploaded_image = st.file_uploader("Upload drawing image (.png, .jpg)", type=["png", "jpg", "jpeg"], key="draw_upload")
    
    if uploaded_image is not None:
        with col_img1:
            st.image(uploaded_image, caption="Original Upload", use_container_width=True)
            
        if st.button("Extract & Save Kinematic Features", key="extract_vision", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(uploaded_image.read())
            try:
                with st.spinner("Running Computer Vision Analysis..."):
                    import vision_features
                    feats, processed_img = vision_features.analyze_spiral(f.name)
                    
                    con.execute("INSERT INTO spiral_s VALUES (?,?,?,?,?)", 
                                (user, datetime.now().isoformat(timespec="seconds"), 
                                 feats["tremor_index"], feats["thickness_variance"], json.dumps(feats)))
                    con.commit()

                with col_img2:
                    st.image(processed_img, channels="BGR", caption="Processed Vision Overlay", use_container_width=True)
                
                st.success("Kinematic features extracted and saved!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tremor Index", f"{feats['tremor_index']:.3f}", help="Higher means more jagged lines.")
                col2.metric("Thickness Variance", f"{feats['thickness_variance']:.2f}", help="Proxy for pressure hesitation.")
                col3.metric("Drawing Length", f"{feats['drawing_length']:.1f}")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    st.markdown("### 📈 Longitudinal Monitoring (Handwriting)")
    hist_sp = pd.read_sql("SELECT ts, tremor, variance FROM spiral_s WHERE user=? ORDER BY ts", con, params=(user,))
    if hist_sp.empty:
        st.info("No spiral test sessions recorded yet.")
    else:
        hist_sp["ts"] = pd.to_datetime(hist_sp["ts"])
        st.write("**Tremor Index Over Time**")
        st.line_chart(hist_sp.set_index("ts")["tremor"], use_container_width=True)
        st.write("**Thickness Variance (Pressure Hesitation) Over Time**")
        st.line_chart(hist_sp.set_index("ts")["variance"], use_container_width=True)