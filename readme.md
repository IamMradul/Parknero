# PARKNEROO

It is an AI-powered, non-invasive health monitoring platform designed to help identify and track digital patterns associated with Parkinson’s disease. The system analyses multiple measurable signals such as voice, hand and finger movements, gait, and handwriting using machine learning, signal processing, and computer vision. Instead of relying on a single symptom, PARKNEROO combines information from different modalities and compares changes with an individual’s personal baseline over time. The platform can provide an easy-to-understand monitoring report highlighting changes in movement, speech, or other digital biomarkers and can encourage timely professional medical evaluation when persistent changes are observed. Its biotechnology component connects these digital patterns with the biological understanding of Parkinson’s disease, while an optional bioinformatics module can explore publicly available molecular and gene-expression datasets to support research. The major benefits of PARKNEROO are non-invasive monitoring, early awareness of concerning changes, longitudinal tracking, multimodal analysis, explainable AI, and research support. However, PARKNEROO is intended as a research and monitoring-support system, not a replacement for professional neurological diagnosis or treatment.

**Research prototype. Not a diagnostic tool.**

## Current Modalities

### 1. Voice Analysis
Voice-based pattern classifier with explainability and personal-baseline tracking.

#### Run
```bash
pip install -r requirements.txt
python train.py          # downloads UCI data, runs subject-wise CV, saves outputs/model.joblib
streamlit run app.py     # dashboard
```

#### Pipeline
1. `features.py` – Praat (parselmouth) extracts 16 voice features (F0, jitter, shimmer, HNR) matching the UCI dataset.
2. `train.py` – LogReg vs RandomForest vs XGBoost, **StratifiedGroupKFold by subject**, reports ROC-AUC / sensitivity / specificity / F1, SHAP summary.
3. `app.py` – upload .wav → score + SHAP waterfall → saved to SQLite → baseline (first 3 sessions) vs recent trend.

### 2. Handwriting & Movement (Coming Soon)
Planned feature to analyze spiral drawings and hand movement tracking to complement voice data.

## Research ideas for your report
- Compare random split vs subject-wise split to show how much leakage inflates accuracy.
- Feature ablation: jitter-only vs shimmer-only vs all.
- Add the larger Sakar dataset (252 subjects) as an independent external test set.
- Implement hand/finger movement analysis via computer vision (e.g., OpenCV, MediaPipe) on spiral drawings and handwriting.

## Known limitations (state these honestly)
- Only 31 subjects (23 PD) in the voice model, so metrics have wide uncertainty.
- Dataset features came from a different recorder than a phone mic; NHR is approximated from HNR.
- Healthy-vs-PD classification is not the same as detecting change within one person; the baseline module is a monitoring heuristic and needs real longitudinal data to validate.