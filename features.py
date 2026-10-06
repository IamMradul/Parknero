"""Praat-based voice features matching the first 16 columns of the UCI Parkinson's dataset.
Training and inference use the SAME extractor family, so the app is consistent with the model."""
import numpy as np
import parselmouth
from parselmouth.praat import call

FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]


def extract_features(path, f0min=75, f0max=500):
    """Return a dict of FEATURES from a sustained-vowel .wav file (e.g. 'aaah' for 3-5 s)."""
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 10:
        raise ValueError("Could not detect a stable voiced signal. Record a sustained vowel.")

    pp = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
    jit = lambda n: call(pp, n, 0, 0, 0.0001, 0.02, 1.3)
    shim = lambda n: call([snd, pp], n, 0, 0, 0.0001, 0.02, 1.3, 1.6)

    hnr = call(snd.to_harmonicity_cc(0.01, f0min, 0.1, 1.0), "Get mean", 0, 0)
    return {
        "MDVP:Fo(Hz)": float(np.mean(f0)), "MDVP:Fhi(Hz)": float(np.max(f0)), "MDVP:Flo(Hz)": float(np.min(f0)),
        "MDVP:Jitter(%)": jit("Get jitter (local)"),
        "MDVP:Jitter(Abs)": jit("Get jitter (local, absolute)"),
        "MDVP:RAP": jit("Get jitter (rap)"),
        "MDVP:PPQ": jit("Get jitter (ppq5)"),
        "Jitter:DDP": jit("Get jitter (ddp)"),
        "MDVP:Shimmer": shim("Get shimmer (local)"),
        "MDVP:Shimmer(dB)": shim("Get shimmer (local_dB)"),
        "Shimmer:APQ3": shim("Get shimmer (apq3)"),
        "Shimmer:APQ5": shim("Get shimmer (apq5)"),
        "MDVP:APQ": shim("Get shimmer (apq11)"),
        "Shimmer:DDA": shim("Get shimmer (dda)"),
        "NHR": float(10 ** (-hnr / 10)),  # approximation from HNR (dB)
        "HNR": float(hnr),
    }