import os
import gc
import uuid
import warnings

import streamlit as st
import joblib
import numpy as np
import mne
from scipy.signal import stft, butter, sosfiltfilt
from scipy.stats import kurtosis, skew

print("[DEBUG] Streamlit app starting...", flush=True)

# --- Page Config ---
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)
warnings.filterwarnings("ignore")

# --- Constants ---
TARGET_SFREQ = 256  # model expects 256Hz (CHB-MIT is usually 256)
MODEL_DIR = "UTIL_DYNAMIC"
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_PATH = os.path.join(MODEL_DIR, "common_channels.txt")

WIN_SEC = 10.0
STEP_SEC = 10.0
DEFAULT_MAX_WINDOWS = 500  # 500 windows * 10s = ~83 min

# ----------------------------
# Feature extraction (no cache)
# ----------------------------
def compute_features(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    window_data: shape (n_channels, n_samples)
    returns: 1D feature vector (float32)
    """
    window_data = np.asarray(window_data)
    if np.iscomplexobj(window_data):
        window_data = np.real(window_data)
    window_data = window_data.astype(np.float32, copy=False)

    feats = []
    for ch_data in window_data:
        ch_data = np.asarray(ch_data)
        if np.iscomplexobj(ch_data):
            ch_data = np.real(ch_data)
        ch_data = ch_data.astype(np.float32, copy=False)

        mean = float(np.mean(ch_data))
        std = float(np.std(ch_data))
        sk = float(skew(ch_data))
        ku = float(kurtosis(ch_data))

        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq // 2))
        Pxx = np.abs(Zxx) ** 2

        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
        }
        band_powers = [float(np.sum(Pxx[(f >= lo) & (f <= hi), :])) for (lo, hi) in bands.values()]

        feats.extend([mean, std, sk, ku, *band_powers])

    return np.asarray(feats, dtype=np.float32)

# ----------------------------
# Robust per-window preprocessing (SciPy SOS)
# ----------------------------
def _make_bandpass_sos(sfreq: float, l_freq=0.5, h_freq=48.0, order=4):
    nyq = 0.5 * sfreq
    low = max(l_freq / nyq, 1e-6)
    high = min(h_freq / nyq, 0.999999)
    return butter(order, [low, high], btype="bandpass", output="sos")

def preprocess_window(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Constant-memory window preprocessing:
    - force real float64 contiguous
    - bandpass using SciPy SOS (avoids MNE dtype checks)
    - average reference
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.real(x)

    # Force float64 contiguous always
    x = np.ascontiguousarray(x, dtype=np.float64)

    # Bandpass (SciPy)
    sos = _make_bandpass_sos(sfreq, 0.5, 48.0, order=4)
    x = sosfiltfilt(sos, x, axis=1)

    # Average reference
    x = x - x.mean(axis=0, keepdims=True)
    return x

# ----------------------------
# Window iterator (streaming)
# ----------------------------
def iter_fixed_windows(raw: mne.io.BaseRaw, win_sec: float, step_sec: float, max_windows: int):
    sfreq = float(raw.info["sfreq"])
    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    n_times = raw.n_times

    produced = 0
    for start in range(0, n_times - win + 1, step):
        stop = start + win
        x = raw.get_data(start=start, stop=stop)  # only this chunk in RAM
        yield x, start / sfreq, stop / sfreq
        produced += 1
        if produced >= max_windows:
            break

# ----------------------------
# Load EDF (memory safe)
# ----------------------------
def load_and_standardize_raw(edf_path: str) -> mne.io.BaseRaw | None:
    print(f"[DEBUG] load_and_standardize_raw: {edf_path}", flush=True)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=["EKG", "ECG"])
        print(f"[DEBUG] sfreq={raw.info['sfreq']}, ch={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)
        return raw
    except Exception as e:
        print(f"[DEBUG] ERROR reading EDF: {e}", flush=True)
        st.error(f"Error loading EDF: {e}")
        return None

# ----------------------------
# Load assets once
# ----------------------------
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Model/scaler load error: {e}")
        return None, None, None

    try:
        with open(CHANNELS_PATH, "r") as f:
            common_channels = [line.strip().upper() for line in f if line.strip()]
    except Exception as e:
        st.error(f"Channels list load error: {e}")
        return None, None, None

    print(f"[DEBUG] Loaded {len(common_channels)} common channels.", flush=True)
    return model, scaler, common_channels

# ----------------------------
# UI
# ----------------------------
st.title("🧠 EEG Seizure Detection")
st.write("Upload an EDF file to check for seizure activity.")

model, scaler, common_channels = load_model_assets()

max_windows = st.slider(
    "Max windows to process (10s each)",
    min_value=50,
    max_value=1500,
    value=DEFAULT_MAX_WINDOWS,
    step=50,
)

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if "running" not in st.session_state:
    st.session_state.running = False
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

def run_inference(edf_temp_path: str):
    raw = load_and_standardize_raw(edf_temp_path)
    if raw is None:
        return

    # Check channels
    available = set(raw.ch_names)
    required = set(common_channels)
    missing = sorted(list(required - available))
    if missing:
        st.error(f"Missing required channels ({len(missing)}): {missing}")
        return

    # IMPORTANT: no ordered=True (compat)
    raw.pick(common_channels, verbose="ERROR")
    sfreq_native = float(raw.info["sfreq"])

    # crop to reduce reads
    max_sec = max_windows * WIN_SEC
    try:
        raw.crop(tmin=0, tmax=max_sec, include_tmax=False)
    except Exception:
        pass

    progress = st.progress(0.0)
    status = st.empty()

    predictions = []
    results = []

    for i, (x, t0, t1) in enumerate(iter_fixed_windows(raw, WIN_SEC, STEP_SEC, max_windows), start=1):
        status.write(f"Processing window {i}/{max_windows} ({int(t0)}s–{int(t1)}s)")

        x = preprocess_window(x, sfreq_native)

        feats = compute_features(x, sfreq_native).reshape(1, -1)
        X_scaled = scaler.transform(feats)

        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0, 1])

        predictions.append(pred)
        results.append({
            "Time": f"{int(t0)}s - {int(t1)}s",
            "Status": "Seizure" if prob > 0.5 else "Normal",
            "Prob": f"{prob * 100:.2f}%"
        })

        del x, feats, X_scaled
        if i % 25 == 0:
            gc.collect()

        progress.progress(min(i / max_windows, 1.0))

    status.empty()
    progress.empty()

    seizure_windows = int(np.sum(predictions))
    if seizure_windows > 0:
        st.error(f"**Seizure detected in {seizure_windows} windows.**")
    else:
        st.success("**No seizure detected.**")

    st.dataframe(results, use_container_width=True)
    del raw
    gc.collect()

# ----------------------------
# Run
# ----------------------------
if uploaded_file is not None and model is not None and scaler is not None and common_channels is not None:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.running:
        st.info("Processing already running...")
    else:
        if st.session_state.last_file_id != file_id:
            st.session_state.last_file_id = file_id
            st.session_state.running = True

            temp_name = None
            try:
                with st.spinner("Processing EEG file..."):
                    temp_name = f"temp_{uuid.uuid4().hex}.edf"
                    with open(temp_name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    print(f"[DEBUG] Saved upload as {temp_name}", flush=True)
                    run_inference(temp_name)

            finally:
                try:
                    if temp_name and os.path.exists(temp_name):
                        os.remove(temp_name)
                        print(f"[DEBUG] Removed {temp_name}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Temp cleanup failed: {e}", flush=True)

                st.session_state.running = False
                gc.collect()
