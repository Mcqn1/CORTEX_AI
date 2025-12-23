import os
import gc
import uuid
import warnings

import streamlit as st
import joblib
import numpy as np
import mne
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from mne.filter import filter_data

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
DEFAULT_MAX_WINDOWS = 500  # 500 windows * 10s = 5000s (~83 min)


# ---------------------------------------------------------------------
# Feature extraction (DON'T cache — avoids caching large arrays)
# ---------------------------------------------------------------------
def compute_features(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    window_data: shape (n_channels, n_samples)
    returns: 1D feature vector
    """
    # safe conversion for stats/STFT
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

        # short STFT window for speed
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


def preprocess_window(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Per-window preprocessing (constant memory):
    - force real float64 + contiguous (MNE filter requirement)
    - bandpass (0.5–48)
    - avg reference
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.real(x)

    # MNE filter requires real floating; float64 is safest
    x = np.ascontiguousarray(x, dtype=np.float64)

    # Debug (optional): uncomment if you want to confirm dtype
    # print("[DEBUG] window dtype:", x.dtype, "shape:", x.shape, flush=True)

    x = filter_data(
        x,
        sfreq=sfreq,
        l_freq=0.5,
        h_freq=48.0,
        method="fir",
        verbose="ERROR",
    )

    # average reference per-window
    x = x - x.mean(axis=0, keepdims=True)
    return x


def iter_fixed_windows(raw: mne.io.BaseRaw, win_sec: float, step_sec: float, max_windows: int):
    """Yield (x, t_start, t_stop) where x has shape (n_ch, win_samples)."""
    sfreq = float(raw.info["sfreq"])
    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    n_times = raw.n_times

    produced = 0
    for start in range(0, n_times - win + 1, step):
        stop = start + win
        x = raw.get_data(start=start, stop=stop)  # loads ONLY this slice
        yield x, start / sfreq, stop / sfreq
        produced += 1
        if produced >= max_windows:
            break


def load_and_standardize_raw(edf_path: str) -> mne.io.BaseRaw | None:
    """
    Memory-safe load:
    - preload=False keeps RAM low
    """
    print(f"[DEBUG] load_and_standardize_raw: edf_path={edf_path}", flush=True)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        print("[DEBUG] EDF header loaded (preload=False).", flush=True)

        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=["EKG", "ECG"])

        print(f"[DEBUG] sfreq={raw.info['sfreq']}, ch={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)
        return raw

    except Exception as e:
        print(f"[DEBUG] ERROR reading EDF: {e}", flush=True)
        st.error(f"Error loading EDF: {e}")
        return None


@st.cache_resource
def load_model_assets():
    """Load model/scaler/channels once per app instance."""
    print("[DEBUG] load_model_assets called.", flush=True)

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error("Model or scaler missing in UTIL_DYNAMIC. Please run training first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None, None

    try:
        with open(CHANNELS_PATH, "r") as f:
            common_channels = [line.strip().upper() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("common_channels.txt missing. Please retrain model.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading channels list: {e}")
        return None, None, None

    print(f"[DEBUG] Loaded channels={len(common_channels)}", flush=True)
    return model, scaler, common_channels


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🧠 EEG Seizure Detection")
st.write("Upload an EDF file to check for seizure activity.")

model, scaler, common_channels = load_model_assets()

max_windows = st.slider(
    "Max windows to process (10s each)",
    min_value=50,
    max_value=1500,
    value=100,
    step=50,
    help="Higher = longer analysis, slower. 500 windows = ~83 minutes of EEG.",
)

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

# session guard: prevents rerun loops
if "running" not in st.session_state:
    st.session_state.running = False
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None


def run_inference(edf_temp_path: str):
    raw = load_and_standardize_raw(edf_temp_path)
    if raw is None:
        return

    # Validate channels BEFORE pick
    available = set(raw.ch_names)
    required = set(common_channels)
    missing = sorted(list(required - available))
    if missing:
        st.error(f"Missing required channels ({len(missing)}): {missing}")
        return

    # Compatibility: no ordered=True
    raw.pick(common_channels, verbose="ERROR")
    sfreq_native = float(raw.info["sfreq"])

    # Optional crop for efficiency: max_windows * 10 seconds
    max_sec = max_windows * WIN_SEC
    try:
        raw.crop(tmin=0, tmax=max_sec, include_tmax=False)
        print(f"[DEBUG] Cropped to first ~{max_sec}s", flush=True)
    except Exception as e:
        print(f"[DEBUG] Crop skipped: {e}", flush=True)

    progress = st.progress(0.0)
    status = st.empty()

    predictions = []
    results = []

    for i, (x, t0, t1) in enumerate(iter_fixed_windows(raw, WIN_SEC, STEP_SEC, max_windows=max_windows), start=1):
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
                    print(f"[DEBUG] Failed to remove temp file: {e}", flush=True)

                st.session_state.running = False
                gc.collect()
