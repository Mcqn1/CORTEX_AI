import streamlit as st
import joblib
import numpy as np
import mne
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from mne.filter import filter_data
import os
import warnings
import gc

print("[DEBUG] Streamlit app starting...", flush=True)

# --- Page Config ---
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)
warnings.filterwarnings("ignore")

# --- Constants ---
TARGET_SFREQ = 256  # Hz (your model expects this)
MODEL_DIR = "UTIL_DYNAMIC"
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_PATH = os.path.join(MODEL_DIR, "common_channels.txt")

print(f"[DEBUG] MODEL_DIR={MODEL_DIR}", flush=True)
print(f"[DEBUG] MODEL_PATH={MODEL_PATH}", flush=True)
print(f"[DEBUG] SCALER_PATH={SCALER_PATH}", flush=True)
print(f"[DEBUG] CHANNELS_PATH={CHANNELS_PATH}", flush=True)

# --- Feature Extraction ---
# IMPORTANT: Do NOT cache this. Streamlit caching huge numpy arrays can explode memory.
def compute_features(epoch_data, sfreq):
    """Computes features for a single EEG window (n_channels, n_samples)."""
    # Light debug only
    print(f"[DEBUG] compute_features called. epoch_data.shape={epoch_data.shape}, sfreq={sfreq}", flush=True)

    feats = []
    for ch_data in epoch_data:
        # Ensure float32 for less memory + decent speed
        ch_data = ch_data.astype(np.float32, copy=False)

        mean, std = float(np.mean(ch_data)), float(np.std(ch_data))
        sk, ku = float(skew(ch_data)), float(kurtosis(ch_data))

        # Short STFT window for speed
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq // 2))
        Pxx = np.abs(Zxx) ** 2

        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30)
        }
        band_powers = [float(np.sum(Pxx[(f >= lo) & (f <= hi), :])) for (lo, hi) in bands.values()]
        feats.extend([mean, std, sk, ku, *band_powers])

    return np.asarray(feats, dtype=np.float32)

def load_and_standardize_raw(edf_path, target_sfreq):
    """
    MEMORY SAFE:
    - preload=False: don't load all samples into RAM
    - pick channels early
    - avoid full-file filtering/epoching
    """
    print(f"[DEBUG] load_and_standardize_raw called. edf_path={edf_path}, target_sfreq={target_sfreq}", flush=True)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        print("[DEBUG] EDF header loaded (preload=False).", flush=True)
        print(f"[DEBUG] Original sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)

        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=["EKG", "ECG"])

        print(f"[DEBUG] After pick_types(eeg=True): total_channels={len(raw.ch_names)}", flush=True)

        # If sfreq != target_sfreq, resampling the entire file would be memory-heavy.
        # We'll keep native sfreq and handle mismatch gracefully (or crop+resample if needed).
        return raw

    except Exception as e:
        print(f"[DEBUG] ERROR in load_and_standardize_raw: {e}", flush=True)
        st.error(f"Error loading EDF file: {e}")
        return None

@st.cache_resource
def load_model_assets():
    print("[DEBUG] load_model_assets called.", flush=True)
    try:
        model = joblib.load(MODEL_PATH)
        print("[DEBUG] Model loaded from disk.", flush=True)
        scaler = joblib.load(SCALER_PATH)
        print("[DEBUG] Scaler loaded from disk.", flush=True)
    except FileNotFoundError:
        st.error("Model or scaler missing in UTIL_DYNAMIC. Please run training first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

    try:
        with open(CHANNELS_PATH, "r") as f:
            common_channels = [line.strip().upper() for line in f.readlines() if line.strip()]
        print(f"[DEBUG] Loaded {len(common_channels)} common channels.", flush=True)
    except FileNotFoundError:
        st.error("Channel list missing. Please retrain model.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading channel list: {e}")
        return None, None, None

    return model, scaler, common_channels

def iter_fixed_windows(raw, win_sec=10.0, step_sec=10.0, max_windows=500):
    """Yield (x, t_start, t_stop) where x has shape (n_ch, win_samples)."""
    sfreq = float(raw.info["sfreq"])
    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)

    n_times = raw.n_times
    produced = 0

    for start in range(0, n_times - win + 1, step):
        stop = start + win
        x = raw.get_data(start=start, stop=stop)  # loads ONLY this slice into RAM
        yield x, start / sfreq, stop / sfreq
        produced += 1
        if produced >= max_windows:
            break

def preprocess_window(x, sfreq):
    """
    Per-window preprocessing (constant memory):
    - float32
    - bandpass (0.5–48)
    - avg reference
    """
    x = x.astype(np.float32, copy=False)

    # Bandpass per-window (instead of raw.filter on full file)
    x = filter_data(
        x, sfreq=sfreq, l_freq=0.5, h_freq=48.0,
        method="fir", verbose="ERROR"
    )

    # Average reference (per-window)
    x = x - x.mean(axis=0, keepdims=True)
    return x

# --- Load Assets ---
model, scaler, common_channels = load_model_assets()
print(f"[DEBUG] model is None? {model is None}", flush=True)
print(f"[DEBUG] scaler is None? {scaler is None}", flush=True)
print(f"[DEBUG] common_channels is None? {common_channels is None}", flush=True)

# --- UI ---
st.title("🧠 EEG Seizure Detection")
st.write("Upload an EDF file to check for seizure activity.")

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if uploaded_file is not None:
    print(f"[DEBUG] Uploaded file: name={uploaded_file.name}, size={len(uploaded_file.getbuffer())} bytes", flush=True)
else:
    print("[DEBUG] No file uploaded yet.", flush=True)

if uploaded_file is not None and model is not None and scaler is not None and common_channels is not None:
    with st.spinner("Processing EEG file..."):
        print("[DEBUG] Entered main processing block.", flush=True)

        # Save upload to disk (avoid keeping all bytes in memory)
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        print("[DEBUG] Saved uploaded file as temp.edf", flush=True)

        raw = load_and_standardize_raw("temp.edf", TARGET_SFREQ)

        if raw is None:
            st.error("Could not process the uploaded EDF file.")
        else:
            # Pick common channels early (still cheap with preload=False)
            try:
                print("[DEBUG] Picking common_channels from raw.", flush=True)
                raw.pick(common_channels, ordered=True, verbose="ERROR")
                print(f"[DEBUG] After pick: total_channels={len(raw.ch_names)}", flush=True)
            except Exception as e:
                st.error(f"Missing required channels or pick failed: {e}")
                try:
                    os.remove("temp.edf")
                except Exception:
                    pass
                st.stop()

            # Optional: crop to first MAX_EPOCHS * 10 seconds to reduce reads
            MAX_EPOCHS = 500
            WIN_SEC = 10.0
            STEP_SEC = 10.0
            max_sec = MAX_EPOCHS * WIN_SEC
            try:
                raw.crop(tmin=0, tmax=max_sec, include_tmax=False)
                print(f"[DEBUG] Cropped raw to first ~{max_sec}s for efficiency.", flush=True)
            except Exception as e:
                print(f"[DEBUG] Crop skipped/failed: {e}", flush=True)

            sfreq_native = float(raw.info["sfreq"])
            print(f"[DEBUG] Using native sfreq={sfreq_native}", flush=True)

            predictions = []
            probabilities = []
            results = []

            # STREAM WINDOWS instead of Epochs(preload=True)
            for i, (x, t_start, t_stop) in enumerate(
                iter_fixed_windows(raw, win_sec=WIN_SEC, step_sec=STEP_SEC, max_windows=MAX_EPOCHS)
            ):
                # per-window preprocessing
                x = preprocess_window(x, sfreq_native)

                # If your sfreq isn't 256, your features differ from training.
                # Ideally ensure sfreq=256 at data level. Most CHB-MIT is 256 anyway.
                feats = compute_features(x, sfreq_native).reshape(1, -1)
                X_scaled = scaler.transform(feats)

                pred = int(model.predict(X_scaled)[0])
                prob = float(model.predict_proba(X_scaled)[0, 1])

                predictions.append(pred)
                probabilities.append(prob)

                status = "Seizure" if prob > 0.5 else "Normal"
                results.append({
                    "Time": f"{int(t_start)}s - {int(t_stop)}s",
                    "Status": status,
                    "Prob": f"{prob * 100:.2f}%"
                })

                # Free per-loop memory
                del x, feats, X_scaled
                if (i + 1) % 50 == 0:
                    gc.collect()

            seizure_epochs = int(np.sum(predictions))
            if seizure_epochs > 0:
                st.error(f"**Seizure detected in {seizure_epochs} windows.**")
            else:
                st.success("**No seizure detected.**")

            st.dataframe(results, use_container_width=True)

        # cleanup
        try:
            os.remove("temp.edf")
            print("[DEBUG] temp.edf removed at end of processing.", flush=True)
        except Exception as e:
            print(f"[DEBUG] Failed to remove temp.edf: {e}", flush=True)

        gc.collect()
