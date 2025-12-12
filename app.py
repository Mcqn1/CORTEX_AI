import streamlit as st
import joblib
import numpy as np
import mne
from scipy.signal import stft
from scipy.stats import kurtosis, skew
import os
import warnings

print("[DEBUG] Streamlit app starting...", flush=True)

# --- Page Config ---
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)
warnings.filterwarnings("ignore")

# --- Constants ---
TARGET_SFREQ = 256  # Hz
MODEL_DIR = "UTIL_DYNAMIC"
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_PATH = os.path.join(MODEL_DIR, "common_channels.txt")

print(f"[DEBUG] MODEL_DIR={MODEL_DIR}", flush=True)
print(f"[DEBUG] MODEL_PATH={MODEL_PATH}", flush=True)
print(f"[DEBUG] SCALER_PATH={SCALER_PATH}", flush=True)
print(f"[DEBUG] CHANNELS_PATH={CHANNELS_PATH}", flush=True)

# --- Feature Extraction ---
@st.cache_data
def compute_features(epoch_data, sfreq):
    """Computes features for a single EEG epoch."""
    print(f"[DEBUG] compute_features called. epoch_data.shape={epoch_data.shape}, sfreq={sfreq}", flush=True)
    feats = []
    for idx, ch_data in enumerate(epoch_data):
        mean, std = np.mean(ch_data), np.std(ch_data)
        sk, ku = skew(ch_data), kurtosis(ch_data)
        print(f"[DEBUG] Channel {idx}: mean={mean}, std={std}, skew={sk}, kurtosis={ku}", flush=True)
        # OPTIMIZED: use a shorter STFT window for speed
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq // 2))
        Pxx = np.abs(Zxx)**2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[(f >= lo) & (f <= hi), :]) for (lo, hi) in bands.values()]
        print(f"[DEBUG] Channel {idx}: band powers={band_powers}", flush=True)
        feats.extend([mean, std, sk, ku, *band_powers])
    print(f"[DEBUG] Feature vector length={len(feats)}", flush=True)
    return np.array(feats)

@st.cache_data
def load_and_standardize_raw(edf_path, target_sfreq, preload=True):
    print(f"[DEBUG] load_and_standardize_raw called. edf_path={edf_path}, target_sfreq={target_sfreq}, preload={preload}", flush=True)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose='ERROR')
        print("[DEBUG] EDF file loaded.", flush=True)
        print(f"[DEBUG] Original sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)
        raw.rename_channels(lambda ch: ch.strip().upper())
        print(f"[DEBUG] Channels after rename (first 10)={raw.ch_names[:10]}", flush=True)
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG'])
        print(f"[DEBUG] Channels after pick_types(eeg=True) (first 10)={raw.ch_names[:10]}, total={len(raw.ch_names)}", flush=True)
        if raw.info['sfreq'] != target_sfreq:
            print(f"[DEBUG] Resampling from {raw.info['sfreq']} Hz to {target_sfreq} Hz", flush=True)
            raw.resample(target_sfreq, verbose='ERROR')
        print("[DEBUG] load_and_standardize_raw completed successfully.", flush=True)
        return raw
    except Exception as e:
        print(f"[DEBUG] ERROR in load_and_standardize_raw: {e}", flush=True)
        st.error(f"Error loading EDF file: {e}")
        return None

# --- Load Model Assets (YOUR FIXED CODE) ---
@st.cache_resource
def load_model_assets():
    print("[DEBUG] load_model_assets called.", flush=True)
    try:
        model = joblib.load(MODEL_PATH)
        print("[DEBUG] Model loaded from disk.", flush=True)
        scaler = joblib.load(SCALER_PATH)
        print("[DEBUG] Scaler loaded from disk.", flush=True)
    except FileNotFoundError:
        print("[DEBUG] FileNotFoundError: model or scaler missing.", flush=True)
        st.error(" Model or scaler missing in UTIL_DYNAMIC. Please run training first.")
        return None, None, None
    except Exception as e:
        print(f"[DEBUG] ERROR loading model/scaler: {e}", flush=True)
        st.error(f" Error loading model or scaler: {e}")
        return None, None, None

    try:
        with open(CHANNELS_PATH, 'r') as f:
            common_channels = [line.strip() for line in f.readlines()]
        print(f"[DEBUG] Loaded {len(common_channels)} common channels from file.", flush=True)
    except FileNotFoundError:
        print("[DEBUG] FileNotFoundError: channel list missing.", flush=True)
        st.error(" Channel list missing. Please retrain model.")
        return None, None, None
    except Exception as e:
        print(f"[DEBUG] ERROR loading channel list: {e}", flush=True)
        st.error(f" Error loading channel list: {e}")
        return None, None, None

    print("[DEBUG] load_model_assets completed successfully.", flush=True)
    return model, scaler, common_channels

model, scaler, common_channels = load_model_assets()
print(f"[DEBUG] model is None? {model is None}", flush=True)
print(f"[DEBUG] scaler is None? {scaler is None}", flush=True)
print(f"[DEBUG] common_channels is None? {common_channels is None}", flush=True)

# --- Streamlit App UI ---
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
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        print("[DEBUG] Saved uploaded file as temp.edf", flush=True)
        
        # 1. Load File
        raw = load_and_standardize_raw("temp.edf", TARGET_SFREQ)
        
        if raw is None:
            print("[DEBUG] raw is None after load_and_standardize_raw. Showing error and returning.", flush=True)
            st.error("Could not process the uploaded EDF file.")
        else:
            print(f"[DEBUG] Raw loaded: sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)
            # 2. Pick Channels
            try:
                print("[DEBUG] Picking common_channels from raw.", flush=True)
                print(f"[DEBUG] Expected channels count={len(common_channels)}. First 10={common_channels[:10]}", flush=True)
                raw.pick(common_channels, verbose='ERROR')
                print(f"[DEBUG] After pick: raw.ch_names (first 10)={raw.ch_names[:10]}, total={len(raw.ch_names)}", flush=True)
            except ValueError as e:
                print(f"[DEBUG] ValueError in raw.pick: {e}", flush=True)
                st.error("Missing required channels.")
                os.remove("temp.edf")
                print("[DEBUG] temp.edf removed after missing channels.", flush=True)
                st.stop()
            
            # 3. Filter
            print("[DEBUG] Applying bandpass filter 0.5–48 Hz and setting average reference.", flush=True)
            raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            print("[DEBUG] Filtering & referencing completed.", flush=True)
            
            # 4. Epochs
            print("[DEBUG] Creating fixed-length events (duration=10s).", flush=True)
            events = mne.make_fixed_length_events(raw, duration=10.0)
            print(f"[DEBUG] Number of events created={len(events)}", flush=True)

            epochs = mne.Epochs(
                raw, events, tmin=0,
                tmax=10.0 - 1/TARGET_SFREQ,
                preload=True, baseline=None, verbose="ERROR"
            )
            print(f"[DEBUG] Number of epochs={len(epochs.events)}", flush=True)

            # --- NEW: Limit max epochs for speed ---
            MAX_EPOCHS = 500  # You can tune this
            if len(epochs.events) > MAX_EPOCHS:
                print(f"[DEBUG] Limiting epochs from {len(epochs.events)} to {MAX_EPOCHS} for faster inference.", flush=True)
                epochs = epochs[:MAX_EPOCHS]
                print(f"[DEBUG] After limiting, epochs={len(epochs.events)}", flush=True)
            # --- END NEW BLOCK ---

            if len(epochs.events) == 0:
                print("[DEBUG] No epochs found. Showing warning.", flush=True)
                st.warning("No valid epochs found.")
            else:
                # 5. Extract Features
                print(f"[DEBUG] Extracting features from {len(epochs.events)} epochs...", flush=True)
                X_features = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
                print(f"[DEBUG] Feature matrix shape={X_features.shape}", flush=True)
                
                # 6. Predict (USING SCALER)
                print("[DEBUG] Scaling features using loaded scaler.", flush=True)
                X_scaled = scaler.transform(X_features)
                print("[DEBUG] Running model.predict and model.predict_proba.", flush=True)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                print(f"[DEBUG] Predictions shape={predictions.shape}, unique={np.unique(predictions, return_counts=True)}", flush=True)
                print(f"[DEBUG] Probabilities shape={probabilities.shape}", flush=True)
                
                # 7. Results
                seizure_epochs = np.sum(predictions)
                print(f"[DEBUG] seizure_epochs={seizure_epochs}", flush=True)
                if seizure_epochs > 0:
                    st.error(f"**Seizure detected in {seizure_epochs} windows.**")
                else:
                    st.success("**No seizure detected.**")
                
                # Table
                print("[DEBUG] Building results table for each window.", flush=True)
                results = []
                for i, prob in enumerate(probabilities):
                    status = "Seizure" if prob > 0.5 else "Normal"
                    results.append({
                        "Time": f"{i*10}s - {(i+1)*10}s", 
                        "Status": status, 
                        "Prob": f"{prob*100:.2f}%"
                    })
                    print(f"[DEBUG] Window {i}: Time={i*10}-{(i+1)*10}s, Status={status}, Prob={prob*100:.2f}%", flush=True)
                st.dataframe(results, use_container_width=True)
                
        os.remove("temp.edf")
        print("[DEBUG] temp.edf removed at end of processing.", flush=True)
