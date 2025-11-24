import streamlit as st
import joblib
import numpy as np
import mne
from scipy.signal import stft
from scipy.stats import kurtosis, skew
import os
import warnings

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

# --- Feature Extraction ---
@st.cache_data
def compute_features(epoch_data, sfreq):
    """Computes features for a single EEG epoch."""
    feats = []
    for ch_data in epoch_data:
        mean, std = np.mean(ch_data), np.std(ch_data)
        sk, ku = skew(ch_data), kurtosis(ch_data)
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq))
        Pxx = np.abs(Zxx)**2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[(f >= lo) & (f <= hi), :]) for (lo, hi) in bands.values()]
        feats.extend([mean, std, sk, ku, *band_powers])
    return np.array(feats)

@st.cache_data
def load_and_standardize_raw(edf_path, target_sfreq, preload=True):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose='ERROR')
        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG'])
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception as e:
        st.error(f"Error loading EDF file: {e}")
        return None

# --- Load Model Assets (YOUR FIXED CODE) ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error("❌ Model or scaler missing in UTIL_DYNAMIC. Please run training first.")
        return None, None, None

    try:
        with open(CHANNELS_PATH, 'r') as f:
            common_channels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("❌ Channel list missing. Please retrain model.")
        return None, None, None

    return model, scaler, common_channels

model, scaler, common_channels = load_model_assets()

# --- Streamlit App UI ---
st.title("🧠 EEG Seizure Detection")
st.write("Upload an EDF file to check for seizure activity.")

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if uploaded_file is not None and model is not None and scaler is not None and common_channels is not None:
    
    with st.spinner("Processing EEG file..."):
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 1. Load File
        raw = load_and_standardize_raw("temp.edf", TARGET_SFREQ)
        
        if raw is None:
            st.error("Could not process the uploaded EDF file.")
        else:
            # 2. Pick Channels
            try:
                raw.pick(common_channels, verbose='ERROR')
            except ValueError:
                st.error("Missing required channels.")
                os.remove("temp.edf")
                st.stop()
            
            # 3. Filter
            raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            
            # 4. Epochs
            events = mne.make_fixed_length_events(raw, duration=5.0)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=5.0 - 1/TARGET_SFREQ, preload=True, baseline=None, verbose="ERROR")
            
            if len(epochs.events) == 0:
                st.warning("No valid epochs found.")
            else:
                # 5. Extract Features
                st.write(f"Extracting features from {len(epochs.events)} windows...")
                X_features = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
                
                # 6. Predict (USING SCALER)
                X_scaled = scaler.transform(X_features)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                # 7. Results
                seizure_epochs = np.sum(predictions)
                if seizure_epochs > 0:
                    st.error(f"**Seizure detected in {seizure_epochs} windows.**")
                else:
                    st.success("**No seizure detected.**")
                
                # Table
                results = []
                for i, prob in enumerate(probabilities):
                    results.append({
                        "Time": f"{i*5}s - {(i+1)*5}s", 
                        "Status": "Seizure" if prob > 0.5 else "Normal", 
                        "Prob": f"{prob*100:.2f}%"
                    })
                st.dataframe(results, use_container_width=True)
                
        os.remove("temp.edf")