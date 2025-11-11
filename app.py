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
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_PATH = os.path.join(MODEL_DIR, "common_channels.txt")

# --- Feature Extraction (Copied from your training script) ---
# We need these to process the new, uploaded EDF file
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
    """Loads and standardizes a single EDF file."""
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

# --- Load Model and Channels ---
@st.cache_resource
def load_model_assets():
    """Loads the model, scaler, and channels list."""
    try:
        # The model .pkl file contains the full pipeline (scaler + classifier)
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Did you push the UTIL_DYNAMIC folder?")
        return None, None
    
    try:
        with open(CHANNELS_PATH, 'r') as f:
            common_channels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error(f"Error: Channels file not found at {CHANNELS_PATH}. Did you push the UTIL_DYNAMIC folder?")
        return None, None
        
    return model, common_channels

model, common_channels = load_model_assets()

# --- Streamlit App UI ---
st.title("🧠 EEG Seizure Detection")
st.write("Upload an EDF file to check for seizure activity. This app will process the file in 5-second windows and predict the probability of a seizure for each window.")

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if uploaded_file is not None and model is not None and common_channels is not None:
    
    with st.spinner("Processing EEG file... This may take a moment."):
        # Save temp file to be read by MNE
        with open("temp.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 1. Load and Standardize File
        raw = load_and_standardize_raw("temp.edf", TARGET_SFREQ)
        
        if raw is None:
            st.error("Could not process the uploaded EDF file.")
        else:
            # 2. Pick Common Channels
            try:
                raw.pick(common_channels, verbose='ERROR')
            except ValueError:
                st.error(f"The uploaded EDF file is missing one or more required channels. This model was trained on: {', '.join(common_channels)}")
                os.remove("temp.edf") # Clean up
                st.stop()
            
            # 3. Filter and Reference
            raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            
            # 4. Create Epochs
            events = mne.make_fixed_length_events(raw, duration=5.0)
            epochs = mne.Epochs(
                raw,
                events,
                tmin=0,
                tmax=5.0 - 1 / TARGET_SFREQ,
                preload=True, # Need to preload for feature extraction
                baseline=None,
                verbose="ERROR",
            )
            
            if len(epochs.events) == 0:
                st.warning("No valid 5-second epochs could be extracted from this file.")
            else:
                # 5. Extract Features
                st.write(f"Extracting features from {len(epochs.events)} 5-second windows...")
                X_features = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
                
                # 6. Get Predictions
                predictions = model.predict(X_features)
                probabilities = model.predict_proba(X_features)[:, 1] # Get probability of "seizure" class
                
                # 7. Display Results
                st.subheader("Prediction Results")
                
                seizure_epochs = np.sum(predictions)
                if seizure_epochs > 0:
                    st.error(f"**Seizure activity detected in {seizure_epochs} out of {len(epochs.events)} windows.**")
                else:
                    st.success("**No seizure activity detected.**")
                
                st.write("Detailed 5-second window predictions:")
                
                results = []
                for i, prob in enumerate(probabilities):
                    window_time = f"{i*5}s - {(i+1)*5}s"
                    status = "Seizure" if prob > 0.5 else "Normal"
                    results.append({"Time Window": window_time, "Status": status, "Seizure Probability": f"{prob*100:.2f}%"})
                
                st.dataframe(results, use_container_width=True)
                
        # Clean up temp file
        os.remove("temp.edf")

elif model is None or common_channels is None:
    st.error("Model assets are missing. Please check the repository and ensure the `UTIL_DYNAMIC` folder is present.")