import os
import glob
import warnings
import numpy as np
import mne
import joblib
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier  # <-- Use this model
from datetime import datetime

# --- Basic setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Ensure outputs go into mounted workspace (/app) ---
MODEL_DIR = os.path.join(os.getcwd(), "UTIL_DYNAMIC")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl") # <-- Need to save scaler
CHANNELS_LIST_PATH = os.path.join(MODEL_DIR, "common_channels.txt")

# --- Constants ---
TARGET_SFREQ = 256  # Hz
BASE_DATA_PATH = "EEG_DATA" # <-- Path to DVC data

# --- Helper Functions (from your code) ---
def parse_time_to_seconds(time_str):
    if '.' in time_str or ':' in time_str:
        time_str = time_str.replace('.', ':')
        parts = list(map(int, time_str.split(':')))
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return int(time_str)

def get_seizure_annotations(summary_file, file_name):
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    file_sections = content.split('File Name:')[1:]
    for section in file_sections:
        if file_name in section:
            lines = section.strip().split('\n')
            num_seizures = 0
            for line in lines:
                if 'Number of Seizures in File' in line:
                    try:
                        num_seizures = int(line.split(':')[-1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
            if num_seizures == 0:
                return None, None, None
            starts, ends = [], []
            if 'Seizure 1 Start Time' in section:
                for i in range(1, num_seizures + 1):
                    for line in lines:
                        if f'Seizure {i} Start Time:' in line:
                            starts.append(float(line.split(':')[-1].strip().replace(' seconds', '')))
                        if f'Seizure {i} End Time:' in line:
                            ends.append(float(line.split(':')[-1].strip().replace(' seconds', '')))
            else:
                reg_start_sec = 0
                for line in lines:
                    if 'Registration start time' in line:
                        reg_start_sec = parse_time_to_seconds(line.split(': ')[1])
                        break
                for line in lines:
                    if 'Seizure start time' in line:
                        starts.append(parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec)
                    if 'Seizure end time' in line:
                        ends.append(parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec)
            onsets = np.array(starts)
            durations = np.array(ends) - onsets
            descriptions = ['seizure'] * len(onsets)
            return onsets, durations, descriptions
    return None, None, None

def load_and_standardize_raw(edf_path, target_sfreq):
    try:
        # MEMORY FIX: preload=False
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR') 
        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG'])
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception:
        return None

def discover_common_channels(base_data_path, target_sfreq):
    print("[PASS 1] Discovering common channels (this is memory-safe)...")
    patient_folders = [
        os.path.join(base_data_path, f)
        for f in os.listdir(base_data_path)
        if os.path.isdir(os.path.join(base_data_path, f))
    ]
    common_channels = None
    for patient_folder in patient_folders:
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            # We load with preload=False, so this is fast and low-memory
            raw = load_and_standardize_raw(edf_file, target_sfreq) 
            if raw:
                current_channels = set(raw.ch_names)
                if common_channels is None:
                    common_channels = current_channels
                else:
                    common_channels.intersection_update(current_channels)
    if common_channels:
        print(f"[✓] Found {len(common_channels)} common channels.")
        return sorted(list(common_channels))
    raise RuntimeError("No common channels found across EDFs.")

def compute_features(epoch_data, sfreq):
    feats = []
    for ch_data in epoch_data:
        mean, std = np.mean(ch_data), np.std(ch_data)
        sk, ku = skew(ch_data), kurtosis(ch_data)
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq))
        Pxx = np.abs(Zxx) ** 2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[(f >= lo) & (f <= hi), :]) for (lo, hi) in bands.values()]
        feats.extend([mean, std, sk, ku, *band_powers])
    return np.array(feats)

# --- Main Training ---
if __name__ == "__main__":
    
    print("Starting MLOps training pipeline (Batch Mode)...")
    start_time = datetime.now()

    # --- PASS 1: Find common channels (low memory) ---
    common_channels = discover_common_channels(BASE_DATA_PATH, TARGET_SFREQ)

    # --- Initialize Model and Scaler ---
    # We MUST use a model that supports partial_fit
    model = SGDClassifier(loss='hinge', # 'hinge' makes it a linear SVM
                        class_weight='balanced', 
                        random_state=42,
                        n_jobs=-1) 
    scaler = StandardScaler()

    # --- PASS 2: Process and train ONE PATIENT at a time (low memory) ---
    print("\n[PASS 2] Starting batch training (one patient at a time)...")
    all_classes = np.array([0, 1]) # We must tell partial_fit about all classes
    
    patient_folders = [
        os.path.join(BASE_DATA_PATH, f)
        for f in os.listdir(BASE_DATA_PATH)
        if os.path.isdir(os.path.join(BASE_DATA_PATH, f))
    ]

    for patient_folder in patient_folders:
        print(f"\n[+] Processing patient: {os.path.basename(patient_folder)}")
        summary_file = next(glob.iglob(os.path.join(patient_folder, "*.txt")), None)
        
        # Load patient files (preload=False)
        raws = [
            load_and_standardize_raw(edf_file, TARGET_SFREQ)
            for edf_file in sorted(glob.glob(os.path.join(patient_folder, "*.edf")))
        ]
        raws = [r for r in raws if r is not None and set(common_channels).issubset(set(r.ch_names))]
        
        if not raws or not summary_file:
            print("[!] Skipping patient (missing files or summary).")
            continue

        for raw in raws:
            raw.pick(common_channels, verbose='ERROR')
            onsets, durations, descriptions = get_seizure_annotations(
                summary_file, os.path.basename(raw.filenames[0]).replace('.edf', '')
            )
            if onsets is not None:
                raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

        # MEMORY FIX: Load and filter the raw data *after* annotations are set
        try:
            raw_combined = mne.concatenate_raws(raws, preload=True) # Preload just one patient
        except Exception as e:
            print(f"[!] Error concatenating raws: {e}. Skipping patient.")
            continue
            
        raw_combined.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR", n_jobs=-1)
        raw_combined.set_eeg_reference("average", projection=False, verbose="ERROR")

        # Make epochs (preload=False, this is a generator)
        events = mne.make_fixed_length_events(raw_combined, duration=5.0)
        epochs = mne.Epochs(
            raw_combined,
            events,
            tmin=0,
            tmax=5.0 - 1 / TARGET_SFREQ,
            preload=False,  # <-- MEMORY FIX
            baseline=None,
            verbose="ERROR",
        )

        if len(epochs.events) == 0:
            print("[!] No epochs found. Skipping patient.")
            continue

        # Create labels (y)
        y_batch = np.zeros(len(epochs.events))
        for i, epoch in enumerate(epochs.iter_as_random_order()): # Use iterator
            for ann in raw_combined.annotations:
                if ann['description'] == 'seizure' and \
                (epochs.events[i, 0] / TARGET_SFREQ) < (ann['onset'] + ann['duration']) and \
                ((epochs.events[i, 0] / TARGET_SFREQ + 5.0) > ann['onset']):
                    y_batch[i] = 1
                    break
        
        # Create features (X) one epoch at a time
        # This is the most memory-intensive part, but we do it in a generator
        def feature_generator(epochs_obj):
            for epoch_data in epochs_obj.iter_as_random_order():
                yield compute_features(epoch_data, TARGET_SFREQ)
        
        X_batch = np.array(list(feature_generator(epochs)))

        if X_batch.shape[0] != y_batch.shape[0]:
            print(f"[!] Mismatch in X and y shapes ({X_batch.shape[0]} vs {y_batch.shape[0]}). Skipping batch.")
            continue
        
        if len(X_batch) == 0:
            print("[!] No features extracted. Skipping batch.")
            continue

        # --- Scale and Train Batch ---
        print(f"[...] Scaling and training on {len(X_batch)} samples...")
        
        # We 'partial_fit' the scaler, just like the model
        scaler.partial_fit(X_batch) 
        X_batch_scaled = scaler.transform(X_batch)
        
        # We 'partial_fit' the model
        model.partial_fit(X_batch_scaled, y_batch, classes=all_classes)

        # MEMORY FIX: Clear memory before next loop
        del raw_combined, raws, epochs, X_batch, y_batch, X_batch_scaled
        print(f"[✓] Patient batch complete.")

    # --- Training Complete ---
    print("\n[✓] Full batch training complete.")
    
    # --- Save Model and Scaler ---
    joblib.dump(model, MODEL_PATH)
    print(f"[✓] Model saved at: {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"[✓] Scaler saved at: {SCALER_PATH}")

    with open(CHANNELS_LIST_PATH, "w") as f:
        for ch in common_channels:
            f.write(f"{ch}\n")
    print(f"[✓] Channels list saved at: {CHANNELS_LIST_PATH}")

    end_time = datetime.now()
    print(f"Total training time: {end_time - start_time}")
    print("Training complete ✅")