import os
import glob
import warnings
import numpy as np
import mne
import joblib
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from datetime import datetime

# --- Basic setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Paths ---
MODEL_DIR = os.path.join(os.getcwd(), "UTIL_DYNAMIC")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_LIST_PATH = os.path.join(MODEL_DIR, "common_channels.txt")
BASE_DATA_PATH = "EEG_DATA" 
TARGET_SFREQ = 256  # Hz

# --- Helper Functions (Unchanged) ---
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

def load_and_standardize_raw(edf_path, target_sfreq, preload=False):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose='ERROR') 
        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG'])
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception:
        return None

def compute_features(epoch_data, sfreq):
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

# --- New Memory-Safe Training Logic ---
def discover_common_channels(base_data_path, target_sfreq):
    print("[PASS 1] Discovering common channels (low-memory)...")
    patient_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    common_channels = None
    for patient_folder in patient_folders:
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            raw = load_and_standardize_raw(edf_file, target_sfreq, preload=False) 
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

def build_job_list(base_data_path):
    print("[PASS 2] Building file job list...")
    job_list = []
    patient_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    for patient_folder in patient_folders:
        summary_file = next(glob.iglob(os.path.join(patient_folder, "*.txt")), None)
        if not summary_file:
            print(f"[!] Warning: No summary .txt file found in {os.path.basename(patient_folder)}, skipping.")
            continue
        
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            job_list.append((edf_file, summary_file))
    print(f"[✓] Job list created with {len(job_list)} EDF files to process.")
    return job_list

# --- Main Training ---
if __name__ == "__main__":
    print("Starting MLOps training pipeline (Ultra-Memory-Efficient Mode)...")
    start_time = datetime.now()

    # PASS 1: Find common channels (low memory)
    common_channels = discover_common_channels(BASE_DATA_PATH, TARGET_SFREQ)

    # PASS 2: Build a list of all files to process (low memory)
    job_list = build_job_list(BASE_DATA_PATH)

    # PASS 3: Initialize model and train one file at a time
    print("\n[PASS 3] Starting batch training (one EDF file at a time)...")
    
    # --- THIS IS THE FIX ---
    # Removed 'class_weight="balanced"' because it's not supported by partial_fit
    model = SGDClassifier(loss='hinge', random_state=42, n_jobs=1) 
    
    scaler = StandardScaler()
    all_classes = np.array([0, 1])
    
    for i, (edf_path, summary_path) in enumerate(job_list):
        print(f"\n[+] Processing file {i+1}/{len(job_list)}: {os.path.basename(edf_path)}")
        
        try:
            # Load ONE file into memory
            raw = load_and_standardize_raw(edf_path, TARGET_SFREQ, preload=True)
            if raw is None or not set(common_channels).issubset(set(raw.ch_names)):
                print("[!] File is invalid or missing common channels, skipping.")
                continue

            # Add annotations for this file
            file_name_base = os.path.basename(edf_path).replace('.edf', '')
            onsets, durations, descriptions = get_seizure_annotations(summary_path, file_name_base)
            if onsets is not None:
                raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

            # Process this single file
            raw.pick(common_channels, verbose='ERROR')
            raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR", n_jobs=1)
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")

            # Create epochs
            events = mne.make_fixed_length_events(raw, duration=5.0)
            epochs = mne.Epochs(
                raw, events, tmin=0, tmax=5.0 - 1 / TARGET_SFREQ,
                preload=True, baseline=None, verbose="ERROR",
            )

            if len(epochs.events) == 0:
                print("[!] No epochs found, skipping.")
                del raw, epochs # Clean up
                continue

            # Create labels (y_batch)
            y_batch = np.zeros(len(epochs.events))
            for i_epoch, epoch in enumerate(epochs):
                for ann in raw.annotations:
                    if ann['description'] == 'seizure' and \
                       (epochs.events[i_epoch, 0] / TARGET_SFREQ) < (ann['onset'] + ann['duration']) and \
                       ((epochs.events[i_epoch, 0] / TARGET_SFREQ + 5.0) > ann['onset']):
                        y_batch[i_epoch] = 1
                        break
            
            # Create features (X_batch)
            X_batch = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])

            if X_batch.shape[0] == 0:
                print("[!] No features extracted, skipping.")
                del raw, epochs, y_batch # Clean up
                continue
                
            # --- Scale and Train Batch ---
            print(f"[...] Scaling and training on {len(X_batch)} samples...")
            scaler.partial_fit(X_batch) 
            X_batch_scaled = scaler.transform(X_batch)
            model.partial_fit(X_batch_scaled, y_batch, classes=all_classes)

            # CRITICAL: Clean up memory before next loop
            del raw, epochs, X_batch, y_batch, X_batch_scaled
            print(f"[✓] File batch complete.")

        except Exception as e:
            print(f"[!!!] CRITICAL ERROR on file {edf_path}: {e}. Skipping file.")
            continue # Skip this file and try the next

    # --- Training Complete ---
    print("\n[✓] Full batch training complete.")
    
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