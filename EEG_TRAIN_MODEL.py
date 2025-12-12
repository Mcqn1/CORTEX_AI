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

print("[DEBUG] EEG_TRAIN_MODEL starting...", flush=True)

# --- Paths ---
MODEL_DIR = os.path.join(os.getcwd(), "UTIL_DYNAMIC")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dynamic_scaler.pkl")
CHANNELS_LIST_PATH = os.path.join(MODEL_DIR, "common_channels.txt")
BASE_DATA_PATH = "EEG_DATA" 
TARGET_SFREQ = 256  # Hz

print(f"[DEBUG] MODEL_DIR={MODEL_DIR}", flush=True)
print(f"[DEBUG] MODEL_PATH={MODEL_PATH}", flush=True)
print(f"[DEBUG] SCALER_PATH={SCALER_PATH}", flush=True)
print(f"[DEBUG] CHANNELS_LIST_PATH={CHANNELS_LIST_PATH}", flush=True)
print(f"[DEBUG] BASE_DATA_PATH={BASE_DATA_PATH}", flush=True)
print(f"[DEBUG] TARGET_SFREQ={TARGET_SFREQ}", flush=True)

# --- Helper Functions (Unchanged) ---
def parse_time_to_seconds(time_str):
    print(f"[DEBUG] parse_time_to_seconds called with time_str={time_str}", flush=True)
    if '.' in time_str or ':' in time_str:
        time_str = time_str.replace('.', ':')
        parts = list(map(int, time_str.split(':')))
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        print(f"[DEBUG] Parsed time_str into {seconds} seconds", flush=True)
        return seconds
    val = int(time_str)
    print(f"[DEBUG] Parsed integer seconds={val}", flush=True)
    return val

def get_seizure_annotations(summary_file, file_name):
    print(f"[DEBUG] get_seizure_annotations called. summary_file={summary_file}, file_name={file_name}", flush=True)
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    file_sections = content.split('File Name:')[1:]
    print(f"[DEBUG] Number of file sections in summary={len(file_sections)}", flush=True)
    for section_idx, section in enumerate(file_sections):
        if file_name in section:
            print(f"[DEBUG] Matched section index={section_idx} for file_name={file_name}", flush=True)
            lines = section.strip().split('\n')
            num_seizures = 0
            for line in lines:
                if 'Number of Seizures in File' in line:
                    try:
                        num_seizures = int(line.split(':')[-1].strip())
                        print(f"[DEBUG] num_seizures={num_seizures}", flush=True)
                        break
                    except (ValueError, IndexError):
                        print("[DEBUG] Failed to parse Number of Seizures line.", flush=True)
                        continue
            if num_seizures == 0:
                print("[DEBUG] num_seizures is 0. Returning None.", flush=True)
                return None, None, None
            starts, ends = [], []
            if 'Seizure 1 Start Time' in section:
                print("[DEBUG] Using 'Seizure X Start/End Time' style annotations.", flush=True)
                for i in range(1, num_seizures + 1):
                    for line in lines:
                        if f'Seizure {i} Start Time:' in line:
                            val = float(line.split(':')[-1].strip().replace(' seconds', ''))
                            starts.append(val)
                            print(f"[DEBUG] Seizure {i} start={val}", flush=True)
                        if f'Seizure {i} End Time:' in line:
                            val = float(line.split(':')[-1].strip().replace(' seconds', ''))
                            ends.append(val)
                            print(f"[DEBUG] Seizure {i} end={val}", flush=True)
            else:
                print("[DEBUG] Using registration-based seizure timestamps.", flush=True)
                reg_start_sec = 0
                for line in lines:
                    if 'Registration start time' in line:
                        reg_start_sec = parse_time_to_seconds(line.split(': ')[1])
                        print(f"[DEBUG] reg_start_sec={reg_start_sec}", flush=True)
                        break
                for line in lines:
                    if 'Seizure start time' in line:
                        val = parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec
                        starts.append(val)
                        print(f"[DEBUG] Relative seizure start={val}", flush=True)
                    if 'Seizure end time' in line:
                        val = parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec
                        ends.append(val)
                        print(f"[DEBUG] Relative seizure end={val}", flush=True)
            onsets = np.array(starts)
            durations = np.array(ends) - onsets
            descriptions = ['seizure'] * len(onsets)
            print(f"[DEBUG] Annotations created. Count={len(onsets)}", flush=True)
            return onsets, durations, descriptions
    print("[DEBUG] No matching section found for this EDF.", flush=True)
    return None, None, None

def load_and_standardize_raw(edf_path, target_sfreq, preload=False):
    print(f"[DEBUG] load_and_standardize_raw called. edf_path={edf_path}, target_sfreq={target_sfreq}, preload={preload}", flush=True)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose='ERROR') 
        print("[DEBUG] EDF header/data loaded.", flush=True)
        print(f"[DEBUG] Original sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}", flush=True)
        raw.rename_channels(lambda ch: ch.strip().upper())
        print(f"[DEBUG] Channels after rename (first 10)={raw.ch_names[:10]}", flush=True)
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG'])
        print(f"[DEBUG] Channels after pick_types(eeg=True) (first 10)={raw.ch_names[:10]}, total={len(raw.ch_names)}", flush=True)
        if raw.info['sfreq'] != target_sfreq:
            print(f"[DEBUG] Resampling from {raw.info['sfreq']} Hz to {target_sfreq} Hz", flush=True)
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception as e:
        print(f"[DEBUG] ERROR in load_and_standardize_raw: {e}", flush=True)
        return None

def compute_features(epoch_data, sfreq):
    print(f"[DEBUG] compute_features called. epoch_data.shape={epoch_data.shape}, sfreq={sfreq}", flush=True)
    feats = []
    for idx, ch_data in enumerate(epoch_data):
        mean, std = np.mean(ch_data), np.std(ch_data)
        sk, ku = skew(ch_data), kurtosis(ch_data)
        print(f"[DEBUG] Channel {idx}: mean={mean}, std={std}, skew={sk}, kurtosis={ku}", flush=True)
        # OPTIMIZED: shorter STFT window
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq // 2))
        Pxx = np.abs(Zxx)**2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[(f >= lo) & (f <= hi), :]) for (lo, hi) in bands.values()]
        print(f"[DEBUG] Channel {idx}: band_powers={band_powers}", flush=True)
        feats.extend([mean, std, sk, ku, *band_powers])
    print(f"[DEBUG] Feature vector length={len(feats)}", flush=True)
    return np.array(feats)

# --- New Memory-Safe Training Logic ---
def discover_common_channels(base_data_path, target_sfreq):
    print("[PASS 1] Discovering common channels (low-memory)...", flush=True)
    patient_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    print(f"[DEBUG] Number of patient folders={len(patient_folders)}", flush=True)
    common_channels = None
    for patient_idx, patient_folder in enumerate(patient_folders):
        print(f"[DEBUG] Scanning patient folder {patient_idx+1}/{len(patient_folders)}: {patient_folder}", flush=True)
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            print(f"[DEBUG] Checking EDF file for common channels: {edf_file}", flush=True)
            raw = load_and_standardize_raw(edf_file, target_sfreq, preload=False) 
            if raw:
                current_channels = set(raw.ch_names)
                print(f"[DEBUG] EDF channels count={len(current_channels)}", flush=True)
                if common_channels is None:
                    common_channels = current_channels
                else:
                    common_channels.intersection_update(current_channels)
    if common_channels:
        print(f"[✓] Found {len(common_channels)} common channels.", flush=True)
        return sorted(list(common_channels))
    raise RuntimeError("No common channels found across EDFs.")

def build_job_list(base_data_path):
    print("[PASS 2] Building file job list...", flush=True)
    job_list = []
    patient_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    print(f"[DEBUG] Number of patient folders={len(patient_folders)}", flush=True)
    for patient_idx, patient_folder in enumerate(patient_folders):
        print(f"[DEBUG] Looking for summary + EDF in folder {patient_idx+1}/{len(patient_folders)}: {patient_folder}", flush=True)
        summary_file = next(glob.iglob(os.path.join(patient_folder, "*.txt")), None)
        if not summary_file:
            print(f"[!] Warning: No summary .txt file found in {os.path.basename(patient_folder)}, skipping.", flush=True)
            continue
        
        print(f"[DEBUG] Found summary file: {summary_file}", flush=True)
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            print(f"[DEBUG] Adding job: edf_file={edf_file}, summary_file={summary_file}", flush=True)
            job_list.append((edf_file, summary_file))
    print(f"[✓] Job list created with {len(job_list)} EDF files to process.", flush=True)
    return job_list

# --- Main Training ---
if __name__ == "__main__":
    print("Starting MLOps training pipeline (Ultra-Memory-Efficient Mode)...", flush=True)
    start_time = datetime.now()

    # PASS 1: Find common channels (low memory)
    common_channels = discover_common_channels(BASE_DATA_PATH, TARGET_SFREQ)
    print(f"[DEBUG] common_channels length={len(common_channels)}", flush=True)

    # PASS 2: Build a list of all files to process (low memory)
    job_list = build_job_list(BASE_DATA_PATH)
    print(f"[DEBUG] Total jobs to process={len(job_list)}", flush=True)

    # PASS 3: Initialize model and train one file at a time
    print("\n[PASS 3] Starting batch training (one EDF file at a time)...", flush=True)
    
    # --- THIS IS THE FIX ---
    # Changed loss to 'log_loss' to enable predict_proba()
    model = SGDClassifier(loss='log_loss', random_state=42, n_jobs=1) 
    print("[DEBUG] SGDClassifier initialized with loss='log_loss'", flush=True)
    
    scaler = StandardScaler()
    print("[DEBUG] StandardScaler initialized.", flush=True)
    all_classes = np.array([0, 1])
    
    for i, (edf_path, summary_path) in enumerate(job_list):
        print(f"\n[+] Processing file {i+1}/{len(job_list)}: {os.path.basename(edf_path)}", flush=True)
        print(f"[DEBUG] EDF path={edf_path}", flush=True)
        print(f"[DEBUG] Summary path={summary_path}", flush=True)
        
        try:
            # Load ONE file into memory
            raw = load_and_standardize_raw(edf_path, TARGET_SFREQ, preload=True)
            if raw is None or not set(common_channels).issubset(set(raw.ch_names)):
                print("[!] File is invalid or missing common channels, skipping.", flush=True)
                continue

            print(f"[DEBUG] Raw loaded. n_channels={len(raw.ch_names)}, n_times={raw.n_times}", flush=True)

            # Add annotations for this file
            file_name_base = os.path.basename(edf_path).replace('.edf', '')
            print(f"[DEBUG] Getting seizure annotations for {file_name_base}", flush=True)
            onsets, durations, descriptions = get_seizure_annotations(summary_path, file_name_base)
            if onsets is not None:
                print(f"[DEBUG] Adding {len(onsets)} seizure annotations to raw.", flush=True)
                raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
            else:
                print("[DEBUG] No seizure annotations found for this file.", flush=True)

            # Process this single file
            print("[DEBUG] Picking common_channels from raw.", flush=True)
            raw.pick(common_channels, verbose='ERROR')
            print(f"[DEBUG] After pick: n_channels={len(raw.ch_names)}", flush=True)

            print("[DEBUG] Applying bandpass filter 0.5–48 Hz (n_jobs=1).", flush=True)
            raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR", n_jobs=1)
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            print("[DEBUG] Filter + average reference applied.", flush=True)

            # Create epochs
            print("[DEBUG] Creating fixed-length events (duration=10.0s).", flush=True)
            events = mne.make_fixed_length_events(raw, duration=10.0)
            print(f"[DEBUG] Number of events created={len(events)}", flush=True)

            epochs = mne.Epochs(
                raw, events, tmin=0, tmax=10.0 - 1 / TARGET_SFREQ,
                preload=True, baseline=None, verbose="ERROR",
            )
            print(f"[DEBUG] Number of epochs={len(epochs.events)}", flush=True)

            if len(epochs.events) == 0:
                print("[!] No epochs found, skipping.", flush=True)
                del raw, epochs # Clean up
                continue

            # Create labels (y_batch)
            y_batch = np.zeros(len(epochs.events))
            print("[DEBUG] Creating labels for each epoch...", flush=True)
            for i_epoch, epoch in enumerate(epochs):
                epoch_start_time = epochs.events[i_epoch, 0] / TARGET_SFREQ
                for ann in raw.annotations:
                    if ann['description'] == 'seizure' and \
                       epoch_start_time < (ann['onset'] + ann['duration']) and \
                       (epoch_start_time + 10.0) > ann['onset']:
                        y_batch[i_epoch] = 1
                        break
            print(f"[DEBUG] Label distribution: seizures={int(y_batch.sum())}, normals={len(y_batch)-int(y_batch.sum())}", flush=True)
            
            # Create features (X_batch)
            print("[DEBUG] Extracting features for each epoch...", flush=True)
            X_batch = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
            print(f"[DEBUG] X_batch shape={X_batch.shape}", flush=True)

            if X_batch.shape[0] == 0:
                print("[!] No features extracted, skipping.", flush=True)
                del raw, epochs, y_batch # Clean up
                continue
                
            # --- Scale and Train Batch ---
            print(f"[...] Scaling and training on {len(X_batch)} samples...", flush=True)
            scaler.partial_fit(X_batch) 
            X_batch_scaled = scaler.transform(X_batch)
            print(f"[DEBUG] X_batch_scaled shape={X_batch_scaled.shape}", flush=True)
            model.partial_fit(X_batch_scaled, y_batch, classes=all_classes)
            print("[DEBUG] partial_fit completed for this batch.", flush=True)

            # CRITICAL: Clean up memory before next loop
            del raw, epochs, X_batch, y_batch, X_batch_scaled
            print(f"[✓] File batch complete.", flush=True)

        except Exception as e:
            print(f"[!!!] CRITICAL ERROR on file {edf_path}: {e}. Skipping file.", flush=True)
            continue # Skip this file and try the next

    # --- Training Complete ---
    print("\n[✓] Full batch training complete.", flush=True)
    
    # Save the MODEL
    joblib.dump(model, MODEL_PATH)
    print(f"[✓] Model saved at: {MODEL_PATH}", flush=True)
    
    # Save the SCALER
    joblib.dump(scaler, SCALER_PATH)
    print(f"[✓] Scaler saved at: {SCALER_PATH}", flush=True)

    with open(CHANNELS_LIST_PATH, "w") as f:
        for ch in common_channels:
            f.write(f"{ch}\n")
    print(f"[✓] Channels list saved at: {CHANNELS_LIST_PATH}", flush=True)

    end_time = datetime.now()
    print(f"Total training time: {end_time - start_time}", flush=True)
    print("Training complete ✅", flush=True)
