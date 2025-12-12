import os
import glob
import numpy as np
import mne
import joblib
from sklearn.metrics import accuracy_score
from EEG_TRAIN_MODEL import (
    load_and_standardize_raw,
    compute_features,
    get_seizure_annotations,
    TARGET_SFREQ,
    MODEL_PATH,
    SCALER_PATH,
    CHANNELS_LIST_PATH,
    BASE_DATA_PATH
)

def load_common_channels():
    with open(CHANNELS_LIST_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

def build_eval_job_list(base_data_path, max_files=3):
    patient_folders = [f.path for f in os.scandir(base_data_path) if f.is_dir()]
    jobs = []
    for patient_folder in patient_folders:
        summary_file = next(glob.iglob(os.path.join(patient_folder, "*.txt")), None)
        if not summary_file:
            continue
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            jobs.append((edf_file, summary_file))
            if len(jobs) >= max_files:
                return jobs
    return jobs

def main():
    print("[EVAL] Loading model, scaler, and channels...", flush=True)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(CHANNELS_LIST_PATH):
        print("[EVAL] Missing model/scaler/channels. Cannot evaluate.", flush=True)
        raise SystemExit(1)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    common_channels = load_common_channels()

    jobs = build_eval_job_list(BASE_DATA_PATH, max_files=3)
    if not jobs:
        print("[EVAL] No evaluation files found.", flush=True)
        raise SystemExit(1)

    all_y = []
    all_y_hat = []

    for i, (edf_path, summary_path) in enumerate(jobs):
        print(f"[EVAL] Processing eval file {i+1}/{len(jobs)}: {edf_path}", flush=True)
        raw = load_and_standardize_raw(edf_path, TARGET_SFREQ, preload=True)
        if raw is None:
            print("[EVAL] raw is None, skipping.", flush=True)
            continue

        if not set(common_channels).issubset(set(raw.ch_names)):
            print("[EVAL] Missing common channels, skipping.", flush=True)
            continue

        file_name_base = os.path.basename(edf_path).replace(".edf", "")
        onsets, durations, descriptions = get_seizure_annotations(summary_path, file_name_base)
        if onsets is not None:
            raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
        else:
            print("[EVAL] No seizure annotations for this file, skipping.", flush=True)
            continue

        raw.pick(common_channels, verbose="ERROR")
        raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR", n_jobs=1)
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

        events = mne.make_fixed_length_events(raw, duration=10.0)
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=10.0 - 1 / TARGET_SFREQ,
            preload=True, baseline=None, verbose="ERROR"
        )

        if len(epochs.events) == 0:
            print("[EVAL] No epochs, skipping.", flush=True)
            continue

        y_batch = np.zeros(len(epochs.events))
        for i_epoch, epoch in enumerate(epochs):
            epoch_start_time = epochs.events[i_epoch, 0] / TARGET_SFREQ
            for ann in raw.annotations:
                if (ann["description"] == "seizure" and
                    epoch_start_time < (ann["onset"] + ann["duration"]) and
                    (epoch_start_time + 10.0) > ann["onset"]):
                    y_batch[i_epoch] = 1
                    break

        X_batch = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
        if X_batch.shape[0] == 0:
            print("[EVAL] No features extracted, skipping.", flush=True)
            continue

        X_scaled = scaler.transform(X_batch)
        y_pred = model.predict(X_scaled)

        all_y.extend(list(y_batch))
        all_y_hat.extend(list(y_pred))

    if not all_y:
        print("[EVAL] No labels collected, cannot compute accuracy.", flush=True)
        raise SystemExit(1)

    y_true = np.array(all_y)
    y_hat = np.array(all_y_hat)

    acc = accuracy_score(y_true, y_hat)
    print(f"[EVAL] Validation accuracy on small eval set: {acc:.4f}", flush=True)

    # Threshold to decide pass/fail
    THRESHOLD = 0.60
    if acc < THRESHOLD:
        print(f"[EVAL] Accuracy {acc:.4f} < threshold {THRESHOLD}. Failing CI.", flush=True)
        raise SystemExit(1)
    else:
        print(f"[EVAL] Accuracy {acc:.4f} >= threshold {THRESHOLD}. CI passes.", flush=True)

if __name__ == "__main__":
    main()
