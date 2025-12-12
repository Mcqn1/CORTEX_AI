import streamlit as st
import pandas as pd
import os

LOG_PATH = os.path.join("UTIL_DYNAMIC", "inference_logs.csv")

st.set_page_config(
    page_title="EEG Model Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EEG Seizure Model Monitoring")

st.write(
    "This page summarizes all inference runs logged by the EEG Seizure Detection app. "
    "Each record corresponds to one uploaded EDF file."
)

if not os.path.exists(LOG_PATH):
    st.warning("No inference logs found yet. Run some predictions first in the main app.")
else:
    df = pd.read_csv(LOG_PATH)

    if df.empty:
        st.warning("Log file exists but is empty.")
    else:
        # Basic stats
        total_runs = len(df)
        total_windows = df["num_windows"].sum()
        total_seizure_windows = df["num_seizure_windows"].sum()
        seizure_rate = (total_seizure_windows / total_windows * 100) if total_windows > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Inference Runs", total_runs)
        with col2:
            st.metric("Total Windows Evaluated", int(total_windows))
        with col3:
            st.metric("Total Seizure Windows", int(total_seizure_windows))
        with col4:
            st.metric("Overall Seizure Rate", f"{seizure_rate:.2f}%")

        st.markdown("---")
        st.subheader("Recent Inference Runs")

        df_sorted = df.sort_values("timestamp_utc", ascending=False)
        st.dataframe(df_sorted.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Seizure Rate per File")

        df_rates = df_sorted.copy()
        df_rates["seizure_rate_%"] = df_rates.apply(
            lambda row: (row["num_seizure_windows"] / row["num_windows"] * 100) if row["num_windows"] > 0 else 0.0,
            axis=1
        )

        st.dataframe(
            df_rates[["timestamp_utc", "file_name", "num_windows", "num_seizure_windows", "seizure_rate_%"]],
            use_container_width=True
        )
