import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the model from the local file."""
    try:
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        st.error("Error: model.joblib file not found. Please run the training pipeline.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- App UI ---
st.title("🧠 EEG Seizure Detection")
st.write("Use the sliders to input EEG feature values and predict if a seizure is occurring.")

if model:
    with st.form("prediction_form"):
        st.subheader("Input Patient EEG Features:")
        
        # --- IMPORTANT ---
        # You MUST replace these with your model's actual features.
        # I am just guessing. Check your training script for the correct column names.
        
        feature_1 = st.slider("Feature 1 (e.g., Delta Wave)", 0.0, 100.0, 50.0)
        feature_2 = st.slider("Feature 2 (e.g., Theta Wave)", 0.0, 100.0, 50.0)
        feature_3 = st.slider("Feature 3 (e.g., Alpha Wave)", 0.0, 100.0, 50.0)
        feature_4 = st.slider("Feature 4 (e.g., Beta Wave)", 0.0, 100.0, 50.0)
        
        # Add more sliders or st.number_input for all your features...

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Create a DataFrame for the model
            input_data = pd.DataFrame(
                {
                    # Make sure these column names match your model's training
                    "Feature 1": [feature_1],
                    "Feature 2": [feature_2],
                    "Feature 3": [feature_3],
                    "Feature 4": [feature_4],
                }
            )
            
            st.write("Input data:")
            st.dataframe(input_data)
            
            # Get prediction
            prediction = model.predict(input_data)
            result = prediction[0]

            st.subheader("Prediction Result:")
            if result == 1:
                st.error("Prediction: Seizure Detected (1)")
            else:
                st.success("Prediction: No Seizure Detected (0)")
else:
    st.error("Model could not be loaded. Run the 'TRIGGER: Initial model training' Action on GitHub to create one.")