# app.py
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved Q-network and scaler
MODEL_PATH = r"D:\Data_Science_Project\Clinical\RL_Model\Saved_Models\q_network_model.h5"
SCALER_PATH = r"D:\Data_Science_Project\Clinical\RL_Model\Saved_Models\scaler.save"
METADATA_PATH = r"D:\Data_Science_Project\Clinical\RL_Model\Saved_Models\metadata.json"

q_model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load metadata
import json
with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

# Streamlit App
st.title("Clinical Treatment Recommendation (Offline RL)")
st.markdown("Predict the best treatment action using saved Q-network model.")

# User input
age = st.number_input("Enter patient age:", min_value=0, max_value=120, value=50)
gender = st.selectbox("Select patient gender:", options=["M", "F"])

# Convert gender to numeric
gender_num = 0 if gender == "M" else 1

# Predict button
if st.button("Predict Best Action"):
    # Prepare input
    state = np.array([[age, gender_num]])
    state_scaled = scaler.transform(state)
    
    # Get action from model
    q_values = q_model.predict(state_scaled, verbose=0)
    action_idx = np.argmax(q_values[0])
    
    # Map action to human-readable
    action_name = meta["action_mapping"][str(action_idx)]
    
    # Predicted survival (simplified)
    predicted_survival = "Survived" if action_idx == 0 else "High risk"
    
    st.success(f"Recommended Action: {action_name}")
    st.info(f"Predicted Outcome: {predicted_survival}")
    st.write(f"Q-values: {q_values[0]}")
