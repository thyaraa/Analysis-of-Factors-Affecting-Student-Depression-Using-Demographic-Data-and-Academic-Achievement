import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from fnn import load_fnn_model, predict_fnn
from random_forest import load_rf_model, predict_rf

# Load models
try:
    fnn_model = load_fnn_model("/Users/diamondamor/Documents/Coolyeah/Semester 7/UAP/src/model/ffnn_model.h5")
    rf_model = load_rf_model("/Users/diamondamor/Documents/Coolyeah/Semester 7/UAP/src/model/random_model.h5")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Initialize scaler for numerical features
scaler = StandardScaler()

def main():
    st.title("Depression Analysis for Students")
    st.write("This application analyzes the likelihood of depression based on input features.")

    # Collect user input
    st.sidebar.header("Input Features")
    academic_pressure = st.sidebar.slider("Academic Pressure", min_value=0, max_value=100, step=1)
    work_pressure = st.sidebar.slider("Work Pressure", min_value=0, max_value=100, step=1)
    cgpa = st.sidebar.slider("CGPA", min_value=0.0, max_value=4.0, step=0.1)
    study_satisfaction = st.sidebar.slider("Study Satisfaction (1-10)", min_value=1, max_value=10, step=1)
    suicidal_thoughts = st.sidebar.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    financial_stress = st.sidebar.slider("Financial Stress (1-10)", min_value=1, max_value=10, step=1)
    family_history = st.sidebar.selectbox("Family History of Mental Illness", ["No", "Yes"])

    # Convert categorical input to numerical
    suicidal_thoughts_numeric = 1 if suicidal_thoughts == "Yes" else 0
    family_history_numeric = 1 if family_history == "Yes" else 0

    # Create input vector (7 fitur asli)
    input_features = np.array([[academic_pressure, work_pressure, cgpa, study_satisfaction,
                                 suicidal_thoughts_numeric, financial_stress, family_history_numeric]])

    # Normalize numerical features
    processed_input = scaler.fit_transform(input_features)

    # Tambahkan placeholder untuk mencapai 9 fitur
    additional_features = np.zeros((processed_input.shape[0], 2))  # Sesuaikan jumlah placeholder
    processed_input = np.hstack((processed_input, additional_features))

    # Debugging: Tampilkan dimensi akhir input
    st.write(f"Processed Input Shape: {processed_input.shape}")

    # Model selection
    model_choice = st.selectbox("Choose a model to use:", ["FFNN", "Random Forest"])

    if st.button("Analyze"):
        try:
            if model_choice == "FFNN" and fnn_model:
                prediction = predict_fnn(fnn_model, processed_input)
                st.write(f"FFNN Prediction: {prediction[0]}")
            elif model_choice == "Random Forest" and rf_model:
                prediction = predict_rf(rf_model, processed_input)
                st.write(f"Random Forest Prediction: {prediction[0]}")
            else:
                st.error("Selected model is not loaded.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()