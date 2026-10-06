import streamlit as st
import numpy as np
import pickle

# ===== Load the model ======
with open("models/scores_model.pkl", "rb") as f:
    model = pickle.load(f)


# ======= Page Configuration ==========
st.set_page_config(page_title="Score Predictor", page_icon="📊")


# ===== Title and description ==========
st.title("📊 Student Score Predictor")
st.write(
    "Enter the number of hours studied, and this app will predict the exam score using a Simple Linear Regression model."
)

# Learned Equation
slope = model.coef_[0]
intercept = model.intercept_
st.markdown(f"**Model equation:** Score = {intercept:.2f} + {slope:.2f} × Hours")


# Input Widget
hours = st.number_input(
    "Hours Studies", min_value=0.0, max_value=24.0, value=5.0, step=0.25
)


# Predict button
if st.button("Predict Score"):
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Score: {prediction: .2f}")

    st.progress(int(prediction))

# Footer
st.caption("Model: Simple Linear Regression trained on student_scores.csv")