# app.py
import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("Salary Prediction (Based on Years of Experience)")
st.write("Enter years of experience and the model will predict the expected salary.")

# --- Load model ---
MODEL_PATH = "linear_regression_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}\nRun your backend training script so the model file is created.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Make sure `linear_regression_model.pkl` is present in the same folder. Run the backend to generate it if needed.")
    st.stop()

# --- Input UI ---
years = st.slider("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
years_array = np.array([[years]])  # shape (1,1) expected by sklearn

# Optional: allow direct numeric input
years_exact = st.number_input("Or enter exact years (decimal allowed):", min_value=0.0, max_value=50.0, value=float(years), step=0.1)
# sync slider and input if changed
if years_exact != years:
    years = years_exact
    years_array = np.array([[years]])

if st.button("Predict Salary"):
    # predict
    try:
        pred_salary = model.predict(years_array)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.metric(label=f"Predicted salary for {years:.1f} years experience", value=f"${pred_salary:,.2f}")

    # Show model details (coef, intercept) if available
    try:
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        if coef is not None:
            st.write("**Model parameters:**")
            st.write(f"Coefficient (slope): `{coef[0]:.4f}`")
            st.write(f"Intercept: `{intercept:.2f}`")
    except Exception:
        pass

    # Show a small simulated line + point (simple visual)
    import matplotlib.pyplot as plt
    xs = np.linspace(0, max(50, years+5), 100).reshape(-1, 1)
    ys = model.predict(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="Regression line")
    ax.scatter([years], [pred_salary], color="red", label="Your input prediction")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)

    # Allow user to download the prediction as a small CSV
    import pandas as pd
    df_out = pd.DataFrame({"years_experience": [years], "predicted_salary": [pred_salary]})
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download prediction (CSV)", data=csv, file_name="salary_prediction.csv", mime="text/csv")

# Footer instructions
st.markdown("---")
st.write("Notes:")
st.write("- This app expects a trained linear regression model saved as `linear_regression_model.pkl` (your backend saved that file).")
st.write("- If you haven't generated the pickle yet, run the backend training script to create `linear_regression_model.pkl` and restart the app.")
