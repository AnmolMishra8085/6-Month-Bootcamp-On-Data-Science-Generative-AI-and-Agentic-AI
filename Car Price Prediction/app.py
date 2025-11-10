import streamlit as st
import pickle
import numpy as np

# --- 1. LOAD THE SAVED MODEL ---
# Open the file in read-binary (rb) mode
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- 2. CREATE THE WEB APP INTERFACE ---

# Set the title
st.title("Car Purchase Price Predictor 🚗")
st.write("Enter the customer details to predict the car purchase amount.")

# Create input fields for the 5 features
# (These must be in the same order as your X features)

# 1. Gender (using a select box)
gender = st.selectbox("Gender", ("Male", "Female"))
# Convert text to 0 or 1
gender_numeric = 0 if gender == "Male" else 1

# 2. Age
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# 3. Annual Salary
annual_salary = st.number_input("Annual Salary ($)", min_value=10000.0, max_value=200000.0, value=60000.0, step=1000.0)

# 4. Credit Card Debt
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0.0, max_value=50000.0, value=5000.0, step=500.0)

# 5. Net Worth
net_worth = st.number_input("Net Worth ($)", min_value=0.0, max_value=1000000.0, value=200000.0, step=10000.0)


# --- 3. CREATE THE PREDICTION BUTTON ---

if st.button("Predict Purchase Amount"):
    # 1. Gather the inputs into a numpy array
    # Make sure the order is correct: [Gender, Age, Annual Salary, Credit Card Debt, Net Worth]
    features = np.array([[
        gender_numeric,
        age,
        annual_salary,
        credit_card_debt,
        net_worth
    ]])

    # 2. Use the model to predict
    prediction = model.predict(features)
    
    # 3. Display the result
    st.success(f"Predicted Car Purchase Amount: ${prediction[0]:,.2f}")
    # The ':, .2f' formats the number nicely (e.g., $45,234.50)