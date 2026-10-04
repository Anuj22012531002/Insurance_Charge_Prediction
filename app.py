import streamlit as st
import pandas as pd
import pickle

model =  pickle.load(open('Insurance_price_prediction1.pkl','rb'))  # rb - read binary

# ---------------- TITLE ----------------
st.title("💰 Insurance Charge Prediction")
st.write("Predict estimated insurance charges using the trained XGBoost model.")

st.divider()

# ---------------- INPUTS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏥 Medical Information")

    coverage_amount = st.number_input(
        "Coverage Amount",
        min_value=0.0,
        value=100000.0
    )

    hospital_visits = st.number_input(
        "Hospital Visits",
        min_value=0,
        value=2
    )

    doctor_visits = st.number_input(
        "Doctor Visits",
        min_value=0,
        value=4
    )

    prescription_count = st.number_input(
        "Prescription Count",
        min_value=0,
        value=3
    )

    emergency_visits = st.number_input(
        "Emergency Visits",
        min_value=0,
        value=1
    )

    claim_count = st.number_input(
        "Claim Count",
        min_value=0,
        value=1
    )

with col2:
    st.subheader("📊 Policy Information")

    annual_medical_expenses = st.number_input(
        "Annual Medical Expenses",
        min_value=0.0,
        value=5000.0
    )

    policy_tenure_years = st.number_input(
        "Policy Tenure (Years)",
        min_value=0,
        value=5
    )

    insurance_plan = st.selectbox(
        "Insurance Plan",
        ["Basic", "Standard", "Premium"]
    )

    preventive_checkup = st.selectbox(
        "Preventive Checkup",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["North", "East", "West", "South"]
    )

    income_level = st.selectbox(
        "Income Level",
        ["Low", "Middle", "High"]
    )

with col3:
    st.subheader("👤 Customer Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=1
    )

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    exercise_level = st.selectbox(
        "Exercise Level",
        ["Low", "Moderate", "High"]
    )

    family_history = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Business", "Retired", "Salaried", "Self-Employed", "Student"]
    )

st.divider()

# ---------------- ENCODING ----------------

insurance_map = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}

sex_map = {
    "Female": 0,
    "Male": 1
}

region_map = {
    "North": 0,
    "East": 1,
    "West": 2,
    "South": 3
}

income_map = {
    "Low": 0,
    "Middle": 1,
    "High": 2
}

exercise_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

# ---------------- PREDICTION BUTTON ----------------

if st.button("🔮 Predict Insurance Charges", use_container_width=True):

    input_data = pd.DataFrame([{
        "coverage_amount": coverage_amount,
        "hospital_visits": hospital_visits,
        "doctor_visits": doctor_visits,
        "prescription_count": prescription_count,
        "emergency_visits": emergency_visits,
        "claim_count": claim_count,
        "annual_medical_expenses": annual_medical_expenses,
        "policy_tenure_years": policy_tenure_years,
        "age": age,
        "bmi": bmi,
        "children": children,

        "insurance_plan": insurance_map[insurance_plan],
        "preventive_checkup": yes_no_map[preventive_checkup],
        "sex": sex_map[sex],
        "smoker": yes_no_map[smoker],
        "region": region_map[region],
        "income_level": income_map[income_level],
        "exercise_level": exercise_map[exercise_level],
        "family_history": yes_no_map[family_history],

        "occupation_Retired": 1 if occupation == "Retired" else 0,
        "occupation_Salaried": 1 if occupation == "Salaried" else 0,
        "occupation_Self-Employed": 1 if occupation == "Self-Employed" else 0,
        "occupation_Student": 1 if occupation == "Student" else 0
    }])

    # Exact feature order
    feature_order = [
        "insurance_plan",
        "coverage_amount",
        "hospital_visits",
        "doctor_visits",
        "prescription_count",
        "emergency_visits",
        "claim_count",
        "annual_medical_expenses",
        "policy_tenure_years",
        "preventive_checkup",
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "income_level",
        "exercise_level",
        "family_history",
        "occupation_Retired",
        "occupation_Salaried",
        "occupation_Self-Employed",
        "occupation_Student"
    ]

    input_data = input_data[feature_order]

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        "💰 Estimated Insurance Charges",
        f"₹ {prediction:,.2f}"
    )

    st.subheader("📋 Input Summary")
    st.dataframe(
        input_data,
        use_container_width=True
    )

    # Download prediction
    result = input_data.copy()
    result["Predicted_Charges"] = prediction

    csv = result.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction",
        csv,
        "insurance_prediction.csv",
        "text/csv"
    )