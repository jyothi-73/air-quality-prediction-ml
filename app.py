import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
model = joblib.load("aqi_rf_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv(r"C:\Users\medag\Downloads\jyothi.csv")

# Title
st.title("🌍 Air Quality Index Prediction")
st.markdown("Predict AQI based on location and pollutant levels")

st.write("---")

# State (constant India)
state = st.selectbox("Select State", df['state'].unique())

# City dropdown
filtered_state = df[df['state'] == state]
city = st.selectbox("Select City", filtered_state['city'].unique())

# Station dropdown
filtered_city = filtered_state[filtered_state['city'] == city]
station = st.selectbox("Select Station", filtered_city['station'].unique())

st.write("---")

# Pollutant Inputs (your model features)
pm10 = st.number_input("PM10", 0.0, 500.0, 100.0)
nh3 = st.number_input("NH3", 0.0, 100.0, 20.0)
ozone = st.number_input("OZONE", 0.0, 200.0, 50.0)

# Prediction
if st.button("Predict AQI"):
    data = np.array([[pm10, nh3, ozone]])
    pred = model.predict(data)
    result = le.inverse_transform(pred)[0]

    # Color output
    if result == "Good":
        st.success(f"🟢 AQI Category: {result}")
    elif result == "Satisfactory":
        st.info(f"🟡 AQI Category: {result}")
    elif result == "Moderate":
        st.warning(f"🟠 AQI Category: {result}")
    else:
        st.error(f"🔴 AQI Category: {result}")
    st.success(f" {city} - {station} AQI Category: {result}")
    
   