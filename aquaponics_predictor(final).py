import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model, scaler, and label encoders
with open("saved_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_encoder_fish.pkl", "rb") as file:
    label_encoder_fish = pickle.load(file)

with open("label_encoder_plant.pkl", "rb") as file:
    label_encoder_plant = pickle.load(file)

# Streamlit app UI
st.title("Aquaponics Success Rate Predictor")

# User inputs for the physical aspects
temperature = st.number_input("Temperature")
ph = st.number_input("pH")
dissolved_oxygen = st.number_input("Dissolved Oxygen")
ammonia = st.number_input("Ammonia")
nitrate = st.number_input("Nitrate")
fish_growth_rate = st.number_input("Fish Growth Rate")
leaf_area = st.number_input("Leaf Area")
yield_value = st.number_input("Yield")
fish_species = st.text_input("Fish Species")
plant_species = st.text_input("Pairable Plant Species")

# Preprocess user inputs
if st.button("Predict"):
    try:
        # Encode categorical variables using the pre-fitted LabelEncoder
        fish_species_encoded = label_encoder_fish.transform([fish_species])[0]
        plant_species_encoded = label_encoder_plant.transform([plant_species])[0]

        # Combine numerical features
        numerical_features = np.array([
            temperature, ph, dissolved_oxygen, ammonia, nitrate,
            fish_growth_rate, leaf_area, yield_value
        ]).reshape(1, -1)

        # Scale only numerical features
        numerical_features_scaled = scaler.transform(numerical_features)

        # Combine scaled numerical features with categorical features
        input_features = np.hstack((
            numerical_features_scaled,
            [[fish_species_encoded, plant_species_encoded]]
        ))

        # Predict the success class (Yes or No)
        prediction = svm_model.predict(input_features)

        # Display the result
        if prediction[0] == 'Yes':
            st.success(f"Prediction: Success rate is high!")
        else:
            st.error(f"Prediction: Success rate is low.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Button to navigate to blockchain supply chain page
st.markdown("""
    <a href='http://localhost/supplychain/' target='_blank'>
        <button style='background-color: green; color: white; padding: 10px 15px; font-size: 16px; border: none; cursor: pointer;'>
            Go to Supply Chain
        </button>
    </a>
    """, unsafe_allow_html=True)