import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

st.title("🍷 Wine Cluster Prediction (DBSCAN)")

st.write("Enter wine chemical properties to predict cluster.")

# User inputs
alcohol = st.number_input("Alcohol", 0.0, 20.0)
malic_acid = st.number_input("Malic Acid", 0.0, 10.0)
ash = st.number_input("Ash", 0.0, 5.0)
ash_alcanity = st.number_input("Ash Alcanity", 0.0, 50.0)
magnesium = st.number_input("Magnesium", 0.0, 200.0)
total_phenols = st.number_input("Total Phenols", 0.0, 5.0)
flavanoids = st.number_input("Flavanoids", 0.0, 5.0)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", 0.0, 1.0)
proanthocyanins = st.number_input("Proanthocyanins", 0.0, 5.0)
color_intensity = st.number_input("Color Intensity", 0.0, 20.0)
hue = st.number_input("Hue", 0.0, 5.0)
od280 = st.number_input("OD280", 0.0, 5.0)
proline = st.number_input("Proline", 0.0, 2000.0)

if st.button("Predict Cluster"):
    # Create dataframe from input
    input_data = pd.DataFrame([[alcohol, malic_acid, ash, ash_alcanity, magnesium,
                                total_phenols, flavanoids, nonflavanoid_phenols,
                                proanthocyanins, color_intensity, hue, od280, proline]],
                              columns=['alcohol','malic_acid','ash','ash_alcanity','magnesium',
                                       'total_phenols','flavanoids','nonflavanoid_phenols',
                                       'proanthocyanins','color_intensity','hue','od280','proline'])

    # Standardize
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data)

    # Apply DBSCAN
    model = DBSCAN(eps=2, min_samples=2)
    cluster = model.fit_predict(scaled_input)

    st.success(f"Predicted Cluster: {cluster[0]}")