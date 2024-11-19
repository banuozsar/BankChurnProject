import pandas as pd
# %pip install streamlit
import streamlit as st
import pickle
import os
import base64
import random
import joblib
from catboost import CatBoostClassifier
# %pip install shap
import shap



# %streamlit run streamlit_ChurnProject.py

@st.cache_data
def load_data():
    return joblib.load('customer_data.pkl')

# Call the function to load and cache `df`
df = load_data()

# Load the trained CatBoost model
catboost_model = joblib.load('catboost_model.pkl')

# Title for the app
st.title("Müşteri Terk Tahminleme Sorgusu")

# Description
st.write("Aşağıdaki özellikleri girerek bir müşterinin ayrılıp ayrılmayacağını tahmin edin:")

# Define additional input features (excluding Gender, Education Level, Card Category, and Income Category)
MONTHS_ON_BOOK = st.number_input("Müşteri Yaşı (Ay)", min_value=1, max_value=50, value=36)
TOTAL_CT_CHNG_Q4_Q1 = st.number_input("İşlem Sayısındaki Değişim (4. Çeyrek / 1. Çeyrek)", min_value=0.0, max_value=5.0,
                                      value=1.0, step=0.01)
NEW_AMT_PER_TRANS = st.number_input("İşlem Başına Ortalama Tutar", min_value=0.0, max_value=10000.0, value=1000.0)
TOTAL_REVOLVING_BAL = st.number_input("Toplam Bakiye", min_value=0.0, max_value=50000.0, value=1500.0)

# Default values for features (including Gender, Education Level, Card Category, Income Category)
default_values = {
    "CUSTOMER_AGE": df["CUSTOMER_AGE"].mean(),
    "CREDIT_LIMIT": df["CREDIT_LIMIT"].mean(),
    "AVG_OPEN_TO_BUY": df["AVG_OPEN_TO_BUY"].mean(),
    "TOTAL_AMT_CHNG_Q4_Q1": df["TOTAL_AMT_CHNG_Q4_Q1"].mean(),

    # Set defaults for removed inputs
    "GENDER": 1,  # Assuming "M" as default
    "EDUCATION_LEVEL": 3,  # Assuming "College" as default (encoded as 3)
    "CARD_CATEGORY": 0,  # Assuming "Blue" as default (encoded as 0)
    "INCOME_CATEGORY": 2,  # Assuming "$40K - $60K" as default (encoded as 2)

    # One-hot encoded default values
    "MARITAL_STATUS_Married": 1,
    "MARITAL_STATUS_Single": 0,
    "MARITAL_STATUS_Unknown": 0,

    "DEPENDENT_COUNT_1": 1,  # Assuming most customers have 1 dependent
    "DEPENDENT_COUNT_2": 0,
    "DEPENDENT_COUNT_3": 0,
    "DEPENDENT_COUNT_4": 0,
    "DEPENDENT_COUNT_5": 0,

    "TOTAL_RELATIONSHIP_COUNT_2": 1,  # Assuming mode is 2 relationships
    "TOTAL_RELATIONSHIP_COUNT_3": 0,
    "TOTAL_RELATIONSHIP_COUNT_4": 0,
    "TOTAL_RELATIONSHIP_COUNT_5": 0,
    "TOTAL_RELATIONSHIP_COUNT_6": 0,

    "MONTHS_INACTIVE_12_MON_1": 1,  # Assuming 1 month inactive as common
    "MONTHS_INACTIVE_12_MON_2": 0,
    "MONTHS_INACTIVE_12_MON_3": 0,
    "MONTHS_INACTIVE_12_MON_4": 0,
    "MONTHS_INACTIVE_12_MON_5": 0,
    "MONTHS_INACTIVE_12_MON_6": 0,

    "CONTACTS_COUNT_12_MON_1": 1,  # Assuming 1 contact as common
    "CONTACTS_COUNT_12_MON_2": 0,
    "CONTACTS_COUNT_12_MON_3": 0,
    "CONTACTS_COUNT_12_MON_4": 0,
    "CONTACTS_COUNT_12_MON_5": 0,
    "CONTACTS_COUNT_12_MON_6": 0,
}

# Retrieve the model's feature names to ensure correct order and naming
model_features = catboost_model.feature_names_

# Create input data using the model's feature order
input_data = [[default_values.get(feature, 0) for feature in model_features]]

# Update specific input values based on user inputs
input_data[0][model_features.index('MONTHS_ON_BOOK')] = MONTHS_ON_BOOK
input_data[0][model_features.index('TOTAL_CT_CHNG_Q4_Q1')] = TOTAL_CT_CHNG_Q4_Q1
input_data[0][model_features.index('NEW_AMT_PER_TRANS')] = NEW_AMT_PER_TRANS
input_data[0][model_features.index('TOTAL_REVOLVING_BAL')] = TOTAL_REVOLVING_BAL

# Convert to DataFrame
input_df = pd.DataFrame(input_data, columns=model_features)

# When the "Predict" button is clicked
if st.button("Tahminle"):
    # Get prediction probabilities
    probabilities = catboost_model.predict_proba(input_df)
    churn_probability = probabilities[0][1]  # Probability of the "churn" class

    # Adjust threshold if needed
    threshold = 0.5  # Adjust this value based on your analysis
    prediction = 1 if probabilities[0][1] > threshold else 0

    # Display prediction based on threshold
    if prediction == 1:
        st.write("Model, bu müşterinin **terk edeceğini** tahmin ediyor.")
    else:
        st.write("Model, bu müşterinin **terk etmeyeceğini** tahmin ediyor.")

    st.write("Müşteri Ayrılma Olasılığı:", churn_probability)

# Retrieve feature names and their importance scores from the model
feature_names = catboost_model.feature_names_
feature_importances = catboost_model.get_feature_importance()

# Check if feature_names and feature_importances have the same length
if len(feature_names) != len(feature_importances):
    st.write("Error: Mismatch between feature names and importance scores.")
else:
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    #     import matplotlib.pyplot as plt
    #     # Calculate SHAP values
    #     explainer = shap.TreeExplainer(catboost_model)
    #     shap_values = explainer.shap_values(input_df)

    #     # Plot top 10 features for this prediction
    #     st.title("SHAP Feature Importance (Impact on Prediction)")
    #     shap.summary_plot(shap_values, input_df, plot_type="bar", max_display=10, show=False)
    #     plt.gca().set_title("Top 10 Feature Impact on Churn Prediction (Direction & Magnitude)")
    #     st.pyplot(plt)
    #     shap.summary_plot(shap_values, input_df)