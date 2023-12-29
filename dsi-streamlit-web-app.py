# import libraries
import streamlit as st
import pandas as pd
import joblib

# load model pipeline object
model = joblib.load('model.joblib')

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for purchasing likelihood")

# Age input form
age = st.number_input(
    label = "01.  Enter customer age",
    min_value = 18,
    max_value = 105,
    value = 35
    )

# Gender input form
gender = st.radio(
    label = "02.  Enter customer gender",
    options = ["M", "F"]
    )

# Credit Score
credit_score = st.number_input(
    label = "03.  Enter customer credit score",
    min_value = 0,
    max_value = 1000,
    value = 500
    )

# Submit inputs to model
if st.button("Submit for Prediction"):
    
    # store our data in dataframe for prediction
    new_data = pd.DataFrame({"age" : [age],
                             "gender": [gender],
                             "credit_score": [credit_score]})
    
    # apply model pipeline to input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    #output prediction on webapp
    st.subheader(f"Based on the customer attributes \nOur model predicts a purchase probability of: {pred_proba: .0%}")






