import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.keras')
with open('geo_encoder.pkl', 'rb') as fp: geo_encoder = pickle.load(fp)
with open('scaler.pkl', 'rb') as fp: scaler = pickle.load(fp)

st.title('Customer Churn App')

geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Gender': gender == 'Male',
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}])

geo_encoded = geo_encoder.transform(pd.DataFrame({'Geography': [geography]}))
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out())

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

print('*'*10)
print(input_data.columns)
print('*'*10)
print(scaler.feature_names_in_)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5: st.write('The customer is likely to churn.')
else: st.write('The customer is not likely to churn.')
