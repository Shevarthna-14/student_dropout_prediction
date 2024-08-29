import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = joblib.load(r'C:\Users\admin\Downloads\model.pkl')

# List of features based on the dataset
features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Nationality',
    'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation', 'Admission grade',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# Initialize Streamlit app
st.title('Student Dropout and Academic Success Prediction')

# Input fields for each feature
input_data = {}
for feature in features:
    input_data[feature] = st.text_input(f'Input {feature}', '')

d={0:'Dropout',1:'Enrolled',2:'Graduate'}
# Predict button
if st.button('Predict'):
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])
   
    # Preprocess the input data
    scaler = StandardScaler()
    input_df = scaler.fit_transform(input_df)
   
    # Make prediction
    prediction = np.argmax(model.predict(input_df), axis=-1)

    if prediction[0]==0:
    st.write('Prediction: Dropout')

    elif prediction[0]==1:
    st.write('Prediction: Enrolled')

    else:
    st.write('Prediction: Graduate')
