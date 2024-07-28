import streamlit as st
import xgboost as xgb
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model3.pkl')

# Mengonfigurasi Firebase
firebase_creds = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
}

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://himedislogin-default-rtdb.firebaseio.com/'
    })

# Mengakses Realtime Database
ref = db.reference('/predictions')

# Fungsi prediksi
def predict(sensor_value_ir, sensor_value_red):
    features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Aplikasi Streamlit
st.title("Prediksi Menggunakan Model XGBoost dan Firebase")

sensor_value_ir = st.number_input("Masukkan nilai sensor IR:", min_value=0.0, step=0.1)
sensor_value_red = st.number_input("Masukkan nilai sensor Red:", min_value=0.0, step=0.1)

if st.button("Prediksi"):
    prediction = predict(sensor_value_ir, sensor_value_red)
    result = {
        'sensor_value_ir': sensor_value_ir,
        'sensor_value_red': sensor_value_red,
        'prediction': prediction
    }
    ref.push(result)
    st.write("Hasil Prediksi:", prediction)
