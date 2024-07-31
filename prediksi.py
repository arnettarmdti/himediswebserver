import streamlit as st
import numpy as np
import joblib
import xgboost
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
ref = db.reference('/dataSensor')  # Path untuk data sensor dari Firebase

# Fungsi prediksi
def predict(sensor_value_ir, sensor_value_red):
    features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Aplikasi Streamlit
st.title("Prediksi Menggunakan Model XGBoost dan Firebase")

# Memantau perubahan pada nilai sensor dari Firebase
def monitor_sensor_changes(event):
    if event.data:
        sensor_value_ir = event.data.get('irValue')
        sensor_value_red = event.data.get('redValue')
        sensor_temperature = event.data.get('suhu')
        sensor_heartrate = event.data.get('bpm')

        # Jika ingin menyertakan sensor suhu dan bpm dalam prediksi atau data
        prediction = predict(sensor_value_ir, sensor_value_red)
        result = {
            'sensor_value_ir': sensor_value_ir,
            'sensor_value_red': sensor_value_red,
            'sensor_temperature': sensor_temperature,
            'sensor_heartrate': sensor_heartrate,
            'prediction': prediction
        }
        # Mengupdate path yang sama atau path lain jika diinginkan
        ref.push(result)

ref.listen(monitor_sensor_changes)

# Input nilai sensor (opsional)
sensor_value_ir = st.number_input("Masukkan nilai sensor IR:", min_value=0.0, step=0.1)
sensor_value_red = st.number_input("Masukkan nilai sensor Red:", min_value=0.0, step=0.1)
