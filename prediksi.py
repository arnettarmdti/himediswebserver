from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import streamlit as st
import joblib
import firebase_admin
from firebase_admin import credentials, db

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model3.pkl')

# Mengonfigurasi Firebase
cred = credentials.Certificate(
    {"type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]})
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://himedislogin-default-rtdb.firebaseio.com/'
})

# Mengakses Realtime Database
ref = db.reference('/predictions')  # Sesuaikan dengan path yang sesuai di database

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari request
        data = request.json
        sensor_value_ir = data['sensor_value_ir']
        sensor_value_red = data['sensor_value_red']

        # Mengolah data sensor untuk prediksi
        features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)

        # Melakukan prediksi
        prediction = model.predict(features)[0]

        # Konversi prediksi ke tipe float
        prediction = float(prediction)

        # Membuat data untuk dikirim ke Realtime Database Firebase
        result = {
            'sensor_value_ir': sensor_value_ir,
            'sensor_value_red': sensor_value_red,
            'prediction': prediction
        }

        # Menyimpan data ke Realtime Database
        ref.push(result)

        # Mengirimkan hasil prediksi kembali sebagai JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
