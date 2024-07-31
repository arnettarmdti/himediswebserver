import streamlit as st
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
        'databaseURL': 'https://himedis-default-rtdb.firebaseio.com/'
    })

# Mengakses Realtime Database
data_sensor_ref = db.reference('/dataSensor')  # Path untuk data sensor dari Firebase
pred_ref = db.reference('/predictions')  # Path untuk hasil prediksi ke Firebase

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
        st.write("Data diterima dari Firebase:", event.data)  # Debug: tampilkan data yang diterima
        for key, data in event.data.items():
            sensor_value_ir = data.get('sensor_value_ir')
            sensor_value_red = data.get('sensor_value_red')
            
            if sensor_value_ir is None or sensor_value_red is None:
                st.write("Data sensor tidak lengkap:", data)  # Debug: tampilkan data jika tidak lengkap
                continue
            
            # Lakukan prediksi
            prediction = predict(sensor_value_ir, sensor_value_red)
            st.write("Hasil Prediksi:", prediction)  # Debug: tampilkan hasil prediksi
            
            result = {
                'sensor_value_ir': sensor_value_ir,
                'sensor_value_red': sensor_value_red,
                'prediction': prediction
            }
            
            # Kirim hasil prediksi ke path baru atau update path yang sama
            pred_ref.child(key).update(result)
            st.write("Hasil prediksi diperbarui di Firebase:", result)  # Debug: konfirmasi hasil pengiriman

data_sensor_ref.listen(monitor_sensor_changes)

# Tampilkan input nilai sensor (opsional, bisa dihilangkan jika inputnya hanya dari Firebase)
sensor_value_ir = st.number_input("Masukkan nilai sensor IR:", min_value=0.0, step=0.1)
sensor_value_red = st.number_input("Masukkan nilai sensor Red:", min_value=0.0, step=0.1)

if st.button("Prediksi"):
    prediction = predict(sensor_value_ir, sensor_value_red)
    result = {
        'sensor_value_ir': sensor_value_ir,
        'sensor_value_red': sensor_value_red,
        'prediction': prediction
    }
    # Tambahkan hasil prediksi ke path baru
    pred_ref.push(result)
    st.write("Hasil Prediksi:", prediction)
    st.write("Hasil prediksi dikirim ke Firebase:", result)  # Debug: konfirmasi hasil pengiriman
