import streamlit as st
import xgboost as xgb
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json

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
ref = db.reference('/predictions')

# Fungsi prediksi
def predict(sensor_value_ir, sensor_value_red):
    features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Kelas untuk menangani HTTP POST requests
class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        sensor_value_ir = data.get('sensor_value_ir')
        sensor_value_red = data.get('sensor_value_red')

        if sensor_value_ir is None or sensor_value_red is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid input')
            return

        prediction = predict(sensor_value_ir, sensor_value_red)
        result = {
            'sensor_value_ir': sensor_value_ir,
            'sensor_value_red': sensor_value_red,
            'prediction': prediction
        }
        ref.push(result)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = json.dumps({'prediction': prediction})
        self.wfile.write(response.encode())

# Menjalankan HTTP Server di thread terpisah
def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    httpd.serve_forever()

thread = threading.Thread(target=run_server)
thread.daemon = True
thread.start()

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
