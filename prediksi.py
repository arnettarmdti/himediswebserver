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
data_sensor_ref = db.reference('/dataSensor')
predictions_ref = db.reference('/predictions')

# Fungsi prediksi
def predict(sensor_value_ir, sensor_value_red):
    features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Kelas untuk menangani HTTP POST requests
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/process_data':
            # Ambil data dari Firebase
            data_snapshot = data_sensor_ref.get()
            if data_snapshot:
                for key, data in data_snapshot.items():
                    sensor_value_ir = data.get('sensor_value_ir')
                    sensor_value_red = data.get('sensor_value_red')

                    if sensor_value_ir is None or sensor_value_red is None:
                        continue

                    # Prediksi
                    prediction = predict(sensor_value_ir, sensor_value_red)
                    result = {
                        'sensor_value_ir': sensor_value_ir,
                        'sensor_value_red': sensor_value_red,
                        'prediction': prediction
                    }

                    # Kirim hasil prediksi ke Firebase
                    predictions_ref.push(result)
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'Data processed and predictions sent'})
                self.wfile.write(response.encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'No data found'})
                self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'status': 'Not Found'})
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

# Menampilkan status
st.write("Server HTTP sedang berjalan dan memproses data sensor dari Firebase.")

# Notifikasi
st.write("HTTP GET request ke `/process_data` untuk memproses data sensor dan mengirim hasil prediksi.")
