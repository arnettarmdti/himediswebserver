import xgboost as xgb
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model3.pkl')

# Mengonfigurasi Firebase
firebase_creds = {
    "type": "YOUR_FIREBASE_TYPE",
    "project_id": "YOUR_PROJECT_ID",
    "private_key_id": "YOUR_PRIVATE_KEY_ID",
    "private_key": "YOUR_PRIVATE_KEY",
    "client_email": "YOUR_CLIENT_EMAIL",
    "client_id": "YOUR_CLIENT_ID",
    "auth_uri": "YOUR_AUTH_URI",
    "token_uri": "YOUR_TOKEN_URI",
    "auth_provider_x509_cert_url": "YOUR_AUTH_PROVIDER_X509_CERT_URL",
    "client_x509_cert_url": "YOUR_CLIENT_X509_CERT_URL"
}

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-database-url.firebaseio.com/'
    })

# Mengakses Realtime Database
ref_sensor = db.reference('/dataSensor')

# Fungsi prediksi
def predict(ir_value, red_value):
    features = np.array([ir_value, red_value]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Fungsi untuk memproses data dan memperbarui Firebase
def process_and_update_data():
    # Ambil semua data dari path '/dataSensor'
    sensor_data = ref_sensor.get()

    if sensor_data:
        for data_id, data in sensor_data.items():
            ir_value = data.get('irValue')
            red_value = data.get('redValue')
            temp = data.get('suhu')
            bpm = data.get('bpm')

            if ir_value is not None and red_value is not None:
                prediction = predict(ir_value, red_value)
                result = {
                    'irValue': ir_value,
                    'redValue': red_value,
                    'suhu': temp,
                    'bpm': bpm,
                    'prediction': prediction
                }
                # Update data di path '/dataSensor' dengan hasil prediksi
                ref_sensor.child(data_id).update(result)

# Jalankan fungsi untuk memproses data dan memperbarui Firebase
process_and_update_data()
