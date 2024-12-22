import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np

# Fungsi untuk memuat model
@st.cache_resource
def load_trained_model():
    return load_model("cnn_model.h5")  # Pastikan file model ada di direktori kerja

model = load_trained_model()

# Parameter input gambar
img_width, img_height = 128, 128  # Ukuran gambar yang sesuai dengan model

# Daftar kelas berdasarkan folder dataset
class_labels = [
    "Big Truck", 
    "City Car", 
    "Multi Purpose Vehicle", 
    "Sedan", 
    "Sport Utility Vehicle", 
    "Truck", 
    "Van"
]

# Fungsi untuk memproses gambar sebelum prediksi
def preprocess_image(image):
    image = image.resize((img_width, img_height))  # Ubah ukuran gambar
    image = np.array(image)  # Konversi ke array numpy
    image = image / 255.0  # Normalisasi piksel
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Fungsi untuk mendekode hasil prediksi
def decode_predictions(predictions):
    decoded = [class_labels[np.argmax(pred)] for pred in predictions]
    return decoded

# Konfigurasi halaman Streamlit
st.title("Klasifikasi Gambar Kendaraan")
st.write("Unggah gambar kendaraan untuk memprediksi kelasnya menggunakan model CNN yang telah dilatih.")

# Tombol untuk unggah banyak gambar
uploaded_files = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write("Gambar yang diunggah:")
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)
        images.append(image)

    # Button untuk prediksi
    if st.button("Prediksi Semua Gambar"):
        st.write("Memproses dan memprediksi gambar...")

        # Proses dan prediksi semua gambar
        predictions = []
        for image in images:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predictions.append(prediction)

        decoded_predictions = decode_predictions(predictions)

        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"{uploaded_file.name} -> *{decoded_predictions[i]}*")
