# PROJEK KLASIFIKASI KENDARAAN JALAN RAYA
# IDENTITAS
Ahmad Alif Dzaky Fadhilla
NIM : 202110370311452
## Link to Dataset
[Click Me^^](https://drive.google.com/file/d/1HhWLZYjQyvDvXBg2RldXVsGEccnRm69v/view?usp=sharing)
## Link to Model
[Click Me^^](https://drive.google.com/file/d/1HDUu1IAkLZ7YW0x1ose-3rY7TK09iis5/view?usp=drive_link)
## Setup Inveronment
Menggunakan 3.11 Python environment untuk menghindari Error pada Tensorflow dan Streamlit
## Windows Pwershell/VSCode Powershell
```
cd UAPPrakt
python --version
pip --version
python -m ensurepip --upgrade
```
## Streamlit
Menggunakan PIP Install dalam menginstall Streamlit
```
PIP install Streamlit
```
## Run Streamlit
```
Streamlit run app.py
```
# KLASIFIKASI KENDARAAN JALAN RAYA
Proyek ini bertujuan untuk mengembangkan sebuah sistem klasifikasi gambar yang dapat mengenali dan membedakan kategori gambar kendaraan, seperti "Big Truck," "City Car," "Multi Purpose Vehicle," "Sedan," "Sport Utility Vehicle," "Truck," dan "Van." Sistem ini dirancang untuk mendukung implementasi aplikasi yang memerlukan kemampuan identifikasi jenis kendaraan berdasarkan gambar, seperti sistem pengelompokan kendaraan untuk manajemen transportasi, survei data otomotif, atau aplikasi berbasis pembelajaran mesin di bidang otomotif.

Model yang digunakan dalam proyek ini adalah Convolutional Neural Network (CNN) dengan 3 lapisan utama untuk klasifikasi gambar kendaraan. 
### MobilenetV2 Architecture
![The-proposed-MobileNetV2-network-architecture](https://github.com/user-attachments/assets/b58f226d-a442-4752-bab7-f09225cea028)
### MobilenetV2 Implementation
```
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
mobilenet_model.trainable = False

model_mobilenet = Sequential([
    mobilenet_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
### CNN Architecture 
![41Q35ZMU](https://github.com/user-attachments/assets/a2f382af-a9d0-4588-8465-f187015f434c)
### CNN Impelementation
```
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])
```
### Dataset
Dataset terdiri atas 15,645 data yang terbagi menjadi 7 Kelas yaitu _'Big Truck', 'City Car', Sedan', 'Multi Purpose Vehicle', 'Sport Utility Car', 'Truck', 'Van'_.
### Model Accuracy Comparison
![Model Accuracy Comparison](https://github.com/user-attachments/assets/87b0d725-fb00-427d-bc8b-4d75c1f9e628)
### Model Loss Comparison
![Model Loss Comparison](https://github.com/user-attachments/assets/a6e5cf48-120e-4efd-937c-e100c2305e9e)

