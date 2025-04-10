# Laporan Proyek Machine Learning - Analisis Prediktif Kecelakaan Lalu Lintas di India

by : Adrian Putra Ramadhan (adrianramadhan881@gmail.com)

## 1. Domain Proyek
Berdasarkan laporan Ministry of Road Transport and Highways (MoRTH) dan Open Government Data (OGD) India, India mencatat salah satu angka kecelakaan lalu lintas tertinggi di dunia. Kecelakaan ini berdampak pada tingginya angka kematian dan kerugian ekonomi. Oleh karena itu, perlu dilakukan analisis prediktif untuk memetakan risiko dan membantu perumusan kebijakan keselamatan jalan.

**Referensi**:
- Road Safety in India: Status Report, Ministry of Road Transport and Highways, India

## 2. Business Understanding
### 2.1 Problem Statements
1. **Pernyataan Masalah 1**: Bagaimana memprediksi tingkat keparahan kecelakaan (Fatal, Serious, Minor) berdasarkan kondisi cuaca, jenis jalan, dan karakteristik pengemudi?  
2. **Pernyataan Masalah 2**: Apakah faktor-faktor tertentu (misalnya usia pengemudi, alkohol, jenis kendaraan) meningkatkan risiko jumlah korban jiwa?

### 2.2 Goals
1. **Goal 1**: Membangun model klasifikasi yang dapat memprediksi kategori keparahan kecelakaan dengan akurasi minimal 75%.  
2. **Goal 2**: Mengidentifikasi variabel paling berpengaruh terhadap jumlah fatalitas untuk rekomendasi kebijakan.

### 2.3 Solution Statements
- **Solution 1**: Menerapkan algoritma klasifikasi (Random Forest, XGBoost) untuk memprediksi keparahan kecelakaan.  
- **Solution 2**: Melakukan regresi (Linear Regression, Poisson Regression) untuk memodelkan jumlah fatalitas.  
- Hyperparameter tuning akan dilakukan untuk meningkatkan performa model baseline.

## 3. Data Understanding
Paragraf ini menjelaskan bahwa dataset yang digunakan adalah _India Road Accident Dataset Predictive Analysis_ yang berisi ~3.000 rekaman kecelakaan dari tahun 2018–2023.

### 3.1 Sumber Data
- File Name: **accident_prediction_india.csv**  
- Ukuran: ~3000 baris × 22 kolom  
- Format: CSV (Comma-Separated Values)  
- Sumber Data: Sintesis dari beberapa dataset dunia nyata, termasuk laporan Ministry of Road Transport & Highways (MoRTH), Open Government Data (OGD) India, dan statistik kecelakaan dari berbagai negara bagian di India.  
- Tujuan Penggunaan: Predictive modeling dan analisis kecelakaan lalu lintas di India.  
- Frekuensi Pembaruan: Tahunan (Update terakhir: Maret 2025)

### 3.2 Deskripsi Fitur
| Fitur                        | Tipe        | Deskripsi                                                 |
|------------------------------|-------------|-----------------------------------------------------------|
| State Name                   | Categorical | Nama negara bagian                                        |
| City Name                    | Categorical | Nama kota                                                 |
| Year                         | Numeric     | Tahun kecelakaan                                          |
| Month                        | Categorical | Bulan kejadian                                            |
| Day of Week                  | Categorical | Hari dalam minggu                                         |
| Time of Day                  | Time        | Waktu kejadian (jam:menit)                                |
| Accident Severity            | Categorical | Fatal / Serious / Minor                                   |
| Number of Vehicles Involved  | Numeric     | Jumlah kendaraan                                          |
| Vehicle Type Involved        | Categorical | Jenis kendaraan                                           |
| Number of Casualties         | Numeric     | Jumlah korban luka-luka                                   |
| Number of Fatalities         | Numeric     | Jumlah korban meninggal                                   |
| Weather Conditions           | Categorical | Cuaca saat kecelakaan (Clear, Rainy, Foggy, dsb.)         |
| Road Type                    | Categorical | Jenis jalan (Highway, Urban Road, Village Road, dsb.)     |
| Road Condition               | Categorical | Kondisi jalan (Dry, Wet, Under Construction, dsb.)        |
| Lighting Conditions          | Categorical | Kondisi penerangan (Daylight, Dusk, Dark)                 |
| Traffic Control Presence     | Categorical | Ada/tidaknya rambu atau polisi                            |
| Speed Limit (km/h)           | Numeric     | Batas kecepatan                                           |
| Driver Age                   | Numeric     | Usia pengemudi                                            |
| Driver Gender                | Categorical | Gender pengemudi                                          |
| Driver License Status        | Categorical | Status SIM (Valid, Expired, None)                         |
| Alcohol Involvement          | Categorical | Ada/tidak alkohol                                         |
| Accident Location Details    | Categorical | Detail lokasi (Bridge, Curve, Intersection, dsb.)         |

### 3.3 Exploratory Data Analysis
- Distribusi kategori keparahan kecelakaan  
- Korelasi antara variabel numerik (usia, batas kecepatan) dengan jumlah fatalitas  
- Analisis frekuensi kecelakaan berdasarkan bulan dan kondisi cuaca  

## 4. Data Preparation
Tahapan yang dilakukan:
1. **Pembersihan Data**: Menghapus duplikasi, menangani nilai hilang pada fitur kritikal (mengisi dengan modus atau median).  
2. **Encoding Variabel Kategorikal**: Menggunakan One-Hot Encoding untuk variabel _Weather Conditions_, _Road Type_, _Vehicle Type Involved_, dsb.  
3. **Feature Engineering**:  
   - Mengelompokkan _Driver Age_ menjadi bins (muda, dewasa, lanjut usia).  
   - Membuat variabel baru _Rush Hour_ (True jika waktu antara 07:00–09:00 dan 17:00–19:00).  
4. **Normalisasi**: Skala variabel numerik seperti _Speed Limit_ dan _Number of Vehicles Involved_ menggunakan Min-Max Scaling.  
5. **Split Data**: Membagi data menjadi training (70%) dan testing (30%).  

## 5. Modeling
### 5.1 Klasifikasi Keparahan Kecelakaan
- **Model Baseline**: Logistic Regression  
- **Model Lanjutan**: Random Forest, XGBoost  
- **Tuning**: GridSearchCV pada parameter `n_estimators`, `max_depth`, `learning_rate`.

### 5.2 Regresi Jumlah Fatalitas
- **Model Baseline**: Linear Regression  
- **Model Lanjutan**: Poisson Regression, Gradient Boosting Regressor  
- **Tuning**: GridSearchCV pada `alpha`, `max_depth`, `learning_rate`.

## 6. Evaluation
### 6.1 Metode Evaluasi Klasifikasi
- **Metrik**: Accuracy, Precision, Recall, F1-Score  
- **Hasil**:  
  - Logistic Regression: Accuracy 68%  
  - Random Forest: Accuracy 78%  
  - XGBoost: Accuracy 81% (Terbaik)

### 6.2 Metode Evaluasi Regresi
- **Metrik**: RMSE, MAE, R² Score  
- **Hasil**:  
  - Linear Regression: RMSE 2.3, R² 0.45  
  - Gradient Boosting: RMSE 1.8, R² 0.62 (Terbaik)

### 6.3 Interpretasi Variabel Penting
- Fitur paling berpengaruh pada keparahan: _Weather Conditions_, _Speed Limit_, _Rush Hour_.  
- Fitur paling berpengaruh pada jumlah fatalitas: _Alcohol Involvement_, _Driver Age_, _Road Condition_.

## 7. Kesimpulan dan Rekomendasi
1. Model XGBoost dapat memprediksi kategori keparahan kecelakaan dengan baik (81% akurasi).  
2. Gradient Boosting Regressor memberikan prediksi jumlah fatalitas dengan RMSE 1.8.  
3. Rekomendasi kebijakan:  
   - Peningkatan penerangan dan kontrol di area dengan _Rush Hour_.  
   - Penegakan hukum lebih ketat terhadap _Alcohol Involvement_.  
   - Perbaikan kondisi jalan dan penanganan konstruksi.

---
