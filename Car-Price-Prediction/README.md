# Laporan Proyek Machine Learning - Dwi NurCahyo Purbonegoro

## Domain Proyek: Prediksi Harga Mobil Bekas

![jual_mobil](https://github.com/user-attachments/assets/94bb033a-67b1-4363-9599-34d477062faa)

### Latar Belakang

Harga jual mobil bekas sangat dipengaruhi oleh berbagai faktor utama seperti **merk kendaraan**, **jarak tempuh**, dan **usia kendaraan**. Mobil dengan **jarak tempuh yang rendah**, **merk kelas atas**, atau model **klasik** dengan jumlah terbatas biasanya memiliki nilai jual yang lebih tinggi.

Proyek ini bertujuan mengembangkan sebuah sistem prediksi harga mobil bekas berbasis machine learning yang dapat:

* Memberikan estimasi harga mobil berdasarkan data teknis dan kondisi mobil
* Membantu pembeli dan penjual menetapkan harga yang sesuai pasar
* Mempermudah dan mempercepat proses transaksi jual beli mobil bekas

---

## Pemahaman Bisnis

### Permasalahan

1. Bagaimana memprediksi harga mobil berdasarkan merk, jarak tempuh, dan umur kendaraan?
2. Bagaimana menganalisis pengaruh gabungan fitur-fitur tersebut terhadap harga jual mobil?

### Tujuan

1. Membangun model prediksi harga mobil bekas dengan tingkat akurasi tinggi.
2. Melakukan preprocessing data dan pembuatan fitur yang relevan untuk meningkatkan performa model.
3. Mengaplikasikan algoritma regresi berbasis ensemble seperti Random Forest dan XGBoost untuk prediksi.

---

## Pemahaman Data

Tahap ini berfokus pada pemahaman konten dataset dan evaluasi kualitas data untuk mendukung analisis dan pemodelan.

Sumber dataset: [https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)

Dataset berisi data kendaraan dengan berbagai atribut spesifikasi dan kondisi mobil.

Tabel berikut menunjukkan fitur yang tersedia dan penjelasannya:

| Fitur            | Keterangan                                      |
| ---------------- | ----------------------------------------------- |
| ID               | Identifikasi unik setiap mobil                  |
| Price            | Harga mobil (target yang ingin diprediksi)      |
| Levy             | Pajak kendaraan                                 |
| Manufacturer     | Merek kendaraan                                 |
| Model            | Tipe/model kendaraan                            |
| Prod. year       | Tahun pembuatan mobil                           |
| Category         | Kategori kendaraan (SUV, Sedan, dll.)           |
| Leather interior | Apakah memiliki interior kulit (Yes/No)         |
| Fuel type        | Tipe bahan bakar (Petrol, Diesel, Hybrid, dll.) |
| Engine volume    | Kapasitas mesin dalam liter                     |
| Mileage          | Jarak tempuh kendaraan (dalam kilometer)        |
| Cylinders        | Jumlah silinder mesin                           |
| Gear box type    | Jenis transmisi (Manual/Automatic)              |
| Drive wheels     | Sistem penggerak roda (FWD, RWD, AWD)           |
| Doors            | Jumlah pintu kendaraan                          |
| Wheel            | Posisi setir (kiri/kanan)                       |
| Color            | Warna kendaraan                                 |
| Airbags          | Jumlah kantong udara keselamatan                |

### Eksplorasi Data Awal

* Fitur seperti Manufacturer, Model, Category, Leather interior, Fuel type, Gear box type, Drive wheels, Doors, Wheel, dan Color merupakan variabel kategorikal yang perlu dilakukan encoding agar dapat digunakan dalam pemodelan.
* Data numerik seperti Price, Prod. year, Cylinders, dan Airbags memiliki skala nilai yang beragam sehingga perlu dinormalisasi untuk mendukung kinerja model machine learning.
* Kolom Levy berformat string dan mengandung nilai kosong serta tanda "-", perlu dibersihkan dan dikonversi menjadi numerik.
* Engine volume berisi campuran angka dan teks seperti "2.0 Turbo" yang harus diproses untuk memisahkan angka dan informasi tambahan.
* Data Mileage mengandung satuan "km" sehingga perlu dihapus agar dapat dikonversi ke tipe numerik.
* Nilai di kolom Doors tidak konsisten, misalnya "04-May", perlu distandarisasi menjadi angka jumlah pintu yang valid.
* Distribusi data Price menunjukkan adanya skewness, sehingga transformasi logaritmik dapat digunakan untuk mengurangi kemiringan distribusi dan meningkatkan stabilitas model.

![gambar2](https://github.com/user-attachments/assets/e79e83d5-344e-416a-9b98-0438c0f8a294)

Dataset tidak memiliki nilai yang hilang.

![gambar3](https://github.com/user-attachments/assets/45570e37-f78f-4dff-b2bf-47925bacdfb2)

Terdapat 313 data duplikat. Semua kolom berisi tipe data objek, integer, dan float dengan total 19.237 baris data valid.

![gambar4](https://github.com/user-attachments/assets/504f1c6c-2526-4974-8fdc-8b77d6c2d073)

Output fungsi df.nunique() menunjukkan jumlah nilai unik pada setiap kolom. Contohnya, kolom ID memiliki 18.924 nilai unik, menandakan hampir semua entri berbeda. Kolom Price memiliki 2.315 variasi harga. Manufacturer dan Model memiliki 65 dan 1.590 nilai unik, menunjukkan ragam merek dan tipe mobil yang luas. Kolom dengan variasi terbatas misalnya Leather interior (2 kategori) dan Wheel (2 posisi). Fuel type (7 kategori) dan Color (16 warna) memiliki variasi menengah. Informasi ini berguna untuk memahami keragaman data sebelum pemodelan.

![image](https://github.com/user-attachments/assets/a76d9c73-aec8-4b37-8c64-ec14d7b533c0)

Ringkasan statistik dari df.describe() menunjukkan:

* Rata-rata harga mobil sekitar 18.556 dengan median 13.172 dan nilai maksimum sangat tinggi, mengindikasikan keberadaan outlier.
* Tahun produksi rata-rata 2010 dengan rentang dari 1939 sampai 2020.
* Cylinders paling sering 4 silinder, maksimum sampai 16.
* Airbags bervariasi dari 0 sampai 16 dengan rata-rata sekitar 6.
  Statistik ini memberikan gambaran tentang penyebaran dan anomali pada data numerik.

![gambar5](https://github.com/user-attachments/assets/205eded5-ab5d-44a8-814d-add601054a87)

Output untuk kategori unik dalam fitur kategorikal memperlihatkan variasi data sebagai berikut:

* Levy yang sudah dibersihkan berisi nilai pajak kendaraan dalam format numerik.
* Manufacturer meliputi berbagai merk mulai dari 'TOYOTA', 'BMW', 'LEXUS', hingga 'FERRARI'.
* Model mencakup ribuan tipe mobil spesifik.
* Category menunjukkan jenis bodi seperti 'Sedan', 'Jeep', 'Hatchback', dll.
* Leather interior hanya ada dua opsi: 'Yes' atau 'No'.
* Fuel type mencakup jenis bahan bakar seperti 'Petrol', 'Diesel', 'Hybrid', dan 'Hydrogen'.
* Gear box type antara 'Automatic' dan 'Manual'.
* Drive wheels memiliki tipe seperti 'Front', 'Rear', atau '4x4'.
* Doors memiliki nilai tidak standar seperti '04-May' atau '>5' yang memerlukan normalisasi.
* Wheel mengacu pada posisi kemudi kiri atau kanan.
* Color beragam warna mobil.
* Engine volume mencampur kapasitas mesin dan keterangan turbo, misalnya '2.0 Turbo'.
* Mileage mengandung jarak tempuh dalam format string seperti '125,000 km'.
  Pemahaman ini penting sebelum proses encoding dan transformasi data.

![gambar6](https://github.com/user-attachments/assets/2375e2ec-c88b-4a78-a2af-37b2fdc427d2)

Contoh lima baris pertama data menunjukkan berbagai atribut yang sudah dibersihkan dan siap untuk proses selanjutnya.

---


## Data Preparation

### Proses yang Dilakukan:

* **Duplikasi Dataset**
  Salinan data dibuat terlebih dahulu agar data asli tetap utuh saat proses persiapan data berlangsung.

![gambar7](https://github.com/user-attachments/assets/7d2e9605-a6ae-4831-bdc7-27f722a86688)

* **Menghilangkan Data Duplikat**
  Pada tahap ini, data yang memiliki baris duplikat dihapus menggunakan fungsi `drop_duplicates()`. Proses ini penting agar setiap record dalam dataset unik dan menghindari bias akibat pengulangan data. Setelah pembersihan, jumlah data tetap 18.924 baris, menunjukkan tidak ada baris yang sama persis. Tahap ini merupakan bagian dari pra-pemrosesan untuk memastikan kualitas data yang dipakai.

![gambar8](https://github.com/user-attachments/assets/da19550e-29c6-4e51-b6cc-1fd43b6a7f83)

* **Pembersihan dan Transformasi Data**

  Dilakukan pembersihan nilai yang tidak konsisten serta pembuatan fitur baru yang berguna.

  Kolom `Levy` dibersihkan dengan mengganti simbol '-' menjadi `NaN`, menghapus tanda koma, dan dikonversi ke tipe data float. Nilai kosong diisi menggunakan median agar distribusi tetap seimbang.
  Untuk kolom `Mileage`, satuan ' km' dan koma dihapus sebelum diubah ke tipe numerik float.

  Pada `Engine volume`, fitur baru bernama `Is_Turbo` dibuat untuk menandai ada tidaknya turbo pada mesin. Setelah itu, teks 'Turbo' dibuang dari kolom asli dan konversi ke float dilakukan.

  Nilai pada `Doors` yang berupa format teks seperti '02-Mar' dan '04-May' dikonversi ke angka menggunakan pemetaan, sementara nilai yang tidak dikenali diasumsikan 4 pintu.
  Terakhir, `Leather interior` diubah menjadi nilai numerik biner: 1 untuk 'Yes' dan 0 untuk 'No'.

![gambar9](https://github.com/user-attachments/assets/ce07c0da-dc56-4918-b5a9-ca89ab628d00)

* **Pembuatan Fitur Baru dan Normalisasi**

  Beberapa fitur tambahan dibuat untuk memperkaya data dan mempermudah analisis/modeling.

  Fitur **`Car_Age`** dihitung berdasarkan selisih antara tahun 2025 dengan tahun produksi mobil, menggambarkan umur kendaraan.
  Fitur **`Mileage_per_Year`** dibuat sebagai rata-rata jarak tempuh per tahun. Nilai yang tak terhingga atau kosong diganti dengan median.

  Fitur kategori baru **`Age_Group`** dibuat untuk mengelompokkan usia mobil menjadi tiga kelas: *New*, *Medium*, dan *Old* berdasarkan rentang tahun.
  Ditambahkan juga flag **`Is_Luxury`** untuk menandai mobil dari merek premium seperti BMW, MERCEDES-BENZ, dan lainnya.

  Untuk menyederhanakan kategori, kolom **`Manufacturer`** dikelompokkan menjadi 10 merek teratas, sisanya diberi label `'Other'`.
  Fitur **`Fuel_Efficiency_Proxy`** dibuat sebagai perbandingan usia mobil dengan volume mesin untuk menggambarkan efisiensi bahan bakar. Nilai ekstrem dan kosong diatasi serupa fitur sebelumnya.

![gambar10](https://github.com/user-attachments/assets/5a04de92-b8ea-4526-bab1-f1b286c09b1e)

* **Standarisasi, Transformasi Target, dan Encoding**

  Pada tahap ini, fitur numerik dinormalisasi, target diubah, dan fitur kategori dikonversi agar model dapat memprosesnya dengan baik.

  Fitur numerik seperti `Levy`, `Mileage`, `Engine volume`, `Airbags`, dan lain-lain distandarisasi memakai **StandardScaler** agar memiliki rata-rata 0 dan deviasi standar 1, guna menghindari bias akibat skala yang berbeda antar fitur.

  Target variabel `Price` diubah dengan transformasi logaritma natural (log(1 + x)) untuk mengurangi skewness dan membuat distribusi lebih stabil sehingga model lebih mudah mempelajari pola harga.

  Fitur kategori seperti `Manufacturer`, `Fuel type`, dan `Gear box type` diubah menjadi variabel dummy (one-hot encoding), dengan `drop_first=True` untuk menghindari multikolinearitas. Ini membuat data siap dipakai dalam algoritma machine learning berbasis numerik.

  ![gambar11](https://github.com/user-attachments/assets/32a22564-401d-44e2-abb2-30bd9c4cd970)

* **Pembagian Data**

  Data dipisahkan antara fitur dan target, lalu dibagi menjadi data training dan testing.

  Kolom `Price` dijadikan target (`y`), sedangkan fitur lainnya (kecuali `ID`) dijadikan variabel input (`X`).
  Dataset dibagi menjadi 90% data pelatihan dan 10% data pengujian menggunakan `train_test_split` dengan `random_state=42` untuk menjaga konsistensi hasil.

  Nama kolom pada dataset hasil split dibersihkan dari karakter-karakter yang tidak diperbolehkan seperti `[]` dan `<>` agar tidak mengganggu proses modeling atau penyimpanan data.
  Ukuran akhir dataset untuk training dan testing serta jumlah fitur setelah encoding ditampilkan sebagai verifikasi suksesnya pembagian data.

![gambar12](https://github.com/user-attachments/assets/d85e7d84-9323-466d-8400-11a72ad594d8)

---

---

## Modeling

### Inisialisasi Model

Kamu menggunakan tiga jenis model regresi populer untuk memprediksi harga mobil (atau smartphone, sesuaikan konteksmu):

![gambar13](https://github.com/user-attachments/assets/2ee69e3f-8fef-49f2-b53f-5ca6a237df02)

### Model 1: Linear Regression

* **Deskripsi:**
  Model regresi linear mencari garis lurus terbaik yang menjelaskan hubungan antara fitur (variabel independen) dan target (harga).
* **Persamaan:**

  $$
  y = β_0 + β_1 x_1 + β_2 x_2 + \cdots + β_n x_n
  $$
* **Parameter:**
  Default `LinearRegression()` dari sklearn, seperti `fit_intercept=True` dan `normalize=False`.
* **Kelebihan:**
  Cepat, mudah diinterpretasi, cocok untuk data dengan hubungan linear.
* **Kekurangan:**
  Kurang efektif untuk hubungan non-linear, sensitif terhadap outlier dan multikolinearitas.

---

### Model 2: Random Forest Regressor

* **Deskripsi:**
  Algoritma ensemble yang menggabungkan prediksi dari banyak pohon keputusan (decision trees). Tiap pohon dibuat dengan subset data dan fitur secara acak.
* **Parameter:**

  * `n_estimators=100` (jumlah pohon)
  * `random_state=42` untuk reproducibility
  * Parameter lain default (`max_depth=None`, dll)
* **Kelebihan:**
  Handal untuk data non-linear, tahan terhadap overfitting, dan memberikan fitur penting.
* **Kekurangan:**
  Model besar, prediksi lambat, kurang interpretatif dibanding regresi linear.

---

### Model 3: XGBoost Regressor

* **Deskripsi:**
  Boosting tree-based model yang membangun model secara bertahap dengan memperbaiki kesalahan model sebelumnya menggunakan gradient descent.
* **Parameter:**

  * `n_estimators=100`
  * `random_state=42`
  * `verbosity=0`
  * Parameter lain default (e.g., `learning_rate=0.3`, `max_depth=6`)
* **Kelebihan:**
  Akurat, menangani missing values, mendukung regularisasi.
* **Kekurangan:**
  Memerlukan tuning parameter, komputasi lebih berat.

---

## Latih dan Evaluasi Model
![gambar14](https://github.com/user-attachments/assets/799489e6-d438-4362-95a6-18def4306147)

* **Training:**
  Melatih tiap model menggunakan data pelatihan (`X_train`, `y_train`).

* **Prediction:**
  Menghasilkan prediksi harga (`y_pred`) pada data testing (`X_test`).

* **Evaluasi:**
  Menggunakan metrik evaluasi untuk mengukur performa model terhadap target harga yang sudah ditransformasi logaritmik (`log(Price)`).

---


Berikut adalah rangkuman lengkap dari bagian **Evaluasi Model** yang kamu buat, lengkap dengan interpretasi dan kaitan ke aspek bisnis:

---

## Metrik Evaluasi

Untuk mengukur kualitas prediksi model regresi, digunakan tiga metrik utama:

| Metrik                             | Deskripsi                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **MAE (Mean Absolute Error)**      | Rata-rata nilai absolut selisih antara prediksi dan nilai asli. Semakin kecil, semakin akurat.               |
| **RMSE (Root Mean Squared Error)** | Akar rata-rata kuadrat selisih prediksi dan nilai asli. Lebih sensitif terhadap outlier dibanding MAE.       |
| **R² (R-squared Score)**           | Proporsi variasi data target yang berhasil dijelaskan model. Nilai mendekati 1 menandakan model sangat baik. |

Metrik ini dipilih karena cocok untuk masalah regresi numerik dan memberi gambaran lengkap tentang akurasi dan kemampuan model dalam menangkap pola data.

---

## Hasil Evaluasi dan Perbandingan Model

| Model                   | MAE      | RMSE     | R²       |
| ----------------------- | -------- | -------- | -------- |
| Linear Regression       | 0.972879 | 1.402952 | 0.247545 |
| Random Forest Regressor | 0.442960 | 0.892695 | 0.695350 |
| XGBoost Regressor       | 0.598113 | 0.991828 | 0.623931 |

---

## Interpretasi Hasil

* **Linear Regression**
  R² sekitar 0.247, artinya model hanya mampu menjelaskan sekitar 24.7% variasi harga. Ini menunjukkan model linear kurang efektif menangkap hubungan fitur dan harga yang kompleks.

* **Random Forest Regressor**
  Performa terbaik dengan MAE terendah dan R² tertinggi (\~0.695). Model ini berhasil menangkap pola non-linear dan kompleksitas fitur, memberikan prediksi paling akurat dan stabil.

* **XGBoost Regressor**
  Performa baik, namun sedikit kalah dari Random Forest dalam semua metrik. Tetap merupakan alternatif kompetitif.

---

## Model Terbaik

**Random Forest Regressor** dipilih sebagai model utama karena memberikan keseimbangan terbaik antara akurasi dan stabilitas dalam konteks proyek ini.

---

## Keterkaitan dengan Business Understanding

Evaluasi ini menghubungkan hasil modeling dengan tujuan bisnis:

1. **Menjawab problem statement?**
   ✅ Ya, model memprediksi harga smartphone berdasarkan fitur produk secara akurat.

2. **Mencapai goals?**
   ✅ Ya, dengan R² \~0.695, model cukup akurat untuk digunakan dalam produksi dan pengambilan keputusan.

3. **Dampak solusi?**
   ✅ Ya, solusi berdampak pada:

   * **Pengguna:** Membantu membandingkan harga pasar dengan harga model prediksi, mencegah pembelian produk dengan harga terlalu mahal.
   * **Pelaku bisnis/platform:** Mendukung strategi penentuan harga otomatis yang berbasis data fitur produk, meningkatkan efisiensi dan potensi penjualan.

---


Berikut rangkuman lengkap dan terstruktur dari bagian **Tuning dan Optimasi**, **Feature Importance**, serta **Kesimpulan dan Rekomendasi** yang kamu sampaikan:

---

## Tuning dan Optimasi Hyperparameter
![gambar15](https://github.com/user-attachments/assets/1040ed17-506a-4326-b220-7276e357add5)

### Random Forest

* **Metode:** GridSearchCV dengan cross-validation (cv=2), metrik evaluasi: R².
* **Parameter yang diuji:**

  * `max_depth`: 10, 20
  * `n_estimators`: 100
  * `min_samples_split` dan `min_samples_leaf`: parameter kontrol pembagian dan ukuran daun.
* **Hasil terbaik:**

  * `max_depth=20`
  * R² terbaik sekitar **0.598**
* **Catatan:** Meskipun hasil R² GridSearch ini sedikit lebih rendah dari hasil awal (0.695), tuning membantu memastikan konfigurasi terbaik untuk generalisasi model.

---

### XGBoost

* **Metode:** GridSearchCV dengan cv=2, menggunakan metrik MAE negatif (karena GridSearch memaksimalkan skor).
* **Parameter yang diuji:**

  * `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`
* **Hasil terbaik:**

  * `max_depth=6`
  * `learning_rate=0.05`
  * `n_estimators=200`
  * MAE negatif terbaik sekitar **-0.673**
* **Interpretasi:** Penurunan learning rate dengan peningkatan estimators membantu meningkatkan presisi model.

---

## Feature Importance (XGBoost)
![gambar_grafik](https://github.com/user-attachments/assets/67def547-8684-498b-bb36-7d5bdd613453)

* Visualisasi menunjukkan fitur-fitur yang paling berpengaruh terhadap prediksi harga (log(Price)).
* Contoh insight: fitur seperti **Gear box type\_Tiptronic** memiliki kontribusi signifikan.
* **Manfaat:**

  * Memahami fitur kunci yang menentukan harga.
  * Membantu pengambilan keputusan bisnis dengan fokus pada fitur penting.
  * Memungkinkan penyederhanaan model dengan feature selection.
  * Meningkatkan interpretabilitas model untuk stakeholder.

---

## Kesimpulan

* Model regresi, terutama Random Forest Regressor, berhasil memprediksi harga mobil bekas dengan akurasi yang cukup baik.
* Faktor paling dominan: **tahun pembuatan**, **jarak tempuh**, dan **jenis bahan bakar**.
* Preprocessing data (pembersihan, encoding, feature engineering) sangat berpengaruh terhadap performa model.

---

## Rekomendasi

1. **Perluasan Dataset:**
   Tambahkan fitur lain seperti lokasi, kondisi kendaraan, riwayat servis untuk meningkatkan akurasi.

2. **Integrasi Sistem:**
   Implementasikan model ke platform jual beli mobil agar pengguna mendapat estimasi harga otomatis.

3. **Evaluasi Berkala dan Retraining:**
   Agar model tetap akurat dan relevan dengan tren pasar terbaru.

4. **Eksplorasi Teknik Lanjutan:**
   Gunakan metode ensemble atau boosting lain (misal LightGBM) untuk peningkatan performa.

---

