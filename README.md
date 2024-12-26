### Analysis-of-Factors-Affecting-Student-Depression-Using-Demographic-Data-and-Academic-Achievement

### Analisis Faktor-Faktor yang Mempengaruhi Depresi Siswa Menggunakan Data Demografis dan Kinerja Akademik

#### 📖 Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi tingkat depresi siswa berdasarkan berbagai faktor yang berkontribusi terhadap kesehatan mental siswa. Sistem ini akan menggunakan beberapa model Machine Learning seperti Feedforward Neural Network (FNN) dan Random Forest untuk memprediksi apakah seorang siswa mengalami depresi atau tidak.

**Latar Belakang**

Klasifikasi ini didasarkan pada berbagai faktor yang memengaruhi tingkat depresi siswa, yang mencakup:
- 📈 **Faktor Demografis**
- 📖 **Faktor Akademik**
- 📞 **Faktor Lingkungan Sosial**

**Tujuan Utama**

Tujuan utama dari analisis ini adalah untuk memprediksi apakah seorang siswa mengalami depresi atau tidak, sehingga hasilnya dapat membantu dalam memberikan intervensi dini dan dukungan yang sesuai.

**Sumber Dataset**
- Dataset yang digunakan dalam proyek ini dapat diakses melalui [Kaggle - Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset/data).

**Deskripsi Kolom Dataset**
Dataset ini memiliki beberapa kolom penting yang digunakan dalam analisis:
- **Gender:** Jenis kelamin siswa (Male/Female).
- **Age:** Usia siswa.
- **Academic Pressure:** Tingkat tekanan akademik yang dirasakan siswa.
- **Work Pressure:** Tingkat tekanan kerja atau tanggung jawab tambahan yang dirasakan siswa.
- **CGPA:** Indeks Prestasi Kumulatif (Cumulative Grade Point Average) siswa.
- **Study Satisfaction:** Tingkat kepuasan siswa terhadap metode atau hasil belajar mereka.
- **Sleep Duration:** Rata-rata durasi tidur siswa per malam (dalam jam).
- **Dietary Habits:** Kebiasaan pola makan siswa (sehat atau tidak sehat).
- **Have you ever had suicidal thoughts?:** Pernyataan terkait pengalaman siswa tentang pikiran bunuh diri (Yes/No).
- **Work/Study Hours:** Jumlah rata-rata jam kerja atau belajar siswa dalam sehari.
- **Financial Stress:** Tingkat tekanan finansial yang dirasakan siswa.
- **Family History of Mental Illness:** Riwayat keluarga terkait penyakit mental (Yes/No).

#### 🔧 Langkah Instalasi

**Persyaratan Sistem**
- Python versi 3.10 atau lebih baru.
- Paket-paket yang tercantum dalam file `requirements.txt`.

**Instalasi Dependencies**

1. **Instalasi Otomatis**:
   Jalankan perintah berikut di terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Instalasi Manual**:
   Tambahkan dependensi satu per satu menggunakan perintah berikut:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

**Menjalankan Aplikasi Web**
1. Buka terminal, navigasikan ke direktori proyek.
2. Jalankan perintah berikut:
   ```bash
   streamlit run App.py
   ```
3. Aplikasi akan terbuka di browser pada localhost.

---

#### 🤖 Deskripsi Model

Proyek ini menggunakan beberapa model Machine Learning untuk memprediksi apakah seorang siswa mengalami depresi atau tidak. Berikut adalah model yang digunakan:

1. **Feedforward Neural Network (FNN)**
   - Model jaringan saraf sederhana di mana data bergerak satu arah dari input ke output tanpa loop.

2. **Random Forest Classifier**
   - Algoritma ensemble yang menggabungkan prediksi dari beberapa pohon keputusan untuk meningkatkan akurasi.

**Catatan:** Karena batas ukuran file, model yang telah dilatih tidak diunggah langsung dalam repositori ini. Model dapat diakses melalui link berikut:
[Google Drive - Model Terlatih](https://drive.google.com/drive/folders/1xac-lmiflPh8rsOic-6laqucvFuQmNuL?usp=share_link).

---

#### 🔍 Hasil dan Analisis

**Hasil Perbandingan Model**
Hasil evaluasi menunjukkan performa model sebagai berikut:

1. **Fully Connected Neural Network (FNN):**
   - Akurasi: 85.59%
   - Precision: 0.84 (kelas 0), 0.86 (kelas 1)
   - Recall: 0.80 (kelas 0), 0.90 (kelas 1)
   - F1-Score: 0.82 (kelas 0), 0.88 (kelas 1)

2. **Random Forest Classifier:**
   - Akurasi: 84.96%
   - Precision: 0.83 (kelas 0), 0.86 (kelas 1)
   - Recall: 0.80 (kelas 0), 0.88 (kelas 1)
   - F1-Score: 0.82 (kelas 0), 0.87 (kelas 1)

**Visualisasi Hasil**

1. **Confusion Matrix:** Menampilkan distribusi prediksi benar dan salah untuk setiap model.
   ![Unknown-35](https://github.com/user-attachments/assets/5e02b089-515d-4b26-ab34-5b9e2a754e0f)
   ![Unknown-36](https://github.com/user-attachments/assets/a8838ab4-23d1-49e9-8bf0-79579ac1fdde)

3. **Classification Report:** Menyediakan rincian metrik evaluasi untuk setiap kelas.
   
---

Dengan hasil ini, model **Fully Connected Neural Network (FNN)** menunjukkan performa yang sedikit lebih baik dibandingkan **Random Forest Classifier**, terutama pada metrik recall dan f1-score untuk kelas 1 (Depresi). Sistem ini dapat diintegrasikan dalam aplikasi untuk membantu identifikasi dini siswa yang membutuhkan perhatian lebih.

---
**Penyusun:**

Thyara Mahadewi

202110370311069
