# Deteksi Bahasa Isyarat (Sign Language Detection)

Implementasi sistem deteksi bahasa isyarat *real-time* menggunakan Python, OpenCV, MediaPipe, dan Scikit-Learn. Proyek ini mendemonstrasikan alur kerja *machine learning* (ML) *end-to-end*, mulai dari pengumpulan data hingga inferensi model secara langsung (real-time).

Proyek ini dikembangkan dengan merujuk pada tutorial teknis dari [Computer Vision Engineer di YouTube](https://www.youtube.com/watch?v=MJCSjXepaAM).

---

## Kemampuan Utama (Key Capabilities)

-   **Ekstraksi Fitur Tangan Real-Time:** Pemanfaatan Google MediaPipe untuk mendeteksi dan mengekstrak 21 titik *landmark* tangan (hand landmarks) secara akurat dari *video feed*.
-   **Alur Pengumpulan Data:** Skrip terstruktur untuk mengakuisisi dan memberi label pada set data gambar (frame) untuk setiap kelas isyarat (gesture) melalui webcam.
-   **Pelatihan Model Klasifikasi:** Implementasi model *machine learning* (Random Forest) menggunakan Scikit-Learn untuk mengklasifikasikan *landmark* tangan yang telah diekstraksi.
-   **Inferensi Real-Time:** Aplikasi model yang telah dilatih pada input webcam untuk melakukan prediksi isyarat tangan secara *on-the-fly*.

---

## Tumpukan Teknologi (Tech Stack)

-   **Python 3.x**
-   **OpenCV:** Digunakan untuk akuisisi dan pemrosesan *video stream* dari webcam.
-   **MediaPipe:** Digunakan untuk deteksi *landmark* tangan berkinerja tinggi.
-   **Scikit-Learn (sklearn):** Digunakan untuk melatih dan mengevaluasi model klasifikasi Random Forest.
-   **Pickle:** Digunakan untuk serialisasi (menyimpan) dan deserialisasi (memuat) dataset yang diproses serta model ML yang telah dilatih.

---

## Arsitektur Proyek dan Alur Kerja

Proyek ini dibagi menjadi beberapa modul skrip yang merepresentasikan tahapan dalam alur kerja *computer vision* dan ML:

1.  **`pengumpulan_data.py`**
    -   Menginisiasi *video capture* dari webcam.
    -   Menyimpan *frame* gambar sebagai data mentah ke dalam direktori `data` yang terstruktur berdasarkan kelas.

2.  **`buat_dataset.py`** (atau `train_data.py` bagian awal)
    -   Memuat data gambar mentah dari direktori `data`.
    -   Memproses setiap gambar menggunakan MediaPipe untuk mengekstrak *feature vector* (vektor fitur) dari *landmark* tangan.
    -   Menyimpan dataset yang telah diproses (fitur dan label) ke dalam file `.pickle` (contoh: `model.p`) untuk efisiensi pelatihan.

3.  **`train_data.py`**
    -   Memuat dataset `.pickle` yang telah diproses.
    -   Melakukan pembagian data (train-test split).
    -   Melatih (training) model klasifikasi Random Forest pada data latih.
    -   Melakukan evaluasi performa model pada data uji.
    -   Menyimpan model final yang telah terlatih (contoh: `best.pt` atau `model.p`).

4.  **`deteksi.py`** / **`inference.py`**
    -   Skrip aplikasi utama untuk demonstrasi.
    -   Memuat model terlatih.
    -   Menjalankan *pipeline* deteksi *real-time*: menangkap *frame*, mengekstrak *landmark*, dan melakukan inferensi dengan model.
    -   Menampilkan hasil prediksi klasifikasi (label isyarat) pada *video feed*.

---
