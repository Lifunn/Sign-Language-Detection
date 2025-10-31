import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Buka file pickle
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# PERBAIKAN DIMULAI DI SINI
# --------------------------------------------------------------------
# 1. Muat data sebagai Python list biasa terlebih dahulu
data = data_dict['data']
labels = data_dict['labels']

# 2. Saring data untuk memastikan semua fitur memiliki panjang 42
expected_feature_length = 42
data_filtered = []
labels_filtered = []

for i in range(len(data)):
    # Cek apakah panjang fitur untuk sampel ini sesuai harapan
    if len(data[i]) == expected_feature_length:
        data_filtered.append(data[i])
        labels_filtered.append(labels[i])

# 3. Sekarang konversi list yang sudah bersih ke NumPy array
data_np = np.asarray(data_filtered)
labels_np = np.asarray(labels_filtered)
# --------------------------------------------------------------------
# PERBAIKAN SELESAI

# Lanjutkan dengan data yang sudah bersih
x_train, x_test, y_train, y_test = train_test_split(
    data_np, 
    labels_np, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels_np
)

# Inisialisasi dan latih model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Prediksi dan evaluasi
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Simpan model yang sudah dilatih
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Model has been trained and saved as 'model.p'")