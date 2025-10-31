import os
import random
import cv2
import mediapipe as mp

# --- Inisialisasi MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# --- Path ke direktori data ---
DATA_DIR = './data'

# --- Loop untuk menampilkan gambar secara acak ---
while True:
    # 1. Pilih folder kelas secara acak
    class_dir_name = random.choice(os.listdir(DATA_DIR))
    class_path = os.path.join(DATA_DIR, class_dir_name)

    # Pastikan itu adalah direktori
    if not os.path.isdir(class_path):
        continue

    # 2. Pilih gambar secara acak dari folder tersebut
    img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)

    # 3. Baca dan proses gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"Gagal membaca gambar: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # 4. Gambar landmark jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,  # Gambar yang akan digambar (BGR)
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # 5. Tuliskan label di gambar
    label_text = f'Label: {class_dir_name}'
    cv2.putText(img, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # 6. Tampilkan gambar
    cv2.imshow('Gambar dengan Label dan Landmark', img)

    # --- Kontrol ---
    print(f"Menampilkan gambar dari kelas '{class_dir_name}'. Tekan 'Spasi' untuk gambar lain, atau 'Q' untuk keluar.")
    key = cv2.waitKey(0) # Tunggu tombol ditekan

    if key == ord('q'): # Jika 'q' ditekan, keluar
        break
    elif key == 32: # Jika Spasi ditekan, lanjut ke gambar berikutnya
        continue

# --- Cleanup ---
cv2.destroyAllWindows()
hands.close()