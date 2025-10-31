import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# --- Muat Model ---
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# --- Buka Kamera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

sentence = []
last_save_time = 0
predicted_character = "" 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame, keluar...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        data_aux = []
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        min_x = min(x_coords)
        min_y = min(y_coords)
        
        for x, y in zip(x_coords, y_coords):
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)
        
        if len(data_aux) == 42:
            x1 = int(min(x_coords) * W) - 20
            y1 = int(min(y_coords) * H) - 20
            x2 = int(max(x_coords) * W) + 20
            y2 = int(max(y_coords) * H) + 20
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # --- Menampilkan Kalimat di Layar ---
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, ' '.join(sentence), (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # Keluar dari program
    if key == ord('q'):
        break

    # Simpan huruf (Spasi)
    # Ada jeda 1 detik untuk menghindari input ganda
    if key == 32 and (current_time - last_save_time > 1): # 32 adalah kode ASCII untuk Spasi
        if predicted_character: # Hanya menyimpan jika ada prediksi
            sentence.append(predicted_character)
            last_save_time = current_time
            print(f"Huruf '{predicted_character}' ditambahkan. Kalimat sekarang: {' '.join(sentence)}")

    # Hapus huruf terakhir (Backspace)
    if key == 8:
        if sentence:
            removed_char = sentence.pop()
            print(f"Huruf '{removed_char}' dihapus. Kalimat sekarang: {' '.join(sentence)}")
            # Beri jeda singkat agar tidak menghapus terlalu cepat
            time.sleep(0.2)


cap.release()
cv2.destroyAllWindows()

# if sentence:
#     final_text = ' '.join(sentence)
#     with open('kalimat_tersimpan.txt', 'w') as f:
#         f.write(final_text)
#     print(f"\nKalimat akhir telah disimpan di 'kalimat_tersimpan.txt'")