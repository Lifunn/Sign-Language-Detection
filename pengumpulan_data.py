import os
import cv2
import time

DATA_DIR = './data'
dataset_size = 100
CAMERA_INDEX = 0

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka kamera dengan indeks {CAMERA_INDEX}")
    exit()

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Bersiap mengumpulkan data untuk kelas: "{class_name}"')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera. Keluar...")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        prompt_text = f'Tunjukkan huruf "{class_name}". Siap? Tekan "Q"!'
        cv2.putText(frame, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    print("Pengambilan gambar akan dimulai dalam 3 detik...")
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret: continue
        countdown_text = f"Mulai dalam {i}..."
        cv2.putText(frame, countdown_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000) # Jeda 1 detik

    print(f'Sedang mengambil {dataset_size} gambar untuk kelas "{class_name}"...')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame saat pengambilan data.")
            break
        
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1
        cv2.waitKey(25) 

    print(f'Selesai mengumpulkan data untuk kelas "{class_name}".')

print("👍 Pengumpulan seluruh data selesai.")
cap.release()
cv2.destroyAllWindows()