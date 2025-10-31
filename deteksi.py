import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- KONFIGURASI ---
model = YOLO('best.pt') 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

detection_history = deque(maxlen=20)
current_sentence = []
stable_threshold = 0.8 # Bisa disesuaikan

ui_box_height = 80
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (0, 0, 0)
window_name = "Deteksi Bahasa Isyarat v2.1 (Tekan 'q' atau Close)"

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame_height, frame_width, _ = frame.shape
    results = model(frame, stream=True, verbose=False, conf=0.5) # Ditambah conf threshold

    current_detection = None
    annotated_frame = frame.copy()

    for r in results:
        if len(r.boxes) > 0:
            annotated_frame = r.plot()
            highest_conf_box = max(r.boxes, key=lambda box: box.conf)
            class_id = int(highest_conf_box.cls[0])
            current_detection = model.names[class_id]

    detection_history.append(current_detection)

    if len(detection_history) == detection_history.maxlen:
        valid_detections = [d for d in detection_history if d is not None]

        if valid_detections:
            most_common_letter = max(set(valid_detections), key=valid_detections.count)
            
            stability_ratio = valid_detections.count(most_common_letter) / detection_history.maxlen

            if stability_ratio >= stable_threshold:
                if not current_sentence or current_sentence[-1] != most_common_letter:
                    current_sentence.append(most_common_letter)
                    detection_history.clear()
    
    ui_box = np.ones((ui_box_height, frame_width, 3), dtype=np.uint8) * 255
    display_text = "".join(current_sentence)
    text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = (ui_box_height + text_size[1]) // 2
    cv2.putText(ui_box, display_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    combined_frame = np.vstack((annotated_frame, ui_box))
    cv2.imshow(window_name, combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()