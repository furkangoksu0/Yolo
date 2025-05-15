from ultralytics import YOLO
import cv2
import time
from collections import deque

model = YOLO("HandSignDetector.pt")
cap = cv2.VideoCapture(0)

detected_letters = []
current_letter = ""
letter_locked = False
last_seen_letter = ""
letter_buffer = deque(maxlen=10)  # Son 10 harf örneğini tut

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, show=False)
    annotated_frame = results[0].plot()

    if results and results[0].boxes:
        boxes = results[0].boxes
        class_id = int(boxes.cls[0].item())
        letter = model.names[class_id]
        letter_buffer.append(letter)

        # Eğer son 10 karede aynı harf çoğunluktaysa, o harfi sabitle
        most_common_letter = max(set(letter_buffer), key=letter_buffer.count)
        if letter_buffer.count(most_common_letter) > 6:  # e.g. 7/10 kere aynıysa
            if not letter_locked or most_common_letter != current_letter:
                current_letter = most_common_letter
                letter_locked = True

    else:
        # El kaybolmuşsa ve harf kilitlenmişse, onu yazıya ekle
        if letter_locked and current_letter != "":
            detected_letters.append(current_letter)
            print(f"Sabitlenen harf: {current_letter}")
            current_letter = ""
            letter_locked = False
            letter_buffer.clear()
            time.sleep(0.5)

    # Ekran çizimleri
    cv2.rectangle(annotated_frame, (10, 10), (630, 60), (0, 0, 0), -1)
    text = "Detected: " + "".join(detected_letters)
    cv2.putText(annotated_frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    if current_letter:
        cv2.putText(annotated_frame, f"Current: {current_letter}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Sign Language Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        detected_letters = []
        current_letter = ""
        letter_locked = False
        letter_buffer.clear()

cap.release()
cv2.destroyAllWindows()
