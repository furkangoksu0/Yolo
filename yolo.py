from ultralytics import YOLO
import cv2


model = YOLO("HandSignDetector.pt")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, conf=0.5, show=False)


    annotated_frame = results[0].plot()


    cv2.imshow("Sign Language Detection", annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
