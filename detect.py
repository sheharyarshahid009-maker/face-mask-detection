"""
Face Mask Detection - LIVE Webcam (DNN Face Detector)
Kala mask bhi detect karega!
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Models load ho rahe hain...")

# Mask classifier (Colab pe train hua)
model = load_model("mask_detector.h5")

# Face detector (DNN - mask wale chehre bhi pakadta hai!)
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)
print("Ready!")

cap = cv2.VideoCapture(0)
print("Webcam ON! 'q' dabao band karne ke liye!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # DNN se chehre dhoondo
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Box image ke andar rakho
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0][0]

            if pred > 0.5:
                label = "No Mask"
                color = (0, 0, 255)
                conf = pred
            else:
                label = "Mask"
                color = (0, 255, 0)
                conf = 1 - pred

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.0f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection ('q' = band)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Ho gaya!")
