"""
Model ka test - pata lagao masla kahan hai
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("mask_detector.h5")

# Dataset ki TEST tasveeron pe check karo (inke jawab humein pata hain!)
base = r"C:\Users\NeXus\Downloads\Face Mask Dataset\Test"

for true_label in ["WithMask", "WithoutMask"]:
    folder = os.path.join(base, true_label)
    files = os.listdir(folder)[:5]  # pehli 5 tasveeren
    print(f"\n--- {true_label} (asal jawab) ---")
    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)[0][0]
        predicted = "WithoutMask" if pred > 0.5 else "WithMask"
        result = "SAHI ✅" if predicted == true_label else "GHALAT ❌"
        print(f"  {f}: model bola {predicted} ({pred:.2f}) {result}")

