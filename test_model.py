import tensorflow as tf
import numpy as np
import json
import cv2
import os

# === Config ===
MODEL_PATH = 'best_model.h5'  # or 'hovermath_cnn_model.h5'
LABEL_MAP_PATH = 'label_map.json'
IMG_SIZE = 45

# === Load model and label map ===
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}

# === Preprocess image ===
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[ERROR] Cannot load image from '{img_path}'.")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 45, 45, 1)
    return img

# === Predict symbol ===
def predict_symbol(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    pred_idx = np.argmax(preds)
    pred_label = idx_to_label[pred_idx]
    confidence = preds[0][pred_idx]
    print(f"\n🔍 Prediction: {pred_label} (confidence: {confidence:.4f})\n")

# === Main Loop ===
if __name__ == "__main__":
    print("Enter path to an image (type 'exit' to quit):\n")
    while True:
        img_path = input("Image path: ").strip()
        if img_path.lower() == "exit":
            break
        if not os.path.exists(img_path):
            print("[]File does not exist. Try again.")
            continue
        try:
            predict_symbol(img_path)
        except Exception as e:
            print(f"[] Error: {e}")
