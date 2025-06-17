import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from gesture_tracker import hands, mp_draw, fingers_up, get_gesture_label, HAND_CONNECTIONS

# Load trained CNN model and label map
model = load_model("hovermath_cnn_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Initialize webcam
cap = cv2.VideoCapture(0)
drawing = False
strokes = []
current_stroke = []

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    output = frame.copy()

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(output, handLms, HAND_CONNECTIONS)

            finger_states = fingers_up(handLms)
            total_fingers = sum(finger_states)
            gesture = get_gesture_label(finger_states)

            cv2.putText(output, f"{gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            # Get index finger tip position
            h, w, _ = frame.shape
            index_finger = handLms.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            if gesture == "Writing":
                drawing = True
                current_stroke.append((cx, cy))
            elif gesture == "Stop Writing" and drawing:
                drawing = False
                if current_stroke:
                    strokes.append(current_stroke.copy())
                    current_stroke.clear()
            elif gesture == "Erase":
                if strokes:
                    strokes.pop()
            elif gesture == "Reset":
                strokes.clear()
                current_stroke.clear()
            elif gesture == "Solve" and strokes:
                print("Solving...")

                canvas = np.zeros((480, 640), dtype=np.uint8)
                for stroke in strokes:
                    for i in range(1, len(stroke)):
                        cv2.line(canvas, stroke[i - 1], stroke[i], 255, 6)

                _, thresh = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                symbol_images = []
                bboxes = []

                for cnt in contours:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    if w_box > 10 and h_box > 10:
                        roi = thresh[y:y + h_box, x:x + w_box]
                        roi_resized = cv2.resize(roi, (45, 45))
                        bboxes.append((x, roi_resized))

                bboxes.sort(key=lambda x: x[0])
                symbols = []

                for _, img in bboxes:
                    img = img / 255.0
                    img = img.reshape(1, 45, 45, 1)
                    pred = model.predict(img, verbose=0)
                    pred_class = int(np.argmax(pred))
                    label = inv_label_map[pred_class]
                    symbols.append(label)

                expression = "".join(symbols)
                print(f"Detected expression: {expression}")

                cv2.putText(output, f"Expr: {expression}", (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                strokes.clear()
                current_stroke.clear()

    # Draw strokes
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(output, stroke[i - 1], stroke[i], (255, 255, 255), 3)

    for i in range(1, len(current_stroke)):
        cv2.line(output, current_stroke[i - 1], current_stroke[i], (255, 255, 255), 3)

    cv2.imshow("HoverMath - Drawing", output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
