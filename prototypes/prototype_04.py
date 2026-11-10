import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------
# Character Mapping for EMNIST ByClass (62 classes)
# ---------------------------------------------------------------
def get_emnist_labels():
    # 0–9, A–Z, a–z
    digits = [str(i) for i in range(10)]
    uppercase = [chr(i) for i in range(65, 91)]   # A–Z
    lowercase = [chr(i) for i in range(97, 123)]  # a–z
    return digits + uppercase + lowercase

emnist_labels = get_emnist_labels()  # length 62


# ---------------------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------------------
def preprocess_roi(canvas):
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = canvas[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    # Resize with aspect ratio preserved
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(w * (20.0 / h))
    else:
        new_w, new_h = 20, int(h * (20.0 / w))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on 28×28 canvas
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    # Optional stroke thickening
    kernel = np.ones((2, 2), np.uint8)
    canvas28 = cv2.dilate(canvas28, kernel, iterations=1)

    # Normalize, invert, reshape
    roi_norm = canvas28.astype('float32') / 255.0
    roi_norm = 1.0 - roi_norm
    roi_norm = np.expand_dims(roi_norm, axis=(0, -1))  # (1,28,28,1)
    return roi_norm


# ---------------------------------------------------------------
# Recognition Loop
# ---------------------------------------------------------------
def recognition_loop(model):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    canvas = None
    prev_point = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if canvas is None:
            canvas = np.zeros(img.shape[:2], dtype=np.uint8)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            h, w, _ = img.shape

            tip_ids = [4, 8, 12, 16, 20]
            fingers = [1 if lm[tip].y < lm[tip-2].y else 0 for tip in tip_ids]

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)

            # Index finger only → draw
            if fingers[1] == 1 and fingers[2] == 0 and sum(fingers[3:]) == 0:
                if prev_point is None:
                    prev_point = (ix, iy)
                cv2.line(canvas, prev_point, (ix, iy), 255, 8)
                prev_point = (ix, iy)

            # Index + middle finger → move cursor (no drawing)
            elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
                cv2.circle(img, (ix, iy), 10, (0, 255, 255), 2)
                prev_point = None

            # Open palm → clear canvas
            elif sum(fingers) == 5:
                canvas[:] = 0
                prev_point = None

            else:
                prev_point = None

        # Prediction
        if np.sum(canvas) > 1000:
            roi = preprocess_roi(canvas)
            if roi is not None:
                try:
                    preds = model.predict(roi)
                    top3_idx = preds[0].argsort()[-3:][::-1]
                    top3_conf = preds[0][top3_idx]

                    for i, (idx, conf) in enumerate(zip(top3_idx, top3_conf)):
                        char = emnist_labels[idx]
                        text = f"{char}: {conf*100:.1f}%"
                        cv2.putText(img, text, (10, 70 + i*40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0) if i == 0 else (255, 255, 0), 2)
                except Exception as e:
                    print("Prediction error:", e)

        overlay = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow("Character Recognition", overlay)
        cv2.imshow("Canvas", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    model = load_model("emnist_byclass_best.h5")
    recognition_loop(model)
