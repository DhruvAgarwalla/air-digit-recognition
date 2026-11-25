import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# -------------------------------------------------------------------
# Preprocessing Function
# -------------------------------------------------------------------
def preprocess_roi(canvas):
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = canvas[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(w * (20.0 / h))
    else:
        new_w, new_h = 20, int(h * (20.0 / w))

    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    kernel = np.ones((2, 2), np.uint8)
    canvas28 = cv2.dilate(canvas28, kernel, iterations=1)

    roi_norm = canvas28.astype('float32') / 255.0
    roi_norm = 1.0 - roi_norm
    roi_norm = np.expand_dims(roi_norm, axis=(0, -1))
    return roi_norm

# -------------------------------------------------------------------
# Personalization Module
# -------------------------------------------------------------------
def personalize_model(base_model):
    print("Entering Personalization Mode...")
    X, y = [], []
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    canvas = None
    prev_point = None

    for digit in range(10):
        print(f"Draw digit {digit}, press 's' to save, 'n' for next")
        canvas = np.zeros((480, 640), dtype=np.uint8)
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

                if fingers[1] == 1 and sum(fingers[2:]) == 0:
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                    if prev_point is None:
                        prev_point = (ix, iy)
                    cv2.line(canvas, prev_point, (ix, iy), 255, 8)
                    prev_point = (ix, iy)
                else:
                    prev_point = None

            overlay = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.putText(overlay, f"Draw digit {digit}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(overlay, "Press 's' to save, 'n' for next", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Personalization", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                roi = preprocess_roi(canvas)
                if roi is not None:
                    X.append(roi)
                    y.append(digit)
                    print(f"Saved digit {digit}")
                canvas[:] = 0
                prev_point = None
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return base_model

    cap.release()
    cv2.destroyAllWindows()

    if len(X) > 0:
        X = np.array(X).reshape(-1, 28, 28, 1)
        y = tf.keras.utils.to_categorical(y, 10)
        print(f"Collected {len(X)} samples. Fine-tuning model...")
        base_model.fit(X, y, epochs=5, verbose=1, validation_split=0.2)
    return base_model

# -------------------------------------------------------------------
# Recognition Loop
# -------------------------------------------------------------------
def recognition_loop(model):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
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

            if fingers[1] == 1 and sum(fingers[2:]) == 0:
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                if prev_point is None:
                    prev_point = (ix, iy)
                cv2.line(canvas, prev_point, (ix, iy), 255, 8)
                prev_point = (ix, iy)
            else:
                prev_point = None

            if sum(fingers) == 5:
                canvas[:] = 0

        if np.sum(canvas) > 1000:
            roi = preprocess_roi(canvas)
            if roi is not None:
                try:
                    preds = model.predict(roi)
                    top3_idx = preds[0].argsort()[-3:][::-1]
                    top3_conf = preds[0][top3_idx]
                    for i, (digit, conf) in enumerate(zip(top3_idx, top3_conf)):
                        text = f"{digit}: {conf*100:.1f}%"
                        cv2.putText(img, text, (10, 70 + i*40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0) if i == 0 else (255, 255, 0), 2)
                except Exception as e:
                    print("Prediction error:", e)

        overlay = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow("Digit Recognition", overlay)
        cv2.imshow("Canvas", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    menu_win = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(menu_win, "Press 'P' to Personalize or 'R' to Recognize",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Startup Menu", menu_win)

    mode = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('p'), ord('P')]:
            mode = 'P'
            break
        elif key in [ord('r'), ord('R')]:
            mode = 'R'
            break

    cv2.destroyWindow("Startup Menu")

    if mode == 'P':
        base_model = load_model("../mnist_cnn.h5")
        model = personalize_model(base_model)
    else:
        model = load_model("../mnist_cnn.h5")

    recognition_loop(model)
