import cv2                                      # video capture
import mediapipe as mp                          # hand recognition
import numpy as np                              # array operations on the dataset
import tensorflow as tf                         # model loading and training
from tensorflow.keras.models import load_model  
import time
print("Script started...")

# -------------------------------------------------------------------
# Helper: Capture a 28×28 ROI from a binary canvas (white-on-black)
# -------------------------------------------------------------------
def capture_digit_from_canvas(canvas):
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = canvas[y:y + h_box, x:x + w_box]
    if roi.size == 0:
        return None

    roi_small = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi_small = roi_small.astype('float32') / 255.0
    roi_small = 1.0 - roi_small
    roi_small = (roi_small * 255).astype('uint8')
    return roi_small


# -------------------------------------------------------------------
# Helper: Freeze all but last layer(s) for fine-tuning
# -------------------------------------------------------------------
def get_finetune_model(original_model):
    for layer in original_model.layers[:-1]:
        layer.trainable = False
    original_model.layers[-1].trainable = True

    original_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return original_model


# -------------------------------------------------------------------
# Personalization Routine
# -------------------------------------------------------------------
def personalize_model():
    base_model = load_model("mnist_cnn.h5")
    ft_model = get_finetune_model(base_model)

    examples_per_digit = 2
    total_digits = 10

    X_pers, y_pers = [], []
    cap_p = cv2.VideoCapture(0)
    canvas_p = None
    mp_h = mp.solutions.hands
    hands_p = mp_h.Hands(max_num_hands=1,
                         min_detection_confidence=0.7,
                         min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    for digit_label in range(total_digits):
        collected = 0
        while collected < examples_per_digit:
            ret, img_p = cap_p.read()
            if not ret:
                continue

            img_p = cv2.flip(img_p, 1)
            img_rgb_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)

            if canvas_p is None:
                canvas_p = np.zeros(img_p.shape[:2], dtype=np.uint8)

            results_p = hands_p.process(img_rgb_p)
            finger_state = [0, 0, 0, 0, 0]

            if results_p.multi_hand_landmarks:
                hand_land = results_p.multi_hand_landmarks[0]
                h_p, w_p, _ = img_p.shape
                mp_draw.draw_landmarks(img_p, hand_land, mp_h.HAND_CONNECTIONS)

                tip_ids = [4, 8, 12, 16, 20]
                for i, tip in enumerate(tip_ids):
                    if hand_land.landmark[tip].y < hand_land.landmark[tip - 2].y:
                        finger_state[i] = 1

                if (finger_state[1] == 1 and
                    finger_state[2] == 0 and
                    finger_state[3] == 0 and
                    finger_state[4] == 0):
                    ix_p = int(hand_land.landmark[8].x * w_p)
                    iy_p = int(hand_land.landmark[8].y * h_p)
                    cv2.circle(img_p, (ix_p, iy_p), 10, (255, 0, 255), -1)
                    cv2.circle(canvas_p, (ix_p, iy_p), 8, 255, -1)

                elif sum(finger_state) == 5:
                    captured = capture_digit_from_canvas(canvas_p)
                    if captured is not None:
                        X_pers.append(captured)
                        y_pers.append(digit_label)
                        collected += 1
                        print(f"Captured example {collected} for digit {digit_label}")
                    canvas_p[:] = 0

            overlay_canvas = cv2.cvtColor(canvas_p, cv2.COLOR_GRAY2BGR)
            display_frame = cv2.addWeighted(img_p, 0.5, overlay_canvas, 0.5, 0)

            cv2.putText(display_frame,
                        f"Draw digit '{digit_label}' ({collected}/{examples_per_digit})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Personalize Model - Draw Now", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.5)

    cap_p.release()
    cv2.destroyAllWindows()

    X_pers = np.array(X_pers)
    X_pers = X_pers.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_pers = np.array(y_pers)

    print(f"Total examples collected: {X_pers.shape[0]}")

    ft_model.fit(
        X_pers, y_pers,
        epochs=2,
        batch_size=max(2, len(y_pers) // 2),
        verbose=2
    )

    ft_model.save("mnist_cnn_personalized.h5")
    print("Personalized model saved as 'mnist_cnn_personalized.h5'")

    return load_model("mnist_cnn_personalized.h5")


# -------------------------------------------------------------------
# Recognition Loop with smoothing & line drawing
# -------------------------------------------------------------------
def recognition_loop(model):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    canvas = None
    points_history = []  # for smoothing

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
            fingers = []
            for i, tip in enumerate(tip_ids):
                if lm[tip].y < lm[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if (fingers[1] == 1 and
                fingers[2] == 0 and
                fingers[3] == 0 and
                fingers[4] == 0):
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                points_history.append((ix, iy))
                if len(points_history) > 5:
                    points_history.pop(0)

                avg_x = int(np.mean([p[0] for p in points_history]))
                avg_y = int(np.mean([p[1] for p in points_history]))

                if len(points_history) >= 2:
                    cv2.line(canvas, points_history[-2], (avg_x, avg_y), 255, 12)
                else:
                    cv2.circle(canvas, (avg_x, avg_y), 8, 255, -1)

            if sum(fingers) == 5:
                canvas[:] = 0
                points_history.clear()

        if np.sum(canvas) > 1000:
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))
                digit_roi = canvas[y:y + h_box, x:x + w_box]

                try:
                    roi = cv2.resize(digit_roi, (28, 28))
                    roi = roi.astype('float32') / 255.0
                    roi = 1.0 - roi
                    roi = np.expand_dims(roi, axis=(0, -1))

                    pred = model.predict(roi)
                    digit = np.argmax(pred)

                    cv2.putText(img, f'Predicted: {digit}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                except Exception as e:
                    print("Prediction error:", e)

        overlay = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow("Draw with Finger - Recognition", overlay)
        cv2.imshow("Canvas", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Create menu window
    menu_win = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(menu_win,
                "Press 'P' to Personalize or 'R' to Recognize",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    mode = None
    while True:
        cv2.imshow("Startup Menu", menu_win)
        key = cv2.waitKey(10) & 0xFF  # small delay is safer on Windows
        if key in [ord('p'), ord('P')]:
            mode = 'P'
            break
        elif key in [ord('r'), ord('R')]:
            mode = 'R'
            break
        elif key == 27:  # ESC to quit
            mode = None
            break

    cv2.destroyAllWindows()

    if mode == 'P':
        model = personalize_model()
        recognition_loop(model)
    elif mode == 'R':
        model = load_model("mnist_cnn.h5")
        recognition_loop(model)
