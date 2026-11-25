import cv2                                      #video capture
import mediapipe as mp                          #hand recognition
import numpy as np                              #array opperations on the dataset
import tensorflow as tf                         #model loading and training
from tensorflow.keras.models import load_model  
import time

# -------------------------------------------------------------------
# Helper: Capture a 28×28 ROI from a binary canvas (white-on-black)
# -------------------------------------------------------------------
def capture_digit_from_canvas(canvas):
    """
    Input:
      canvas: uint8 2D array (H, W), where drawn strokes are white (255) on black (0).
    Output:
      A 28×28 uint8 image (white digit on black) or None if no content.
    """
    # Find contours of nonzero (white) pixels
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour as the digit
    x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = canvas[y:y + h_box, x:x + w_box]
    if roi.size == 0:
        return None

    # Resize to 28×28
    roi_small = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize to [0,1], then invert so stroke is white on black
    roi_small = roi_small.astype('float32') / 255.0
    roi_small = 1.0 - roi_small  # stroke→1, background→0
    roi_small = (roi_small * 255).astype('uint8')  # convert back to 0–255 range

    return roi_small


# -------------------------------------------------------------------
# Helper: Freeze all but last layer(s) for fine-tuning
# -------------------------------------------------------------------
def get_finetune_model(original_model):
    """
    Takes a loaded Keras model, freezes all layers except the final Dense layer,
    recompiles with a low learning rate, and returns it ready for fine-tuning.
    """
    # Freeze all layers except last Dense
    for layer in original_model.layers[:-1]:
        layer.trainable = False
    original_model.layers[-1].trainable = True

    # Recompile with smaller learning rate
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
    """
    Collects a few hand-drawn examples of digits 0–9 (in the air), fine-tunes
    the last layer of the CNN on those examples, and returns the personalized model.
    """
    # 1) Load the base model and prepare it for fine-tuning
    base_model = load_model("../mnist_cnn.h5")
    ft_model = get_finetune_model(base_model)

    # 2) Parameters for few-shot collection
    examples_per_digit = 2   # Number of examples to collect for each digit 0–9
    total_digits = 10        # Digits 0 through 9

    # Prepare lists to hold captured images/labels
    X_pers = []
    y_pers = []

    # 3) Initialize webcam and MediaPipe for collection
    cap_p = cv2.VideoCapture(0)
    canvas_p = None
    mp_h = mp.solutions.hands
    hands_p = mp_h.Hands(max_num_hands=1,
                         min_detection_confidence=0.7,
                         min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Loop over each digit label
    for digit_label in range(total_digits):
        collected = 0
        while collected < examples_per_digit:
            ret, img_p = cap_p.read()
            if not ret:
                continue

            # Flip and prepare for MediaPipe
            img_p = cv2.flip(img_p, 1)
            img_rgb_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)

            # Initialize canvas on first frame
            if canvas_p is None:
                canvas_p = np.zeros(img_p.shape[:2], dtype=np.uint8)

            # Process hand landmarks
            results_p = hands_p.process(img_rgb_p)
            finger_state = [0, 0, 0, 0, 0]  # thumb, index, middle, ring, pinky

            if results_p.multi_hand_landmarks:
                hand_land = results_p.multi_hand_landmarks[0]
                h_p, w_p, _ = img_p.shape

                # Draw landmarks for feedback
                mp_draw.draw_landmarks(img_p, hand_land, mp_h.HAND_CONNECTIONS)

                # Determine which fingers are up
                tip_ids = [4, 8, 12, 16, 20]
                for i, tip in enumerate(tip_ids):
                    if hand_land.landmark[tip].y < hand_land.landmark[tip - 2].y:
                        finger_state[i] = 1

                # If only index finger is up, draw on canvas
                if (finger_state[1] == 1 and
                    finger_state[2] == 0 and
                    finger_state[3] == 0 and
                    finger_state[4] == 0):
                    ix_p = int(hand_land.landmark[8].x * w_p)
                    iy_p = int(hand_land.landmark[8].y * h_p)
                    cv2.circle(img_p, (ix_p, iy_p), 10, (255, 0, 255), -1)
                    cv2.circle(canvas_p, (ix_p, iy_p), 8, 255, -1)

                # If open palm (all 5 fingers up), capture example & clear canvas
                elif sum(finger_state) == 5:
                    captured = capture_digit_from_canvas(canvas_p)
                    if captured is not None:
                        X_pers.append(captured)
                        y_pers.append(digit_label)
                        collected += 1
                        print(f"Captured example {collected} for digit {digit_label}")
                    canvas_p[:] = 0  # Clear canvas for next example

            # Overlay canvas on camera feed for user feedback
            overlay_canvas = cv2.cvtColor(canvas_p, cv2.COLOR_GRAY2BGR)
            display_frame = cv2.addWeighted(img_p, 0.5, overlay_canvas, 0.5, 0)

            # Show on-screen instructions
            cv2.putText(display_frame,
                        f"Draw digit '{digit_label}' ({collected}/{examples_per_digit})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Personalize Model - Draw Now", display_frame)

            # Break if user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Small pause before moving to next digit
        time.sleep(0.5)

    # Cleanup collection windows
    cap_p.release()
    cv2.destroyAllWindows()

    # 4) Prepare data for fine-tuning
    X_pers = np.array(X_pers)  # shape = (#examples, 28, 28)
    X_pers = X_pers.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_pers = np.array(y_pers)  # shape = (#examples, )

    print(f"Total examples collected: {X_pers.shape[0]}")

    # 5) Fine-tune the last layer(s) on those new samples
    ft_model.fit(
        X_pers, y_pers,
        epochs=2,
        batch_size=max(2, len(y_pers) // 2),
        verbose=2
    )

    # 6) Save the personalized model
    ft_model.save("mnist_cnn_personalized.h5")
    print("Personalized model saved as 'mnist_cnn_personalized.h5'")

    # Return the newly fine-tuned model
    return load_model("mnist_cnn_personalized.h5")


# -------------------------------------------------------------------
# Recognition Loop: Draw digits in air and predict
# -------------------------------------------------------------------
def recognition_loop(model):
    """
    Runs the real-time digit-recognition loop using the given `model`.
    - Only index finger draws on a virtual canvas.
    - Open palm clears the canvas.
    - When canvas contains strokes, extract ROI → preprocess → predict → overlay result.
    """
    # Prepare MediaPipe for hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Start webcam and initialize canvas
    cap = cv2.VideoCapture(0)
    canvas = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if canvas is None:
            canvas = np.zeros(img.shape[:2], dtype=np.uint8)

        # Detect hand landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            h, w, _ = img.shape

            # Determine which fingers are up
            tip_ids = [4, 8, 12, 16, 20]
            fingers = []
            for i, tip in enumerate(tip_ids):
                if lm[tip].y < lm[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # If only index finger is up, draw on canvas
            if (fingers[1] == 1 and
                fingers[2] == 0 and
                fingers[3] == 0 and
                fingers[4] == 0):
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                cv2.circle(canvas, (ix, iy), 8, 255, -1)

            # If open palm (all five fingers up), clear canvas
            if sum(fingers) == 5:
                canvas[:] = 0

        # If enough pixels on canvas, extract ROI and predict
        if np.sum(canvas) > 1000:
            # Find the bounding box of the largest contour
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))
                digit_roi = canvas[y:y + h_box, x:x + w_box]

                try:
                    # Preprocess ROI: resize → normalize → invert → reshape
                    roi = cv2.resize(digit_roi, (28, 28))
                    roi = roi.astype('float32') / 255.0
                    roi = 1.0 - roi
                    roi = np.expand_dims(roi, axis=(0, -1))  # shape = (1,28,28,1)

                    # Predict digit
                    pred = model.predict(roi)
                    digit = np.argmax(pred)

                    # Overlay prediction text
                    cv2.putText(img, f'Predicted: {digit}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                except Exception as e:
                    print("Prediction error:", e)

        # Blend canvas onto webcam feed
        overlay = cv2.addWeighted(img, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow("Draw with Finger - Recognition", overlay)

        # Show the raw canvas in a separate window
        cv2.imshow("Canvas", canvas)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------
# Main: Startup Menu and Mode Dispatch
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Display a simple menu asking user to press P or R
    menu_win = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(menu_win,
                "Press 'P' to Personalize or 'R' to Recognize",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imshow("Startup Menu", menu_win)
    mode = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p') or key == ord('P'):
            mode = 'P'
            break
        elif key == ord('r') or key == ord('R'):
            mode = 'R'
            break

    cv2.destroyWindow("Startup Menu")

    # Load or personalize as chosen
    if mode == 'P':
        model = personalize_model()
    else:
        model = load_model("../mnist_cnn.h5")

    # Run the recognition loop using the chosen model
    recognition_loop(model)
