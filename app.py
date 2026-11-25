"""
Prototype 4.4 - Air-Drawn Digit Recognition (digits 0-9 only)
===============================================================

Draw a digit in the air with your index finger in front of the webcam and the
CNN (trained on MNIST) recognises it in real time.

Gestures
    Index finger up               -> draw
    Index + middle finger up      -> move the pointer without drawing
    Open palm (hold briefly)      -> clear the canvas
                                     (in Personalize mode: save the sample, then clear)

Modes (chosen from the start menu)
    R  Recognize     live top-3 predictions; press SPACE then the right digit
                     to save a correction when the model is wrong
    P  Personalize   guided collection: draw every digit 0-9 twice
    F  Fine-tune     retrain the model on your saved samples right now
    Q  Quit

Your samples are saved in user_feedback/digits/. Once 100 or more are saved,
the model is fine-tuned automatically. The result goes to mnist_cnn_user.h5,
and the original mnist_cnn.h5 is never changed. Delete mnist_cnn_user.h5 to
go back to the original model.

Changes from prototype 4.3_2
    * The image is no longer inverted. MNIST digits are white on black, the
      same as the canvas. Inverting it was the main cause of wrong predictions.
    * Saved samples are stored in exactly the format the model sees, so
      fine-tuning trains on the same kind of image it predicts on.
    * Fine-tuning always starts from the original model, saves to a separate
      file, mixes in real MNIST samples (when available) so the model does
      not forget, and is only kept if it does not get worse on held-out samples.
    * Digits drawn in several strokes (4, 5, 7 ...) are cropped as a whole,
      not just the largest stroke. Stroke thickness and centring now match
      MNIST better (centre of mass, like the original dataset).
    * One keyboard read per frame (SPACE was sometimes ignored before).
    * Corrections are typed in the camera window instead of the terminal, so
      the video no longer freezes.
    * Smoothed strokes (moving average), no jumping lines when the hand
      leaves and comes back, open palm must be held a moment to clear.
    * Unique file names for samples (no overwriting), paths work from any
      working directory (useful when running from VS Code).
"""

import os
import sys
import json
import time
import platform
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info spam

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ======================================================================
# CONFIG
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.h5")        # never modified
USER_MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_user.h5")   # personalised copy
USER_MODEL_META = os.path.join(BASE_DIR, "mnist_cnn_user.json")
FEEDBACK_DIR = os.path.join(BASE_DIR, "user_feedback", "digits")

LABELS = [str(i) for i in range(10)]
NUM_CLASSES = 10

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
WINDOW = "Air Digit Recognition - Prototype 4.4"

STROKE_THICKNESS = 10       # pen thickness on the 640x480 canvas
SMOOTHING = 0.45            # 0..1, lower = smoother but laggier pen
PALM_HOLD_FRAMES = 6        # frames the open palm must be held to clear / capture
MIN_INK_PIXELS = 150        # ignore tiny scribbles
PREDICT_EVERY = 3           # run the CNN every N frames while drawing

EXAMPLES_PER_DIGIT = 2      # guided personalisation: samples per digit
AUTO_FINETUNE_MIN = 100     # auto fine-tune once this many samples exist
AUTO_FINETUNE_STEP = 20     # ...and again after this many new samples

FT_EPOCHS = 5
FT_LR = 1e-4
FT_REPLAY_MNIST = 3000      # original MNIST images mixed in (0 to disable)
FT_AUGMENT_COPIES = 4       # augmented copies per user sample

os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Colours (BGR)
GREEN = (0, 220, 0)
YELLOW = (0, 220, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
RED = (60, 60, 255)
MAGENTA = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================================================================
# PREPROCESSING  (canvas -> 28x28 MNIST-style image)
# ======================================================================
def preprocess_canvas(canvas):
    """
    Turn the drawing canvas (white strokes on black) into the 28x28 image
    the MNIST model expects: white digit on black, about 20x20 px in size,
    centred by centre of mass.

    Returns (model_input, img28) or None if nothing is drawn.
        model_input : float32 array, shape (1, 28, 28, 1), values 0..1
        img28       : uint8 28x28 image (saved as the feedback sample)
    """
    if canvas.ndim == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    if cv2.countNonZero(canvas) < MIN_INK_PIXELS:
        return None

    # Box around ALL strokes, not just the largest one, so digits drawn in
    # more than one stroke are kept whole.
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(canvas))
    roi = canvas[y:y + h, x:x + w]

    # Thicken strokes in proportion to the digit size so that, after
    # shrinking to 20 px, they are as thick as MNIST strokes (2-3 px).
    side = max(w, h)
    k = int(side * 0.10) - STROKE_THICKNESS
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        roi = cv2.dilate(roi, kernel)
    pad = max(k, 0) // 2 + 1
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Scale the longer side to 20 px, keeping the aspect ratio
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img28 = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    img28[y0:y0 + new_h, x0:x0 + new_w] = small

    # Shift so the centre of mass is in the middle (how MNIST was made)
    m = cv2.moments(img28)
    if m["m00"] > 0:
        dx = 14 - m["m10"] / m["m00"]
        dy = 14 - m["m01"] / m["m00"]
        img28 = cv2.warpAffine(img28, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))

    model_input = (img28.astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return model_input, img28


# ======================================================================
# MODEL LOADING / PREDICTION
# ======================================================================
def load_digit_model():
    """Use the personalised model if there is one, otherwise the original."""
    path = USER_MODEL_PATH if os.path.exists(USER_MODEL_PATH) else BASE_MODEL_PATH
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model file not found: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    name = "personalised" if path == USER_MODEL_PATH else "original"
    print(f"[INFO] Loaded {name} model: {os.path.basename(path)}")
    return model, name


def predict_top3(model, model_input):
    probs = model(model_input, training=False).numpy()[0]
    top = probs.argsort()[-3:][::-1]
    return [(int(i), float(probs[i])) for i in top]


# ======================================================================
# FEEDBACK STORAGE
# ======================================================================
def save_feedback(img28, digit):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(FEEDBACK_DIR, f"N_{digit}_{stamp}.png")
    cv2.imwrite(path, img28)
    return path


def feedback_label(fname):
    """File names look like N_<digit>_<anything>.png -> digit, else None."""
    if not fname.lower().endswith(".png"):
        return None
    parts = fname.split("_")
    if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) > 9:
        return None
    return int(parts[1])


def load_feedback():
    """Load all saved samples from user_feedback/digits/."""
    X, y = [], []
    for fname in sorted(os.listdir(FEEDBACK_DIR)):
        label = feedback_label(fname)
        if label is None:
            continue
        img = cv2.imread(os.path.join(FEEDBACK_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        X.append(img)
        y.append(label)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def count_feedback():
    return sum(1 for f in os.listdir(FEEDBACK_DIR) if feedback_label(f) is not None)


def samples_used_by_user_model():
    if not (os.path.exists(USER_MODEL_PATH) and os.path.exists(USER_MODEL_META)):
        return 0
    try:
        with open(USER_MODEL_META) as f:
            return int(json.load(f).get("samples_used", 0))
    except (OSError, ValueError):
        return 0


# ======================================================================
# FINE-TUNING
# ======================================================================
def augment(img, rng):
    """Small random rotation / scale / shift, the way handwriting varies."""
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
    M[:, 2] += rng.uniform(-2, 2, size=2)
    return cv2.warpAffine(img, M, (28, 28))


def load_mnist_replay(n, rng):
    """Some original MNIST images, so fine-tuning does not forget the basics.
    Keras downloads MNIST (~11 MB) the first time; skipped if offline."""
    if n <= 0:
        return None, None
    try:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    except Exception as e:  # no internet, etc.
        print(f"[WARN] MNIST not available ({e.__class__.__name__}); "
              f"fine-tuning on your samples only.")
        return None, None
    idx = rng.choice(len(x_train), size=min(n, len(x_train)), replace=False)
    return x_train[idx], y_train[idx].astype(np.int64)


def finetune(reason=""):
    """
    Retrain the last layer of the ORIGINAL model on all saved samples
    (+ augmented copies + some MNIST) and save it as mnist_cnn_user.h5.
    """
    X_fb, y_fb = load_feedback()
    if len(X_fb) == 0:
        print("[INFO] No saved samples yet - nothing to fine-tune.")
        return False

    print(f"\n[INFO] Fine-tuning {reason}on {len(X_fb)} of your samples...")
    rng = np.random.default_rng(42)

    # Hold back some samples to check the new model is really better
    order = rng.permutation(len(X_fb))
    n_val = len(X_fb) // 5 if len(X_fb) >= 30 else 0
    val_idx, train_idx = order[:n_val], order[n_val:]

    X_tr, y_tr = [X_fb[train_idx]], [y_fb[train_idx]]
    for _ in range(FT_AUGMENT_COPIES):
        X_tr.append(np.stack([augment(img, rng) for img in X_fb[train_idx]]))
        y_tr.append(y_fb[train_idx])
    X_mn, y_mn = load_mnist_replay(FT_REPLAY_MNIST, rng)
    if X_mn is not None:
        X_tr.append(X_mn)
        y_tr.append(y_mn)
    X_tr = np.concatenate(X_tr).astype("float32")[..., None] / 255.0
    y_tr = np.concatenate(y_tr)

    model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    for layer in model.layers[:-1]:      # keep learned features,
        layer.trainable = False          # retrain only the output layer
    model.layers[-1].trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    def val_accuracy(m):
        if n_val == 0:
            return None
        xv = X_fb[val_idx].astype("float32")[..., None] / 255.0
        return float((m.predict(xv, verbose=0).argmax(1) == y_fb[val_idx]).mean())

    before = val_accuracy(model)
    model.fit(X_tr, y_tr, epochs=FT_EPOCHS, batch_size=32, shuffle=True, verbose=2)
    after = val_accuracy(model)

    if before is not None:
        print(f"[INFO] Accuracy on held-out samples: before {before:.0%} -> after {after:.0%}")
        if after < before:
            print("[WARN] Fine-tuned model was worse - keeping the previous model.")
            return False

    model.save(USER_MODEL_PATH)
    with open(USER_MODEL_META, "w") as f:
        json.dump({"samples_used": int(len(X_fb)),
                   "trained_at": datetime.now().isoformat(timespec="seconds"),
                   "val_before": before, "val_after": after}, f, indent=2)
    print(f"[INFO] Saved personalised model -> {os.path.basename(USER_MODEL_PATH)}\n")
    return True


def maybe_auto_finetune():
    total = count_feedback()
    used = samples_used_by_user_model()
    if total >= AUTO_FINETUNE_MIN and (used == 0 or total - used >= AUTO_FINETUNE_STEP):
        return finetune(reason=f"automatically ({total} samples saved) ")
    return False


# ======================================================================
# HAND TRACKING + VIRTUAL CANVAS (shared by both modes)
# ======================================================================
class AirCanvas:
    """Tracks one hand with MediaPipe and draws on a canvas."""

    TIP_IDS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1,
                                              min_detection_confidence=0.7,
                                              min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils
        self.canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self.smooth_pt = None    # smoothed fingertip position
        self.prev_pt = None      # last point drawn
        self.palm_frames = 0
        self.version = 0         # goes up whenever the canvas changes

    def clear(self):
        self.canvas[:] = 0
        self.prev_pt = None
        self.smooth_pt = None
        self.version += 1

    def has_ink(self):
        return cv2.countNonZero(self.canvas) >= MIN_INK_PIXELS

    def _fingers_up(self, lm):
        # A finger is up when its tip is above its middle joint (PIP).
        # The thumb is ignored: it is unreliable when it sticks out sideways.
        return [1 if lm[t].y < lm[t - 2].y else 0 for t in self.TIP_IDS]

    def process(self, frame):
        """
        Update the canvas from one (already mirrored) frame.
        Returns the gesture: 'none', 'draw', 'move', 'palm' (open palm held
        long enough) or 'other'.
        """
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            self.prev_pt = self.smooth_pt = None  # no jumping lines when the hand returns
            self.palm_frames = 0
            return "none"

        hand = results.multi_hand_landmarks[0]
        self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        lm = hand.landmark
        index, middle, ring, pinky = self._fingers_up(lm)
        tip = np.array([lm[8].x * FRAME_W, lm[8].y * FRAME_H])

        # Moving average of the fingertip to remove jitter
        if self.smooth_pt is None:
            self.smooth_pt = tip
        else:
            self.smooth_pt = SMOOTHING * tip + (1 - SMOOTHING) * self.smooth_pt
        pt = tuple(int(v) for v in self.smooth_pt)

        if index and not middle and not ring and not pinky:
            if self.prev_pt is not None:
                cv2.line(self.canvas, self.prev_pt, pt, 255, STROKE_THICKNESS, cv2.LINE_AA)
            else:
                cv2.circle(self.canvas, pt, STROKE_THICKNESS // 2, 255, -1)
            self.prev_pt = pt
            self.version += 1
            self.palm_frames = 0
            cv2.circle(frame, pt, 8, MAGENTA, -1)
            return "draw"

        self.prev_pt = None   # pen lifted

        if index and middle and not ring and not pinky:
            self.palm_frames = 0
            cv2.circle(frame, pt, 10, YELLOW, 2)
            return "move"

        if index and middle and ring and pinky:
            self.palm_frames += 1
            if self.palm_frames == PALM_HOLD_FRAMES:   # fires once per palm
                return "palm"
            return "other"

        self.palm_frames = 0
        return "other"

    def blend(self, frame):
        colour = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.6, colour, 0.8, 0)

    def close(self):
        self.hands.close()


# ======================================================================
# CAMERA / UI HELPERS
# ======================================================================
def open_camera():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam. Is another app using it?")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.flip(frame, 1)    # mirror, so it feels like a mirror
    if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    return frame


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def put(img, text, org, scale=0.6, colour=WHITE, thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def draw_help(img, text):
    cv2.rectangle(img, (0, FRAME_H - 30), (FRAME_W, FRAME_H), (0, 0, 0), -1)
    put(img, text, (10, FRAME_H - 10), 0.5, GREY)


def draw_preview(img, img28):
    """Show what the model actually sees (enlarged 28x28) in the top-right."""
    if img28 is None:
        return
    big = cv2.resize(img28, (84, 84), interpolation=cv2.INTER_NEAREST)
    x0, y0 = FRAME_W - 94, 10
    img[y0:y0 + 84, x0:x0 + 84] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + 84, y0 + 84), GREY, 1)
    put(img, "model input", (x0 - 6, y0 + 100), 0.4, GREY)


def draw_predictions(img, top3):
    if not top3:
        return
    digit, conf = top3[0]
    put(img, digit_text(digit), (15, 75), 2.4, GREEN, 4)
    put(img, f"{conf * 100:.0f}%", (80, 75), 0.8, GREEN, 2)
    for i, (d, c) in enumerate(top3):
        y = 105 + i * 26
        colour = GREEN if i == 0 else CYAN
        put(img, digit_text(d), (15, y), 0.6, colour, 2)
        cv2.rectangle(img, (40, y - 13), (40 + int(150 * c), y - 1), colour, -1)
        put(img, f"{c * 100:.1f}%", (198, y), 0.5, colour)


def digit_text(d):
    return LABELS[d]


# ======================================================================
# MODE R: RECOGNITION (+ live correction feedback)
# ======================================================================
def recognition_mode(model):
    cap = open_camera()
    if cap is None:
        return
    air = AirCanvas()
    top3, img28 = None, None
    predicted_version = -1
    correcting = False
    message, message_until = "", 0.0
    frame_no = 0
    saved = 0

    print("\n--- RECOGNITION MODE ---")
    print("Index = draw | Index+Middle = move | Open palm = clear")
    print("SPACE then 0-9 = save the correct digit when the guess is wrong | C = clear | Q = menu")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("[ERROR] Lost the webcam feed.")
            break
        frame_no += 1

        if not correcting:
            gesture = air.process(frame)
            if gesture == "palm":
                air.clear()

        # Predict only when the drawing has changed (keeps the video smooth)
        if air.version != predicted_version and frame_no % PREDICT_EVERY == 0:
            pre = preprocess_canvas(air.canvas)
            if pre is None:
                top3, img28 = None, None
            else:
                model_input, img28 = pre
                top3 = predict_top3(model, model_input)
            predicted_version = air.version

        view = air.blend(frame)
        put(view, "RECOGNIZE", (FRAME_W - 230, 30), 0.6, YELLOW, 2)
        draw_predictions(view, top3)
        draw_preview(view, img28)

        if correcting:
            cv2.rectangle(view, (0, FRAME_H // 2 - 35), (FRAME_W, FRAME_H // 2 + 25), (0, 0, 0), -1)
            put(view, "Press the CORRECT digit (0-9)   ESC = cancel",
                (20, FRAME_H // 2), 0.75, YELLOW, 2)
        if message and time.time() < message_until:
            put(view, message, (15, FRAME_H - 45), 0.6, GREEN, 2)
        draw_help(view, "SPACE: correct a wrong guess   C: clear   Q: back to menu"
                        f"   saved this session: {saved}")

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF   # read the keyboard ONCE per frame

        if window_closed():
            break
        if correcting:
            if ord("0") <= key <= ord("9"):
                if img28 is not None:
                    d = key - ord("0")
                    save_feedback(img28, d)
                    saved += 1
                    guess = digit_text(top3[0][0]) if top3 else "?"
                    message = f"Saved as '{d}' (model said '{guess}')"
                    message_until = time.time() + 2
                    print(f"[INFO] Correction saved: {d} (model said {guess})")
                air.clear()
                correcting = False
            elif key == 27:
                correcting = False
            continue

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            if img28 is not None:
                correcting = True
            else:
                message, message_until = "Draw a digit first", time.time() + 1.5
        elif key in (ord("c"), ord("C")):
            air.clear()

    air.close()
    cap.release()
    print(f"[INFO] {saved} correction(s) saved this session.")


# ======================================================================
# MODE P: GUIDED PERSONALISATION (draw each digit N times)
# ======================================================================
def personalization_mode():
    cap = open_camera()
    if cap is None:
        return
    air = AirCanvas()
    total_saved = 0

    print("\n--- PERSONALIZE MODE ---")
    print(f"Draw each digit {EXAMPLES_PER_DIGIT} times. Hold an OPEN PALM (or press SPACE) "
          f"to save. C = redo, N = skip digit, Q = stop.")

    quit_all = False
    for digit in range(NUM_CLASSES):
        collected = 0
        flash_until = 0.0
        while collected < EXAMPLES_PER_DIGIT:
            frame = read_frame(cap)
            if frame is None:
                quit_all = True
                break

            gesture = air.process(frame)

            view = air.blend(frame)
            pre = preprocess_canvas(air.canvas)
            draw_preview(view, pre[1] if pre else None)
            put(view, "PERSONALIZE", (FRAME_W - 230, 30), 0.6, YELLOW, 2)
            put(view, f"Draw:  {digit}", (15, 60), 1.6, YELLOW, 3)
            put(view, f"sample {collected + 1} of {EXAMPLES_PER_DIGIT}", (15, 95), 0.7, WHITE, 2)
            if time.time() < flash_until:
                put(view, "Saved!", (15, 135), 1.0, GREEN, 2)
            draw_help(view, "Open palm / SPACE: save   C: redo   N: skip digit   Q: stop")
            cv2.imshow(WINDOW, view)
            key_pressed = cv2.waitKey(1) & 0xFF

            if window_closed() or key_pressed in (ord("q"), ord("Q"), 27):
                quit_all = True
                break
            if key_pressed in (ord("n"), ord("N")):
                air.clear()
                break
            if key_pressed in (ord("c"), ord("C")):
                air.clear()
                continue

            if gesture == "palm" or key_pressed == ord(" "):
                if pre is not None:
                    save_feedback(pre[1], digit)
                    collected += 1
                    total_saved += 1
                    flash_until = time.time() + 0.8
                    print(f"[INFO] Saved digit {digit} ({collected}/{EXAMPLES_PER_DIGIT})")
                air.clear()
        if quit_all:
            break

    air.close()
    cap.release()
    print(f"[INFO] Personalisation finished: {total_saved} sample(s) saved.")


# ======================================================================
# START MENU
# ======================================================================
def show_menu(model_name):
    total = count_feedback()
    used = samples_used_by_user_model()
    if total < AUTO_FINETUNE_MIN:
        auto = f"auto fine-tune after {AUTO_FINETUNE_MIN - total} more sample(s)"
    else:
        left = max(0, AUTO_FINETUNE_STEP - (total - used)) if used else 0
        auto = f"next auto fine-tune after {left} more sample(s)" if left else "auto fine-tune: due"

    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    put(img, "Air-Drawn Digit Recognition", (30, 60), 1.0, YELLOW, 2)
    put(img, "Prototype 4.4  -  digits 0-9", (30, 95), 0.6, GREY)
    options = [("R", "Recognize (draw digits, live prediction)"),
               ("P", f"Personalize (draw each digit {EXAMPLES_PER_DIGIT}x)"),
               ("F", "Fine-tune now on my samples"),
               ("Q", "Quit")]
    for i, (k, text) in enumerate(options):
        y = 160 + i * 45
        put(img, f"[{k}]", (40, y), 0.8, GREEN, 2)
        put(img, text, (100, y), 0.7, WHITE, 1)
    put(img, f"Model in use: {model_name}", (30, 370), 0.55, CYAN)
    put(img, f"Saved samples: {total}   ({auto})", (30, 400), 0.55, CYAN)
    put(img, "Gestures: index = draw | index+middle = move | open palm = clear",
        (30, 445), 0.45, GREY)
    cv2.imshow(WINDOW, img)


def main():
    print("Prototype 4.4 - Air-Drawn Digit Recognition")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    maybe_auto_finetune()
    model, model_name = load_digit_model()

    while True:
        show_menu(model_name)
        key = cv2.waitKey(50) & 0xFF
        if window_closed():
            break
        if key in (ord("q"), ord("Q"), 27):
            break

        ran_mode = False
        if key in (ord("r"), ord("R")):
            recognition_mode(model)
            ran_mode = True
        elif key in (ord("p"), ord("P")):
            personalization_mode()
            ran_mode = True
        elif key in (ord("f"), ord("F")):
            if finetune(reason="(manual) "):
                model, model_name = load_digit_model()

        if ran_mode:
            if window_closed():   # user closed the window inside a mode
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            if maybe_auto_finetune():
                model, model_name = load_digit_model()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
