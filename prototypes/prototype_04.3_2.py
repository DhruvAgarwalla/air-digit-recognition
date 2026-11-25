import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from tensorflow.keras import layers, models

# ======================================
# CONFIG
# ======================================
FEEDBACK_DIR = "user_feedback"
DIGIT_FEEDBACK = os.path.join(FEEDBACK_DIR, "digits")
ALPHA_FEEDBACK = os.path.join(FEEDBACK_DIR, "alphabets")
os.makedirs(DIGIT_FEEDBACK, exist_ok=True)
os.makedirs(ALPHA_FEEDBACK, exist_ok=True)

DIGIT_MODEL_PATH = "../mnist_cnn.h5"
ALPHA_MODEL_PATH = "emnist_uppercase.h5"

# ======================================
# PREPROCESSING
# ======================================
def preprocess_canvas(canvas):
    """Convert drawn canvas to 28x28 image for prediction."""
    if len(canvas.shape) == 3:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas.copy()

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    # Resize keeping aspect ratio
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(w * (20.0 / h))
    else:
        new_w, new_h = 20, int(h * (20.0 / w))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    # Dilate slightly
    kernel = np.ones((2, 2), np.uint8)
    canvas28 = cv2.dilate(canvas28, kernel, iterations=1)

    roi_norm = canvas28.astype('float32') / 255.0
    roi_norm = 1.0 - roi_norm  # invert
    roi_norm = np.expand_dims(roi_norm, axis=(0, -1))
    return roi_norm, canvas28  # normalized + raw 28x28

# ======================================
# FINE-TUNE UTILITIES
# ======================================
def finetune_model(model_path, feedback_folder, num_classes):
    """
    Fine-tunes last dense layer of model with feedback images.
    """
    print(f"[INFO] Loading base model from {model_path}...")
    base_model = load_model(model_path)

    # Freeze all but last layer
    for layer in base_model.layers[:-1]:
        layer.trainable = False
    base_model.layers[-1].trainable = True

    base_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    # Load feedback images
    X, y = [], []
    for fname in os.listdir(feedback_folder):
        if fname.lower().endswith(".png"):
            img = cv2.imread(os.path.join(feedback_folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                label_char = fname.split("_")[1]
                if num_classes == 10:
                    y.append(int(label_char))
                else:
                    y.append(ord(label_char) - 65)
    if not X:
        print("[INFO] No feedback images found.")
        return base_model

    X = np.array(X).reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y = np.array(y)

    print(f"[INFO] Fine-tuning on {len(X)} samples...")
    base_model.fit(X, y, epochs=3, batch_size=16, verbose=2)

    base_model.save(model_path)
    print(f"[INFO] Model updated and saved back to {model_path}")
    return base_model

def maybe_auto_finetune(mode):
    folder = DIGIT_FEEDBACK if mode == 'N' else ALPHA_FEEDBACK
    model_path = DIGIT_MODEL_PATH if mode == 'N' else ALPHA_MODEL_PATH
    num_classes = 10 if mode == 'N' else 26
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if len(files) >= 100:
        print(f"\n[INFO] Found {len(files)} feedback samples. Auto fine-tuning now...")
        finetune_model(model_path, folder, num_classes)
        print("[INFO] Fine-tuning complete.\n")

# ======================================
# PERSONALIZATION (Guided sample collection)
# ======================================
def personalization_collection(mode, labels, examples_per_class=2):
    """
    Guides the user to draw each class twice and saves them into feedback folder.
    """
    print("\n--- PERSONALIZATION MODE ---")
    print("Draw each prompted character/digit twice using your finger.")
    print("Show OPEN PALM to capture and clear for next sample.")
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    canvas = None
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    target_folder = DIGIT_FEEDBACK if mode == 'N' else ALPHA_FEEDBACK

    for label in labels:
        collected = 0
        while collected < examples_per_class:
            ret, img = cap.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if canvas is None:
                canvas = np.zeros(img.shape[:2], dtype=np.uint8)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                tip_ids = [4, 8, 12, 16, 20]
                fingers = [1 if lm[tip].y < lm[tip-2].y else 0 for tip in tip_ids]
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                if fingers[1] == 1 and fingers[2] == 0 and sum(fingers[3:]) == 0:
                    cv2.circle(canvas, (ix, iy), 8, 255, -1)
                if sum(fingers) == 5:
                    pre = preprocess_canvas(canvas)
                    if pre:
                        _, raw = pre
                        save_path = os.path.join(target_folder, f"{mode}_{label}_{len(os.listdir(target_folder))}.png")
                        cv2.imwrite(save_path, raw)
                        collected += 1
                        print(f"Saved {label} sample {collected}/{examples_per_class}")
                    canvas[:] = 0
            cv2.putText(img, f"Draw '{label}' ({collected}/{examples_per_class})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            overlay = cv2.addWeighted(img,0.5,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),0.5,0)
            cv2.imshow("Personalization",overlay)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Personalization samples collected.")

# ======================================
# MAIN RECOGNITION LOOP
# ======================================
def recognition_loop(model, labels, mode):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    canvas = None
    prev_point = None
    print("\n--- RECOGNITION MODE ---\nDraw with index, open palm clears, SPACE for feedback, Q to quit.")

    while True:
        ret, img = cap.read()
        if not ret: break
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if canvas is None:
            canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            h, w, _ = img.shape
            tip_ids = [4,8,12,16,20]
            fingers = [1 if lm[tip].y<lm[tip-2].y else 0 for tip in tip_ids]
            ix, iy = int(lm[8].x*w), int(lm[8].y*h)
            if fingers[1]==1 and fingers[2]==0 and sum(fingers[3:])==0:
                if prev_point is None: prev_point=(ix,iy)
                cv2.line(canvas, prev_point, (ix,iy),255,8)
                prev_point=(ix,iy)
            elif fingers[1]==1 and fingers[2]==1 and sum(fingers[3:])==0:
                prev_point=None
            elif sum(fingers)==5:
                canvas[:]=0; prev_point=None
            else:
                prev_point=None

        # Prediction & feedback
        if np.sum(canvas)>1000:
            pre=preprocess_canvas(canvas)
            if pre:
                roi,raw=pre
                preds=model.predict(roi,verbose=0)
                top=preds[0].argsort()[-3:][::-1]
                for i,idx in enumerate(top):
                    cv2.putText(img,f"{labels[idx]}:{preds[0][idx]*100:.1f}%",(10,60+i*40),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if i==0 else (255,255,0),2)
                key=cv2.waitKey(1)&0xFF
                if key==32: # space
                    correct=input(f"Model guessed {labels[top[0]]}. Enter correct: ").strip().upper()
                    if correct:
                        folder=DIGIT_FEEDBACK if mode=='N' else ALPHA_FEEDBACK
                        cv2.imwrite(os.path.join(folder,f"{mode}_{correct}_{len(os.listdir(folder))}.png"),raw)
                        print("[INFO] Feedback saved.")

        overlay=cv2.addWeighted(img,0.5,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),0.5,0)
        cv2.imshow("Recognition",overlay)
        cv2.imshow("Canvas",canvas)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# ======================================
# MAIN ENTRY
# ======================================
if __name__=="__main__":
    # Step 1: choose Alphabet/Number
    menu=np.zeros((200,600,3),dtype=np.uint8)
    cv2.putText(menu,"Press N for Numbers or A for Alphabets",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("Menu",menu)
    mode=None
    while True:
        k=cv2.waitKey(1)&0xFF
        if k in [ord('n'),ord('N')]:mode='N';break
        elif k in [ord('a'),ord('A')]:mode='A';break
    cv2.destroyWindow("Menu")

    # Step 2: choose Recognition / Personalization
    menu2=np.zeros((200,600,3),dtype=np.uint8)
    cv2.putText(menu2,"Press R=Recognize or P=Personalize",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("Mode",menu2)
    mode2=None
    while True:
        k=cv2.waitKey(1)&0xFF
        if k in [ord('r'),ord('R')]:mode2='R';break
        elif k in [ord('p'),ord('P')]:mode2='P';break
    cv2.destroyWindow("Mode")

    labels=[str(i) for i in range(10)] if mode=='N' else [chr(i) for i in range(65,91)]
    model=load_model(DIGIT_MODEL_PATH if mode=='N' else ALPHA_MODEL_PATH)

    if mode2=='P':
        personalization_collection(mode,labels)
        maybe_auto_finetune(mode)
    else:
        maybe_auto_finetune(mode)
        recognition_loop(model,labels,mode)
