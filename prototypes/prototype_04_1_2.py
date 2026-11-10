import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Load your model ----
MODEL_PATH = "emnist_upper_digits.h5"   # <-- make sure the .h5 file is in same folder
model = load_model(MODEL_PATH)

# ---- Label Mapping ----
digits = [str(i) for i in range(10)]
uppercase = [chr(i) for i in range(65, 91)]   # A-Z
labels = digits + uppercase   # total 36 classes

# ---- Preprocessing Function ----
def preprocess_canvas(canvas):
    # convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    # Resize with aspect ratio preserved
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(w * (20.0 / h))
    else:
        new_w, new_h = 20, int(h * (20.0 / w))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Put on 28x28 canvas
    canvas28 = np.zeros((28,28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    # Thicken strokes
    kernel = np.ones((2,2), np.uint8)
    canvas28 = cv2.dilate(canvas28, kernel, iterations=1)

    # Normalize & invert
    norm = canvas28.astype('float32') / 255.0
    norm = 1.0 - norm
    norm = np.expand_dims(norm, axis=(0, -1))
    return norm

# ---- Mouse Drawing ----
drawing = False
last_point = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def draw(event, x, y, flags, param):
    global drawing, last_point, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, last_point, (x,y), (255,255,255), 10)
        last_point = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw")
cv2.setMouseCallback("Draw", draw)

print("INSTRUCTIONS:")
print(" - Draw with LEFT MOUSE button")
print(" - Press SPACE to predict")
print(" - Press C to clear")
print(" - Press Q to quit")

while True:
    cv2.imshow("Draw", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    elif key == 32:  # SPACE to predict
        roi = preprocess_canvas(canvas)
        if roi is not None:
            preds = model.predict(roi)
            top3 = preds[0].argsort()[-3:][::-1]
            print("\nTop predictions:")
            for idx in top3:
                print(f"{labels[idx]}: {preds[0][idx]*100:.2f}%")
        else:
            print("No drawing detected!")

cv2.destroyAllWindows()
