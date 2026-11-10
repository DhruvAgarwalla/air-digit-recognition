import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from prototype_04_1_2 import preprocess_canvas   # make sure this function exists in your file

# ---------------------------
# Load your trained model
# ---------------------------
model = load_model("emnist_upper_digits.h5")

# ---------------------------
# Create a drawing canvas
# ---------------------------
canvas_size = 280
canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)  # 3-channel BGR canvas
drawing = False
prev_point = None

print("INSTRUCTIONS:")
print(" - Draw with LEFT MOUSE button")
print(" - Press SPACE to predict")
print(" - Press C to clear")
print(" - Press Q to quit")

# ---------------------------
# Mouse drawing callback
# ---------------------------
def draw(event, x, y, flags, param):
    global drawing, prev_point, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if prev_point is not None:
            cv2.line(canvas, prev_point, (x, y), (255, 255, 255), 8)  # draw white
        prev_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_point = None

cv2.namedWindow("Draw")
cv2.setMouseCallback("Draw", draw)

while True:
    cv2.imshow("Draw", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE to predict
        roi = preprocess_canvas(canvas)
        if roi is not None:
            preds = model.predict(roi)
            top3 = preds[0].argsort()[-3:][::-1]
            print("\nTop predictions:")
            for idx in top3:
                print(f"{idx}: {preds[0][idx]*100:.2f}%")
        else:
            print("No drawing detected!")

    elif key == ord('c'):  # clear canvas
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    elif key == ord('q'):  # quit
        break

cv2.destroyAllWindows()
