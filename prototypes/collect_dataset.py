
import cv2
import mediapipe as mp
import numpy as np
import os

# ------------------- Settings -------------------
SAVE_DIR = "custom_dataset"
CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(65,91)]  # 0-9 + A-Z
SAMPLES_PER_CLASS = 30

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def preprocess_canvas(canvas):
    # crop largest contour, resize to 28x28
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = canvas[y:y+h, x:x+w]
    if roi.size == 0: return None
    # resize keeping aspect ratio
    h,w = roi.shape
    if h>w: new_h,new_w=20,int(w*(20/h))
    else: new_w,new_h=20,int(h*(20/w))
    roi = cv2.resize(roi, (new_w,new_h), interpolation=cv2.INTER_AREA)
    canvas28 = np.zeros((28,28), np.uint8)
    xoff=(28-new_w)//2; yoff=(28-new_h)//2
    canvas28[yoff:yoff+new_h, xoff:xoff+new_w]=roi
    return canvas28

for cls in CLASSES:
    print(f"\nClass: {cls} — Draw {SAMPLES_PER_CLASS} samples")
    os.makedirs(os.path.join(SAVE_DIR,cls), exist_ok=True)
    count = 0
    canvas = np.zeros((480,640),np.uint8)
    prev = None

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            h,w,_=frame.shape
            tip_ids=[4,8,12,16,20]
            fingers=[1 if lm[tip].y<lm[tip-2].y else 0 for tip in tip_ids]

            if fingers[1]==1 and sum(fingers[2:])==0:  # index only -> draw
                ix,iy=int(lm[8].x*w),int(lm[8].y*h)
                if prev is None: prev=(ix,iy)
                cv2.line(canvas, prev, (ix,iy), 255,8)
                prev=(ix,iy)
            else:
                prev=None

        overlay = cv2.addWeighted(frame,0.5,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),0.5,0)
        cv2.putText(overlay,f"{cls} sample {count}/{SAMPLES_PER_CLASS}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(overlay,"SPACE=save  C=clear  N=next",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Collect", overlay)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # save
            roi = preprocess_canvas(canvas)
            if roi is not None:
                fname=os.path.join(SAVE_DIR,cls,f"{count}.png")
                cv2.imwrite(fname, roi)
                count+=1
                print(f"Saved {fname}")
            canvas[:]=0; prev=None
        elif k==ord('c'):  # clear
            canvas[:]=0; prev=None
        elif k==ord('n'):
            break
        elif k==ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit()

cap.release()
cv2.destroyAllWindows()
