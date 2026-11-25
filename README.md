# Air-Digit Recognition Using Hand Gestures

Draw digits **in the air with your finger** in front of a webcam and have them recognised in real time.

The project uses **MediaPipe** to track your hand, **OpenCV** to turn the fingertip path into an image, and a **CNN trained on MNIST** (TensorFlow/Keras) to recognise the digit. It also **learns your handwriting**: you can correct wrong guesses or draw guided samples, and the model fine-tunes itself on them.

## Demo

[![Watch the Prototype 03 demo on YouTube](https://img.youtube.com/vi/7kM_RIgZH9s/hqdefault.jpg)](https://youtu.be/7kM_RIgZH9s)

*Prototype 03: personalising the model by drawing digits in the air (click to watch on YouTube).*

The project's development is also documented in a three-part video series on LinkedIn:
[Part 1: training the CNN](https://www.linkedin.com/feed/update/urn:li:activity:7347905876533350402/) ·
[Part 2: virtual drawing canvas](https://www.linkedin.com/feed/update/urn:li:activity:7349384466055725056/) ·
[Part 3: drawing digits in the air](https://www.linkedin.com/feed/update/urn:li:activity:7350834021146292224/)

---

## Features

- **Webcam only**: no special sensors, depth cameras or gloves
- **Gesture controls**: draw, move the pointer and clear the canvas using only your hand
- **Live top-3 predictions** with confidence bars
- **"Model input" preview**: shows the exact 28×28 image the CNN sees
- **Correction feedback**: when a guess is wrong, press SPACE and type the right digit; the sample is saved
- **Guided personalisation**: draw each digit 0–9 twice to teach the model your style
- **Automatic fine-tuning**: once 100 samples are saved, the model's last layer is retrained on them, with MNIST samples mixed in so it doesn't forget the basics
- **Safe personalisation**: the original model is never overwritten; the personalised model is saved separately and only kept if it does not do worse on held-back samples

## Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Index + middle finger up | Move the pointer without drawing |
| 🖐️ Open palm (hold briefly) | Clear the canvas (in Personalize mode: save the sample) |

| Key | Where | Action |
|---|---|---|
| `R` | Menu | Recognize mode |
| `P` | Menu | Personalize mode |
| `F` | Menu | Fine-tune now on your saved samples |
| `Q` / `Esc` | Anywhere | Back to menu / quit |
| `SPACE` then `0`–`9` | Recognize | Save a correction when the guess is wrong |
| `C` | Recognize / Personalize | Clear the canvas |
| `N` | Personalize | Skip to the next digit |

> Keys only work while the app's window is selected (click on it first).

---

## Getting started

### Requirements
- **Python 3.9 – 3.12** (TensorFlow 2.16 and MediaPipe 0.10.21 do not support 3.13+)
- A webcam

### Installation

```bash
git clone https://github.com/DhruvAgarwalla/air-digit-recognition.git
cd air-digit-recognition

# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The first launch takes 10–20 seconds while TensorFlow loads. A menu window then opens. Press **R** to start drawing.

---

## How it works

```
Webcam frame
   │
   ▼
MediaPipe Hands ──► 21 hand landmarks ──► which fingers are up? ──► gesture
   │
   ▼
Index fingertip (smoothed) drawn as strokes on a black canvas
   │
   ▼
Preprocessing (OpenCV)
   • crop around all strokes
   • thicken strokes to match MNIST
   • resize to fit 20×20, place in 28×28
   • centre by centre of mass (like MNIST)
   │
   ▼
CNN (MNIST) ──► top-3 digits + confidence on screen
```

**Model:** 2 convolution layers (32 and 64 filters) with max-pooling, a 128-unit dense layer and a 10-class softmax, trained on MNIST.

**Personalisation:** samples are saved as 28×28 PNGs in `user_feedback/digits/`. Fine-tuning freezes the convolution layers and retrains only the output layer on your samples (plus augmented copies and MNIST samples). The result is saved as `mnist_cnn_user.h5`. **Delete that file to go back to the original model.**

---

## Project structure

```
├── app.py                        # main application (prototype 4.4)
├── mnist_cnn.h5                  # digit model (MNIST CNN)
├── requirements.txt
├── README.md
├── LICENSE
└── prototypes/                   # earlier versions, kept to show the project's evolution
    ├── prototype_03.py … prototype_04.3_2.py
    ├── collect_dataset.py        # tool to record a custom air-drawn dataset
    ├── debug_preprocess.py       # mouse-drawing test for the 36-class model
    ├── emnist_byclass_best.h5    # 62-class model (0-9, A-Z, a-z)
    ├── emnist_uppercase.h5       # 26-class model (A-Z)
    └── emnist_upper_digits.h5    # 36-class model (0-9 + A-Z)
```

Files created when you use the app (not part of the repo): `user_feedback/`, `mnist_cnn_user.h5`, `mnist_cnn_user.json`.

---

## Project history

| Version | What changed |
|---|---|
| 01 | CNN trained from scratch on MNIST with TensorFlow/Keras (98%+ test accuracy) |
| 02 | Virtual drawing canvas in Google Colab (ipycanvas): draw with the mouse, live predictions |
| 03 | Drawing in the air: MediaPipe hand tracking, open palm to clear, user personalisation |
| 03.1 | Smoother strokes (averaged fingertip position, lines between points) |
| 03.2 | Better preprocessing (keeps proportions, centring), top-3 predictions |
| 04 | 62-class EMNIST model (digits + letters), index + middle "move" gesture |
| 04.1 / 04.2 | Start menu: Numbers or Alphabets, Personalize or Recognize; uppercase-only model |
| 04.3 | Live corrections with in-memory fine-tuning |
| 04.3_2 | Feedback saved to disk, automatic fine-tuning at 100 samples |
| **04.4** (`app.py`) | **Digits-only rewrite:** fixed inverted model input (big accuracy gain), MNIST-style centring, multi-stroke digits, safe personalisation, smoother drawing, corrections typed in the window |

The older prototypes (03 onwards) are in `prototypes/`. Run them **from inside that folder** so they find their models, for example:

```bash
cd prototypes
python prototype_04.3_2.py
```

Any files they create (personalised models, feedback samples) stay inside `prototypes/`, separate from `app.py`'s data.

---

## Future work

- **Two-digit numbers (0–99):** split the canvas into separate digits and recognise each one
- **Palm as a region eraser:** erase only the part of the drawing under the palm, instead of clearing everything
- **Alternative gesture scheme:** index finger only moves the cursor, index + middle finger draws (an option for users who find it more natural)
- **Uppercase letters (A–Z):** bring back alphabet recognition with the 4.4 preprocessing and personalisation fixes
- **AR/VR and mobile:** deploy on headsets and phones using TensorFlow Lite

## Built with

[Python](https://www.python.org/) · [MediaPipe](https://developers.google.com/mediapipe) · [OpenCV](https://opencv.org/) · [TensorFlow / Keras](https://www.tensorflow.org/) · [NumPy](https://numpy.org/)

## Author

**Dhruv Agarwalla**
[GitHub](https://github.com/DhruvAgarwalla) · [LinkedIn](https://www.linkedin.com/in/dhruvagarwalla/)

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
