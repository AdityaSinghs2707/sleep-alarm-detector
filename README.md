# 😴 Sleep Alarm Detector

A real-time drowsiness detection system that monitors your eyes using a webcam and triggers an alarm if you fall asleep — perfect for late-night study sessions or long drives.

---

## 🎥 Demo

> Detects eye closure in real-time using facial landmarks and fires an alarm after 2.5 seconds of closed eyes.

---

## ✨ Features

- 👁️ Real-time Eye Aspect Ratio (EAR) based drowsiness detection
- 🔔 Audio alarm triggered after configurable eye-closed duration
- 📦 Lightweight — runs entirely on CPU, no GPU required
- 🎨 Live overlay showing EAR value, status, and progress bar
- ⚡ Built with MediaPipe Face Landmarker for fast, accurate landmark detection

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `MediaPipe` | Face landmark detection |
| `OpenCV` | Webcam capture & UI overlay |
| `NumPy` | EAR calculation |
| `Pygame` | Alarm sound playback |

---

## 📁 Project Structure

```
Sleep_Detector/
│
├── sleep_alarm.py           # Main script
├── face_landmarker.task     # MediaPipe face landmark model
├── requirements.txt         # Python dependencies
└── dragon-studio-censor-beep-3-372460.mp3   # Alarm sound
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AdityaSinghs2707/sleep-alarm-detector.git
cd sleep-alarm-detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

Download `face_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models) and place it in the project root.

### 5. Run the app

```bash
python sleep_alarm.py
```

Press **Q** to quit.

---

## ⚙️ Configuration

You can tweak these constants at the top of `sleep_alarm.py`:

| Variable | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.22` | EAR value below which eyes are considered closed |
| `EYE_CLOSED_SECONDS` | `2.5` | Seconds of closed eyes before alarm fires |
| `ALARM_SOUND_FILE` | `*.mp3` | Path to your alarm sound file |

---

## 🧠 How It Works

1. Webcam captures each frame in real-time
2. MediaPipe detects 478 facial landmarks
3. Eye Aspect Ratio (EAR) is calculated for both eyes:

$$EAR = \frac{||p_1 - p_5|| + ||p_2 - p_4||}{2 \times ||p_0 - p_3||}$$

4. If EAR drops below threshold for more than `EYE_CLOSED_SECONDS`, the alarm sounds
5. Alarm stops automatically when eyes open again

---

## 📋 Requirements

- Python 3.8+
- Webcam
- macOS / Windows / Linux

---

## 👨‍💻 Author

**Aditya Singh** — [@AdityaSinghs2707](https://github.com/AdityaSinghs2707)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
