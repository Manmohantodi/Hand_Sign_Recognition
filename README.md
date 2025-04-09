# Hand Sign Detection using TensorFlow Lite and OpenCV

This project implements a real-time hand gesture recognition system using OpenCV, MediaPipe (via `cvzone`), and TensorFlow Lite. It detects hand gestures from webcam input and classifies them using a lightweight, efficient TFLite model.

---

## 📌 Features

- Real-time hand gesture detection and classification
- Uses TensorFlow Lite for fast and efficient model inference
- Webcam-based UI with bounding box and label display
- Easy-to-collect dataset pipeline for gesture training
- Clean and modular Python code

---

## 🗂️ Project Structure

```
hand_sign_detection/
├── model_unquant.tflite       # Pre-trained TFLite model
├── labels.txt                 # Labels for hand signs
├── DataCollection.py          # Script to collect hand sign data
├── test.py                    # Main script for real-time detection
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/hand_sign_detection.git
cd hand_sign_detection
```

2. **Create a virtual environment (recommended)**:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Use

### 1. Data Collection

To collect training images for hand signs:

```bash
python DataCollection.py
```

- Make sure your webcam is connected
- Press `S` to save detected hand images
- Press `Q` to quit

### 2. Real-time Hand Sign Detection

Run the detection script with:

```bash
python test.py
```

- Uses webcam input
- Press `Q` in the OpenCV window to quit the program cleanly

---

## 📁 Model & Labels

- `model_unquant.tflite`: A quantized TensorFlow Lite model trained to classify hand gestures.
- `labels.txt`: Contains gesture labels in order of the model’s output indices.

If you want to use a different model:
- Replace the `.tflite` file
- Update `labels.txt` with the new gesture names

---

## 🧠 Custom Model Training (Optional)

If you plan to train your own gesture recognition model:

1. Collect data using `DataCollection.py`
2. Train a model in Keras using your dataset
3. Convert the `.h5` model to `.tflite`
4. Update `test.py` and `labels.txt` accordingly

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Manmohan Todi**  
B.Tech ECE @ IIIT Kalyani  
Email: mannagrwal@gmail.com  
GitHub: [@manmohantodi](https://github.com/manmohantodi)


