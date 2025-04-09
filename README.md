# Hand Sign Detection using TFLite and OpenCV

A real-time hand gesture recognition system using TensorFlow Lite, OpenCV, and cvzone's HandTrackingModule. The project includes data collection, model inference, and classification for various hand signs.

## 🔧 Features
- Hand detection and bounding box extraction using cvzone
- Classification using a custom TensorFlow Lite model
- Real-time video stream from webcam
- Easy-to-extend for more gestures

## 📁 Project Structure
```
hand_sign_detection/
├── Data/                     # Collected hand gesture images (per class folder)
│   ├── Hello/
│   ├── Okay/
│   └── Thanks/
│
├── model_unquant.tflite     # Trained TFLite model
├── labels.txt               # Labels for gestures
├── DataCollection.py        # Script to collect hand gesture data
├── test.py                  # Script to test and predict hand gestures
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── .gitignore               # Files and folders to ignore in Git
```

## 🧠 Model
The model used here is a TensorFlow Lite model optimized for real-time classification on devices. It takes 224x224 hand-cropped images as input and predicts the gesture class.

## 🗃️ Dataset
To use this project, you must have a dataset of hand gestures. You can:
1. Run `DataCollection.py` to collect your own hand gesture images.
2. Organize them into folders named by class (e.g., `Hello`, `Okay`, `Thanks`, etc.) under the `/Data` directory.

> **Note**: The actual dataset is not included in this repository. Please use the script to create your own dataset.

## ▶️ How to Run
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Collect Data (Optional)
```bash
python DataCollection.py
```
Use this script to collect images for each hand gesture class.

### 3. Run the Detection Script
```bash
python test.py
```

- Press `q` to quit the real-time prediction window.

## ✅ Requirements
- Python 3.7+
- OpenCV
- TensorFlow
- cvzone
- Numpy

## 🧾 License
This project is licensed under the MIT License. Feel free to use and modify for your needs.

## 🙌 Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for hand tracking utilities
- TensorFlow Lite for lightweight ML inference

---
Feel free to contribute or customize this project further!

