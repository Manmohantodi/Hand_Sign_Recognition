import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf  # TensorFlow must be installed

# --- Custom TFLite Classifier Class ---
class TFLiteClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def getPrediction(self, image):
        img = cv2.resize(image, (224, 224))  # Adjust to your model's input size
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        index = int(np.argmax(output_data))
        confidence = output_data[index]
        return [confidence], index

# --- Initialize ---
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = TFLiteClassifier(
    "/Users/manmohantodi/hand_sign_detection/model_unquant.tflite",
    "/Users/manmohantodi/hand_sign_detection/labels.txt"
)

offset = 20
imgSize = 300
counter = 0

# Load labels from file
with open("/Users/manmohantodi/hand_sign_detection/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Main Loop ---
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite)
        label_text = labels[index]

        # Draw label and bounding box
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Optional: Show cropped images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)

    # ✅ Press 'q' to quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Quitting...")
        break

# ✅ Clean up resources
cap.release()
cv2.destroyAllWindows()
