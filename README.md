# 🎯 Face Detection and Recognition using YOLOv8 + FaceNet

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange.svg)
![FaceNet](https://img.shields.io/badge/FaceNet-128D_embedding-red.svg)

---

## 📌 Project Overview

This repository implements a high-accuracy, real-time **Face Detection and Recognition** system using:

- 🧠 **YOLOv8** (Ultralytics) for face detection
- 🔍 **FaceNet** (128D Embeddings) for face recognition
- 😷 Support for **masked** and **unmasked** faces
- ✅ Designed to achieve **98%+ recognition accuracy**

---

## 🚀 Features

| Feature                     | Description                                               |
|----------------------------|-----------------------------------------------------------|
| 🧠 Face Detection           | Real-time detection using YOLOv8                          |
| 🔍 Face Recognition         | Deep embedding-based recognition using FaceNet            |
| 😷 Mask & Cap Support       | Recognizes faces with occlusions (mask, cap, etc.)        |
| 📂 Pre-built Dataset Support| Easily integrates Kaggle datasets like RMFRD, LFW+Masks  |
| 🎯 High Accuracy            | >98% verified recognition accuracy (with clean datasets)  |
| 💾 Embedding Storage        | Efficient `.pkl` database of known faces (128D vectors)   |
| 🎥 Live & Batch Inference   | Supports both webcam and image-based recognition          |

---

## 🗂️ Project Structure

<pre> 📦 <b>Facedetectionandrecognition/</b> └──
  📁 <b>face_project/</b>
    ├── 📁 <b>dataset/</b>
  # YOLO-ready training images and labels (images/train, labels/train) 
  ├── 📁 <b>known_faces/</b> # Person-wise folders with labeled images (e.g. /vansh/, /snehal/)
  ├── 📄 <b>facenet_keras.h5</b> # Pretrained FaceNet model 
  ├── 📄 <b>facenet_embedding.py</b> # FaceNet pre-processing and embedding function 
  ├── 📄 <b>train_yolo.py</b> # YOLOv8 training script
  ├── 📄 <b>generate_face_db.py</b> # Generates embedding database (face_db.pkl) from known_faces/
  ├── 📄 <b>recognize_face.py</b> # Full inference pipeline (YOLO + FaceNet + cosine similarity)
  ├── 📄 <b>face.yaml</b> # Dataset config file for YOLOv8 training 
  └── 🖼️ <b>test.jpg</b> # Sample image with multiple known/unknown faces for testing recognition </pre>


---

## 🧠 Model Details

### 🟠 Face Detection (YOLOv8)

- Model: `yolov8n.pt` (Nano, replaceable with `yolov8s.pt`, etc.)
- Trained on: Flattened RMFRD + LFW-Mask datasets
- Classes: 1 (`face`)

### 🔴 Face Recognition (FaceNet)

- Pretrained Keras `.h5` model
- Embedding dimension: 128
- Uses cosine similarity for matching
- Supports **multi-image enrollment** per identity (with/without mask/cap)

---

## 📦 Dataset Used

| Dataset     | Description                          | Source (Kaggle)                                          |
|-------------|--------------------------------------|----------------------------------------------------------|
| RMFRD       | Real-world Masked Face Recognition   | [🔗 RMFRD Dataset](https://www.kaggle.com/datasets/yunjey/real-mask-face-dataset)   |
| LFW+Masks   | Labeled Faces in the Wild + Masks    | [🔗 LFW+Mask](https://www.kaggle.com/datasets/fu0523/lfw-mask-dataset)              |

---

## 🛠️ Setup Instructions

### 1. Clone Repository

```bash
git clone git@github.com:SnehAl2o7/Facedetectionandrecognition.git
cd Facedetectionandrecognition/face_project

