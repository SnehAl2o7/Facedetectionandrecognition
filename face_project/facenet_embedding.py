import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2

# Load FaceNet model (make sure this path is correct)
FACENET_PATH = 'facenet_keras.h5'  # download from below

model = load_model(FACENET_PATH)

# Resize and pre-whiten input image
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    return (face_img - mean) / std

# Generate 128D embedding
def get_embedding(face_img):
    face_img = preprocess_face(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return model.predict(face_img)[0]
