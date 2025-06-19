import cv2
import os
import numpy as np

base_path = "known_faces"
images = []

for person in os.listdir(base_path):
    person_path = os.path.join(base_path, person)
    imgs = os.listdir(person_path)
    if not imgs:
        continue
    img = cv2.imread(os.path.join(person_path, imgs[0]))
    if img is not None:
        images.append(cv2.resize(img, (300, 300)))

# Combine all in a single row
collage = cv2.hconcat(images)
cv2.imwrite("test.jpg", collage)

print("✅ test.jpg created.")
