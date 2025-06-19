import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from facenet_embedding import get_embedding
from numpy.linalg import norm

# ---------------------------
# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load FaceNet database
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)


# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# Threshold for identity confirmation
SIMILARITY_THRESHOLD = 0.75

# ---------------------------
# Load input image (replace with any test image path)
image_path = "test.jpg"
img = cv2.imread(image_path)
assert img is not None, f"Image not found at {image_path}"

# Run YOLO detection
results = model(img)[0]
boxes = results.boxes.xyxy.cpu().numpy()

for box in boxes:
    x1, y1, x2, y2 = map(int, box[:4])
    conf = box[4]

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        continue

    # Generate embedding using FaceNet
    input_embedding = get_embedding(face_crop)

    # Match against known faces
    best_match = "Unknown"
    max_score = -1

    for name, embeddings in face_db.items():
        for emb in embeddings:
            score = cosine_similarity(input_embedding, emb)
            if score > max_score:
                max_score = score
                best_match = name

    # Confirm match
    if max_score < SIMILARITY_THRESHOLD:
        best_match = "Unknown"

    # Annotate image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{best_match} ({max_score:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show result
cv2.imshow("Detected & Recognized Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
