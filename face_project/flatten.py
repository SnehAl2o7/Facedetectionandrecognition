import cv2
import os
from tqdm import tqdm

def convert_to_yolo(img_width, img_height, x, y, w, h):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm

def generate_labels(image_dir, label_dir, class_id):
    os.makedirs(label_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            continue

        img_height, img_width = img.shape[:2]
        label_lines = []

        for (x, y, w, h) in faces:
            x_center, y_center, w_norm, h_norm = convert_to_yolo(img_width, img_height, x, y, w, h)
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

        # Uncomment below to visually verify detections
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

if __name__ == "__main__":
    # Update class_id: 0 for unmasked, 1 for masked, etc.
    class_id = 0  # or 1 depending on folder content

    generate_labels(
        image_dir="datasets/images/train",
        label_dir="datasets/labels/train",
        class_id=class_id
    )

    generate_labels(
        image_dir="datasets/images/val",
        label_dir="datasets/labels/val",
        class_id=class_id
    )
