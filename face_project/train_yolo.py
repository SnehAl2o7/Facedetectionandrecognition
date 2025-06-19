from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8m.pt for better accuracy

model.train(
    data='face.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='face_detector'
)
