from roboflow import Roboflow
from ultralytics import YOLO
import os

def main():
    # Initialize Roboflow with your API key
    rf = Roboflow(api_key="TorpbqiaFqXHWjftu9B2")

    # Download dataset
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    dataset = project.version(11).download("yolov11")
    dataset_yaml = os.path.join(dataset.location, "data.yaml")

    # Load model
    model = YOLO("yolo11s.pt")  # or yolo11s.pt, etc.

    # Train model
    model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=630,
        batch=8,workers=0,
        device=0,
        project='yolov11_roboflow_plate',
        name='plate_detector',
        exist_ok=True,
        val=True,
        save=True
    )
# Save the model
    model.save("yolo11_roboflow_plate.pt")
    
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    