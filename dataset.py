from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Set ROBOFLOW_API_KEY before downloading the dataset.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(11)
dataset = version.download("yolov11")
