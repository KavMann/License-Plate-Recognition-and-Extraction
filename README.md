# Automatic Number Plate Recognition

A Flask web application that detects vehicle license plates from image URLs with an Ultralytics YOLO11 detector and extracts plate text with Tesseract OCR.

## Features

- Detects license plates from remote image URLs
- Extracts text from cropped plate regions using Tesseract OCR
- Applies OpenCV preprocessing before OCR
- Formats Indian registration numbers with regex cleanup
- Returns confidence scores and annotated output images
- Includes light/dark mode UI

## Tech Stack

- Python
- Flask
- Ultralytics YOLO11
- OpenCV
- Tesseract OCR
- Roboflow
- Bootstrap, HTML, CSS, JavaScript

## Model Results

The detector was trained for 30 epochs on a Roboflow license plate dataset. Final validation metrics from `results.csv`:

| Metric | Value |
| --- | ---: |
| Precision | 98.62% |
| Recall | 94.38% |
| mAP@50 | 97.25% |
| mAP@50-95 | 70.71% |

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR and configure the executable path.

```bash
copy .env.example .env
```

Update `.env` if your Tesseract installation or model path is different:

```env
MODEL_PATH=yolov11_plate/plate_detector/weights/best.pt
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

4. Run the Flask app.

```bash
python app.py
```

Open `http://127.0.0.1:5000` and submit an image URL.

## Training

Set your Roboflow API key locally before running the training script:

```bash
set ROBOFLOW_API_KEY=your_roboflow_api_key_here
python yolomodel.py
```

Do not commit API keys, generated uploads, generated outputs, or model checkpoints directly to Git.

## Resume Summary

Built a Flask-based ANPR web app using YOLO11 for plate detection and Tesseract OCR for text extraction. Fine-tuned the detector on a Roboflow license plate dataset, achieving 97.25% mAP@50, 98.62% precision, and 94.38% recall, with OpenCV preprocessing, confidence scoring, and Indian plate formatting.
