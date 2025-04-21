from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import requests
import cv2
import pytesseract
import re
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv11 model
model = YOLO("yolov11_roboflow_plate/plate_detector/weights/best.pt")

# Plate number formatter
def format_plate_number(raw_text):
    formatted_text = raw_text.strip().upper()
    formatted_text = re.sub(r'([A-Z]{2})(\d{2})([A-Z])(\d{3,4})', r'\1 \2 \3 \4', formatted_text)
    return formatted_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/process', methods=['POST'])
def process():
    image_url = request.form.get('imageURL')
    if not image_url:
        return jsonify({'error': 'No image URL provided.'})

    try:
        # Download image
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Unable to download image.'})

        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Run YOLO
        img = cv2.imread(image_path)
        results = model(image_path)
        regex = r'[A-Z]{2}\s?\d{2}\s?(?:TC\s?)?[A-Z]{1,3}\s?\d{3,4}'

        detected_plates = []

        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                plate_crop = img[y1:y2, x1:x2]

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                raw_text = pytesseract.image_to_string(resized, config=config)

                cleaned = raw_text.strip().replace("\n", "").replace(" ", "")
                matches = re.findall(regex, raw_text)
                formatted_plate = format_plate_number(matches[0] if matches else 'None')

                confidence = confidences[i] if i < len(confidences) else 0
                label = f"{formatted_plate} ({confidence*100:.2f}%)"

                # Draw on original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                detected_plates.append({
                    "formatted_plate": formatted_plate,
                    "confidence": round(confidence * 100, 2)
                })

        # Save processed image
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{image_filename}")
        cv2.imwrite(output_path, img)

        return jsonify({
            "plates": detected_plates,
            "processed_image_url": f"/outputs/output_{image_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
