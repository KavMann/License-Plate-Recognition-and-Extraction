import cv2
import pytesseract
from ultralytics import YOLO
import re
import matplotlib.pyplot as plt  # Added import for matplotlib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv11 model
model = YOLO("yolov11_roboflow_plate/plate_detector/weights/best.pt")

# Paths
image_path = 'Plate_1711623405530_1711623405684.jpeg'
img = cv2.imread(image_path)

# Run detection
results = model(image_path)

print("\nDetected Plate Numbers:")

# Define regex for Indian plate format (incl. TC plates like MH20TC7444A)
regex = r'[A-Z]{2}\s?\d{2}\s?(?:TC\s?)?[A-Z]{1,3}\s?\d{3,4}'

# Function to format the detected plate number
def format_plate_number(raw_text):
    # Format using regex
    formatted_text = raw_text.strip().upper()
    
    # Format the detected plate number (e.g., TN87C5106 to TN 87 C 5106)
    formatted_text = re.sub(r'([A-Z]{2})(\d{2})([A-Z])(\d{3,4})', r'\1 \2 \3 \4', formatted_text)
    
    return formatted_text

# Process results
for result in results:
    boxes = result.boxes.xyxy.tolist()
    confidences = result.boxes.conf.tolist()  # Get confidence scores

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        plate_crop = img[y1:y2, x1:x2]

        # Preprocessing
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OCR config
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text = pytesseract.image_to_string(resized, config=config)

        # Clean and format plate number
        cleaned = raw_text.strip().replace("\n", "").replace(" ", "")
        
        # Match format using regex
        matches = re.findall(regex, raw_text)
        formatted_plate = format_plate_number(matches[0] if matches else 'None')

        # Extract the confidence score for this detection
        confidence = confidences[i] if i < len(confidences) else 0
        
        print(f" [{i+1}] Raw OCR: {cleaned}")
        print(f"     Formatted Plate: {formatted_plate}")
        print(f"     Confidence: {confidence:.2f}")

        # Draw rectangle around detected plate number
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Overlay formatted plate text and confidence score on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{formatted_plate} ({confidence*100:.2f}%)"
        cv2.putText(img, label, (x1, y1 - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# Convert BGR to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Save the output image with boxes, formatted text, and confidence score
cv2.imwrite("output_with_plate_and_confidence.jpg", img)
