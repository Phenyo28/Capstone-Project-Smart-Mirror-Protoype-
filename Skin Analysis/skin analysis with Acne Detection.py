import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from fdlite import FaceDetection, FaceDetectionModel
import os

# Initialize face detector
detector = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)

# CSV log file
log_file = "skin_log.csv"
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Date", "Brightness", "Oiliness", "Redness", "Texture", "SkinType", "AcneScore"])
    df.to_csv(log_file, index=False)

def analyze_skin(face_img):
    """Calculate skin metrics."""
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    oiliness = np.mean(hsv[:, :, 1])
    redness = np.mean(face_img[:, :, 2])
    texture = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 3).var()
    return brightness, oiliness, redness, texture

def classify_skin(brightness, oiliness, texture):
    """Simple heuristic for skin type classification."""
    if oiliness > 120:
        if texture < 100:
            return "Oily/Combination"
        else:
            return "Oily"
    elif brightness < 100:
        return "Dry"
    else:
        return "Normal"

def detect_acne(face_img):
    """Detect acne regions and return score."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, acne_mask = cv2.threshold(lap_norm, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    acne_score = len(contours)
    return acne_score

def log_skin_metrics(brightness, oiliness, redness, texture, skin_type, acne_score):
    df = pd.DataFrame([[datetime.now(), brightness, oiliness, redness, texture, skin_type, acne_score]],
                      columns=["Date", "Brightness", "Oiliness", "Redness", "Texture", "SkinType", "AcneScore"])
    df.to_csv(log_file, mode='a', header=False, index=False)

# Video capture
cap = cv2.VideoCapture(0)
margin = 5  # margin to prevent empty face regions

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    if faces:
        face = faces[0]
        height, width, _ = frame.shape
        xmin = max(0, int(face.bbox.xmin) - margin)
        ymin = max(0, int(face.bbox.ymin) - margin)
        xmax = min(width, int(face.bbox.xmax) + margin)
        ymax = min(height, int(face.bbox.ymax) + margin)

        # Debug: print coordinates
        print(f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

        if xmax > xmin and ymax > ymin:
            face_img = frame[ymin:ymax, xmin:xmax]

            # Skin analysis
            brightness, oiliness, redness, texture = analyze_skin(face_img)
            skin_type = classify_skin(brightness, oiliness, texture)
            acne_score = detect_acne(face_img)

            # Log results
            log_skin_metrics(brightness, oiliness, redness, texture, skin_type, acne_score)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display metrics
            cv2.putText(frame, f"Brightness: {brightness:.1f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Oiliness: {oiliness:.1f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Redness: {redness:.1f}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Texture: {texture:.1f}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Skin Type: {skin_type}", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Acne Score: {acne_score}", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            print("Invalid face coordinates, skipping frame")

    cv2.imshow("Smart Mirror Skin Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
