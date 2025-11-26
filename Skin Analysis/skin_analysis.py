import cv2
import numpy as np
from fdlite import FaceDetection, FaceDetectionModel

# Initialize face detector
detector = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)

def analyze_skin(face_img):
    """Analyze brightness, oiliness, redness, and texture of a face image."""
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    oiliness = np.mean(hsv[:, :, 1])
    redness = np.mean(face_img[:, :, 2])
    texture = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 3).var()
    return brightness, oiliness, redness, texture

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break

    # Convert BGR → RGB for fdlite
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    if faces:
        face = faces[0]  # first detected face

        # Convert normalized bbox to pixel coordinates
        h_frame, w_frame = frame.shape[:2]
        xmin = int(face.bbox.xmin * w_frame)
        ymin = int(face.bbox.ymin * h_frame)
        xmax = int(face.bbox.xmax * w_frame)
        ymax = int(face.bbox.ymax * h_frame)

        # Crop face region
        x, y = xmin, ymin
        bw, bh = xmax - xmin, ymax - ymin
        face_img = frame[y:y+bh, x:x+bw]

        # Skin analysis
        brightness, oiliness, redness, texture = analyze_skin(face_img)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

        # Display metrics on the same window
        cv2.putText(frame, f"Brightness: {brightness:.1f}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Oiliness: {oiliness:.1f}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Redness: {redness:.1f}", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Texture: {texture:.1f}", (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Show a single window
    cv2.imshow("Skin Analysis", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
