import cv2
import dlib
import numpy as np

# Paths
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Load dlib detector + predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load all lip images
lip_images = {
    "1": cv2.imread("purple_lip.png", cv2.IMREAD_UNCHANGED),
    "2": cv2.imread("red_lip.png", cv2.IMREAD_UNCHANGED),
    "3": cv2.imread("lip_image.png", cv2.IMREAD_UNCHANGED),
    "4": cv2.imread("pink_lip.png", cv2.IMREAD_UNCHANGED),
    "5": cv2.imread("brownlinerandgloss.png", cv2.IMREAD_UNCHANGED),
}

current_lip = lip_images["1"]  # default lip image

# Function to overlay image with alpha
def overlay_image_alpha(frame, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    alpha = overlay_resized[:, :, 3] / 255.0
    overlay_rgb = overlay_resized[:, :, :3]

    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)

    alpha_crop = alpha[y1 - y:y2 - y, x1 - x:x2 - x]

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha_crop * overlay_rgb[y1 - y:y2 - y, x1 - x:x2 - x, c]
            + (1 - alpha_crop) * frame[y1:y2, x1:x2, c]
        )
    return frame

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        lip_pts = []
        for i in range(48, 61):
            lip_pts.append((landmarks.part(i).x, landmarks.part(i).y))
        lip_pts = np.array(lip_pts)

        x, y, w, h = cv2.boundingRect(lip_pts)

        padding = 5
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        frame = overlay_image_alpha(frame, current_lip, x, y, w, h)

    # Display help text
    cv2.putText(frame, "Press 1-5 to change lip style | ESC to exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Multi-Lip Overlay", frame)

    key = cv2.waitKey(1) & 0xFF

    # Switch between lips
    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
        current_lip = lip_images[chr(key)]

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
