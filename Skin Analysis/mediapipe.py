import cv2
import numpy as np
import csv
import datetime
import mediapipe as mp

# ---------------- FACE DETECTOR ---------------- #
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ---------------- REGION CUTTER ---------------- #
def get_face_roi(frame, box):
    h, w, _ = frame.shape
    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)
    return frame[y1:y2, x1:x2]

# ---------------- FEATURE 1: ACNE SEVERITY ---------------- #
def acne_score(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 9)
    diff = cv2.absdiff(gray, blur)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    acne_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    severity = (acne_pixels / total_pixels) * 100

    return round(severity, 2), thresh

# ---------------- FEATURE 2: DARK SPOTS / HYPERPIGMENTATION ---------------- #
def darkspot_score(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    dark_mask = cv2.threshold(L, 80, 255, cv2.THRESH_BINARY_INV)[1]
    dark_area = np.sum(dark_mask > 0)
    total = dark_mask.size

    return round((dark_area / total) * 100, 2), dark_mask

# ---------------- FEATURE 3: WRINKLE SCORE ---------------- #
def wrinkle_score(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    wrinkle_pixels = np.sum(edges > 0)
    total = edges.size

    return round((wrinkle_pixels / total) * 100, 2), edges

# ---------------- FEATURE 4: PORE DENSITY ---------------- #
def pore_score(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhance = cv2.bilateralFilter(gray, 10, 80, 80)
    diff = cv2.absdiff(gray, enhance)
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    pore_pixels = np.sum(mask > 0)
    total = mask.size

    return round((pore_pixels / total) * 100, 2), mask

# ---------------- FEATURE 5: SKIN TONE UNIFORMITY ---------------- #
def tone_uniformity(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    std = np.std(L)
    uniformity = max(0, 100 - std)

    return round(uniformity, 2)

# ---------------- FEATURE 6: HYDRATION / OILINESS ---------------- #
def hydration_score(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    oil_estimate = np.mean(V)  
    hydration = 100 - oil_estimate  

    return round(hydration, 2)

# ---------------- FEATURE 7: UV DAMAGE PLACEHOLDER ---------------- #
def uv_damage_placeholder():
    return "Requires UV camera or ML model"

# ---------------- FEATURE 8: SAVE TO CSV ---------------- #
def save_csv(data):
    filename = "skin_daily_log.csv"
    header = ["Date", "Acne", "Dark Spots", "Wrinkles", "Pores", "Tone Uniformity", "Hydration"]

    file_exists = False
    try:
        open(filename, "r")
        file_exists = True
    except:
        pass

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

# ---------------- FEATURE 9: SKINCARE RECOMMENDATIONS ---------------- #
def skin_recommend(acne, dark, wrinkles, pores, tone, hydration):
    rec = []

    if acne > 5:
        rec.append("Use salicylic acid cleanser")
    if dark > 8:
        rec.append("Add Vitamin C serum")
    if wrinkles > 10:
        rec.append("Use retinol at night")
    if pores > 10:
        rec.append("Use clay mask weekly")
    if hydration < 40:
        rec.append("Use hyaluronic acid + drink more water")
    if tone < 60:
        rec.append("Try niacinamide for even tone")

    if len(rec) == 0:
        return ["Your skin looks balanced today"]

    return rec

# ---------------- MAIN LOOP ---------------- #
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            face = get_face_roi(frame, box)

            if face is None or face.size == 0:
                continue

            # Call features
            acne, acne_mask = acne_score(face)
            dark, dark_mask = darkspot_score(face)
            wrinkle, wrinkle_mask = wrinkle_score(face)
            pores, pore_mask = pore_score(face)
            tone = tone_uniformity(face)
            hydrate = hydration_score(face)

            # Display simple overlay
            cv2.putText(frame, f"Acne: {acne}%", (30, 30), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Dark Spots: {dark}%", (30, 60), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Wrinkles: {wrinkle}%", (30, 90), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Pores: {pores}%", (30, 120), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Tone: {tone}%", (30, 150), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Hydration: {hydrate}%", (30, 180), 0, 0.8, (0, 255, 0), 2)

    cv2.imshow("Skin Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save daily log once per run
today = datetime.date.today().isoformat()
save_csv([today, acne, dark, wrinkle, pores, tone, hydrate])

print("\n=== SKIN ANALYSIS RESULTS ===")
print(f"Acne: {acne}%")
print(f"Dark Spots: {dark}%")
print(f"Wrinkles: {wrinkle}%")
print(f"Pores: {pores}%")
print(f"Tone Uniformity: {tone}%")
print(f"Hydration: {hydrate}%")
print("\nRecommendations:")
for r in skin_recommend(acne, dark, wrinkle, pores, tone, hydrate):
    print(" -", r)
