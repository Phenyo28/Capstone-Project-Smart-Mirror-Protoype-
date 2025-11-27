# advanced_skin_analysis.py
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from fdlite import FaceDetection, FaceDetectionModel
import os
import math

# ----------------- CONFIG / THRESHOLDS -----------------
LOG_CSV = "advanced_skin_log.csv"
DETECTION_MARGIN = 30            # enlarge face bbox to include cheeks
ACNE_LAPLACIAN_THRESH = 140     # lower -> more sensitive
DARK_SPOT_THRESH = 110          # grayscale threshold for dark spots
WRINKLE_CANNY_LOW = 40
WRINKLE_CANNY_HIGH = 120
PORE_MIN_AREA = 10              # min blob area for pores
PORE_MAX_AREA = 300             # max blob area for pores
SPECULAR_V_THRESH = 230         # V channel above this considered specular highlight
UNIFORMITY_SAMPLE_PATCH = 7     # grid sample size (7x7) for tone uniformity
MIN_FACE_AREA = 10000           # ignore tiny detections
LOG_EVERY_N_FRAMES = 5          # reduce CSV logging freq if needed
# ------------------------------------------------------

# Initialize face detector
detector = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)

# Prepare CSV
if not os.path.exists(LOG_CSV):
    cols = ["Date", "Region", "Brightness", "Oiliness", "Redness", "Texture",
            "SkinType", "AcneScore", "DarkSpotAreaPct", "WrinkleScore", "PoreDensity",
            "ToneUniformity", "SpecularPct", "SunDamageProxy", "Recommendation"]
    pd.DataFrame(columns=cols).to_csv(LOG_CSV, index=False)

# ---------- Utility / analysis functions ----------

def safe_crop(frame, xmin, ymin, xmax, ymax):
    h, w = frame.shape[:2]
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(w, xmax); ymax = min(h, ymax)
    if xmax > xmin and ymax > ymin:
        return frame[ymin:ymax, xmin:xmax]
    return None

def analyze_basic(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:,:,2]))
    oiliness = float(np.mean(hsv[:,:,1]))
    redness = float(np.mean(face_img[:,:,2]))
    texture = float(cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 3).var())
    return brightness, oiliness, redness, texture

def classify_skin(brightness, oiliness, texture):
    # Simple heuristic, tweak as needed
    if oiliness > 125:
        if texture < 120:
            return "Oily/Combination"
        else:
            return "Oily"
    elif brightness < 95:
        return "Dry"
    else:
        return "Normal"

# ACNE detection via Laplacian high-variance spots (per region)
def acne_score_and_mask(face_img, thresh=ACNE_LAPLACIAN_THRESH):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, acne_mask = cv2.threshold(lap_norm, thresh, 255, cv2.THRESH_BINARY)
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 8]
    return len(boxes), acne_mask, boxes

# Dark spot / hyperpigmentation detection
def dark_spot_analysis(face_img, thresh=DARK_SPOT_THRESH):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    dark_area = np.count_nonzero(dark_mask)
    total_area = face_img.shape[0]*face_img.shape[1]
    pct = 100.0 * dark_area / total_area if total_area>0 else 0.0
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 25]
    return pct, dark_mask, boxes

# Wrinkle detection (edge density)
def wrinkle_score(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, WRINKLE_CANNY_LOW, WRINKLE_CANNY_HIGH)
    score = int(np.sum(edges>0))
    return score, edges

# Pore detection via dark circular blobs (needs good resolution)
def pore_detection(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # High-pass filter to boost tiny dark features
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.uint8(np.absolute(lap))
    # Threshold small dark blobs
    _, pore_mask = cv2.threshold(lap_abs, 30, 255, cv2.THRESH_BINARY_INV)
    # Morph ops
    pore_mask = cv2.morphologyEx(pore_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # Blob detection contours in given area range
    contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if PORE_MIN_AREA <= cv2.contourArea(c) <= PORE_MAX_AREA]
    density = len(blobs) / (face_img.shape[0]*face_img.shape[1]) * 1e4  # scaled density
    boxes = [cv2.boundingRect(c) for c in blobs]
    return density, pore_mask, boxes

# Tone uniformity using Lab colors & ΔE sampling
def tone_uniformity(face_img, samples=UNIFORMITY_SAMPLE_PATCH):
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = face_img.shape[:2]
    ys = np.linspace(0, h-1, samples, dtype=int)
    xs = np.linspace(0, w-1, samples, dtype=int)
    samples_list = []
    for yy in ys:
        for xx in xs:
            L,a,b = lab[yy,xx]
            samples_list.append((L,a,b))
    # compute mean Lab and average ΔE from mean
    samples_arr = np.array(samples_list)
    mean = samples_arr.mean(axis=0)
    deltaE = np.sqrt(np.sum((samples_arr - mean)**2, axis=1))
    uniformity = float(np.mean(deltaE))  # lower = more uniform
    return uniformity

# Specular highlights (hydration/oiliness proxy)
def specular_percent(face_img, v_thresh=SPECULAR_V_THRESH):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    spec_mask = (v >= v_thresh).astype(np.uint8)*255
    spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    spec_pct = 100.0 * np.count_nonzero(spec_mask) / (face_img.shape[0]*face_img.shape[1])
    return spec_pct, spec_mask

# Sun damage proxy - use dark_spot_area% as a naive proxy
def sun_damage_proxy(dark_pct):
    # simple mapping: >8% significant, 4-8 moderate, <4 low (tweak for your use)
    if dark_pct > 8:
        return "High"
    elif dark_pct > 4:
        return "Moderate"
    else:
        return "Low"

# Rule-based skincare recommendations
def recommendations(row):
    recs = []
    # acne
    if row["AcneScore"] >= 8:
        recs.append("Acne: consider salicylic acid cleanser, topical benzoyl peroxide")
    elif row["AcneScore"] >= 3:
        recs.append("Mild acne: spot treatment + gentle cleansing")
    # oiliness
    if row["Oiliness"] > 140 or row["SpecularPct"] > 2.5:
        recs.append("Oily: consider oil-control cleanser and mattifying moisturizer")
    # dryness
    if row["Brightness"] < 95:
        recs.append("Dry: hydrating serum (hyaluronic acid) + richer moisturizer")
    # dark spots
    if row["DarkSpotAreaPct"] > 3:
        recs.append("Hyperpigmentation: consider vitamin C or niacinamide")
    # wrinkles
    if row["WrinkleScore"] > 5000:
        recs.append("Wrinkles: consider retinoid / professional treatment consultation")
    # pores
    if row["PoreDensity"] > 0.5:
        recs.append("Pores: exfoliation + clay masks may help")
    if not recs:
        recs.append("Skin is generally stable — maintain current routine")
    return " | ".join(recs)

# -------------------- Main loop --------------------

cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    if faces:
        face = faces[0]
        h, w, _ = frame.shape
        xmin = max(0, int(face.bbox.xmin) - DETECTION_MARGIN)
        ymin = max(0, int(face.bbox.ymin) - DETECTION_MARGIN)
        xmax = min(w, int(face.bbox.xmax) + DETECTION_MARGIN)
        ymax = min(h, int(face.bbox.ymax) + DETECTION_MARGIN)
        # debug print
        # print(xmin, ymin, xmax, ymax)

        if xmax > xmin and ymax > ymin:
            face_img = frame[ymin:ymax, xmin:xmax]
            area = face_img.shape[0]*face_img.shape[1]
            if area < MIN_FACE_AREA:
                # too small — skip logging but still show
                cv2.putText(frame, "Face too small for analysis", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                # split into regions: forehead, left cheek, right cheek, chin
                fh = int(0.2 * face_img.shape[0])
                ch = int(0.45 * face_img.shape[0])
                # Regions within face_img coordinates
                forehead = face_img[0:fh, int(0.2*face_img.shape[1]):int(0.8*face_img.shape[1])]
                left_cheek = face_img[int(0.2*face_img.shape[0]):int(0.7*face_img.shape[0]), 0:int(0.45*face_img.shape[1])]
                right_cheek = face_img[int(0.2*face_img.shape[0]):int(0.7*face_img.shape[0]), int(0.55*face_img.shape[1]):face_img.shape[1]]
                chin = face_img[int(0.7*face_img.shape[0]):face_img.shape[0], int(0.25*face_img.shape[1]):int(0.75*face_img.shape[1])]

                regions = {
                    "forehead": forehead,
                    "left_cheek": left_cheek,
                    "right_cheek": right_cheek,
                    "chin": chin
                }

                # aggregated region metrics
                aggregated = {
                    "Brightness": [], "Oiliness": [], "Redness": [], "Texture": [],
                    "AcneScore": [], "DarkPct": [], "WrinkleScore": [], "PoreDensity": [], "SpecularPct": [], "ToneUniformity": []
                }

                # keep masks/boxes for overlay
                overlay_boxes = {"acne": [], "dark": [], "pores": [], "wrinkle_edges": None, "specular_mask": None}

                for rname, rimg in regions.items():
                    if rimg is None or rimg.size == 0:
                        continue
                    b, o, rd, tex = analyze_basic(rimg)
                    a_score, a_mask, a_boxes = acne_score_and_mask(rimg)
                    d_pct, d_mask, d_boxes = dark_spot_analysis(rimg)
                    w_score, w_edges = wrinkle_score(rimg)
                    pore_density, pore_mask, pore_boxes = pore_detection(rimg)
                    tone_u = tone_uniformity(rimg)
                    spec_pct, spec_mask = specular_percent(rimg)

                    aggregated["Brightness"].append(b)
                    aggregated["Oiliness"].append(o)
                    aggregated["Redness"].append(rd)
                    aggregated["Texture"].append(tex)
                    aggregated["AcneScore"].append(a_score)
                    aggregated["DarkPct"].append(d_pct)
                    aggregated["WrinkleScore"].append(w_score)
                    aggregated["PoreDensity"].append(pore_density)
                    aggregated["SpecularPct"].append(spec_pct)
                    aggregated["ToneUniformity"].append(tone_u)

                    # bboxes are relative to rimg; convert to face_img coordinates for overlay
                    def rel_to_face(boxes, rtop, rleft):
                        return [(rleft + x, rtop + y, w_, h_) for (x,y,w_,h_) in boxes]

                    rtop = 0
                    rleft = 0
                    if rname == "forehead":
                        rtop = 0
                        rleft = int(0.2*face_img.shape[1])
                    elif rname == "left_cheek":
                        rtop = int(0.2*face_img.shape[0])
                        rleft = 0
                    elif rname == "right_cheek":
                        rtop = int(0.2*face_img.shape[0])
                        rleft = int(0.55*face_img.shape[1])
                    elif rname == "chin":
                        rtop = int(0.7*face_img.shape[0])
                        rleft = int(0.25*face_img.shape[1])

                    overlay_boxes["acne"].extend(rel_to_face(a_boxes, rtop, rleft))
                    overlay_boxes["dark"].extend(rel_to_face(d_boxes, rtop, rleft))
                    overlay_boxes["pores"].extend(rel_to_face(pore_boxes, rtop, rleft))
                    # stitch wrinkles edges (single mask) — we overlay later on face_img coords
                    if overlay_boxes["wrinkle_edges"] is None:
                        overlay_boxes["wrinkle_edges"] = np.zeros(face_img.shape[:2], dtype=np.uint8)
                    if w_edges is not None:
                        # place w_edges into the right region
                        overlay_boxes["wrinkle_edges"][rtop:rtop+w_edges.shape[0], rleft:rleft+w_edges.shape[1]] = np.maximum(
                            overlay_boxes["wrinkle_edges"][rtop:rtop+w_edges.shape[0], rleft:rleft+w_edges.shape[1]],
                            w_edges)
                    # specular mask
                    if overlay_boxes["specular_mask"] is None:
                        overlay_boxes["specular_mask"] = np.zeros(face_img.shape[:2], dtype=np.uint8)
                    if spec_mask is not None:
                        overlay_boxes["specular_mask"][rtop:rtop+spec_mask.shape[0], rleft:rleft+spec_mask.shape[1]] = np.maximum(
                            overlay_boxes["specular_mask"][rtop:rtop+spec_mask.shape[0], rleft:rleft+spec_mask.shape[1]],
                            spec_mask)

                # compute aggregated means
                def mean_or_zero(lst):
                    return float(np.mean(lst)) if lst else 0.0

                row = {
                    "Brightness": mean_or_zero(aggregated["Brightness"]),
                    "Oiliness": mean_or_zero(aggregated["Oiliness"]),
                    "Redness": mean_or_zero(aggregated["Redness"]),
                    "Texture": mean_or_zero(aggregated["Texture"]),
                    "AcneScore": int(sum(aggregated["AcneScore"])),
                    "DarkSpotAreaPct": float(np.mean(aggregated["DarkPct"])) if aggregated["DarkPct"] else 0.0,
                    "WrinkleScore": int(sum(aggregated["WrinkleScore"])),
                    "PoreDensity": float(np.mean(aggregated["PoreDensity"])) if aggregated["PoreDensity"] else 0.0,
                    "ToneUniformity": float(np.mean(aggregated["ToneUniformity"])) if aggregated["ToneUniformity"] else 0.0,
                    "SpecularPct": float(np.mean(aggregated["SpecularPct"])) if aggregated["SpecularPct"] else 0.0
                }
                row["SkinType"] = classify_skin(row["Brightness"], row["Oiliness"], row["Texture"])
                row["SunDamageProxy"] = sun_damage_proxy(row["DarkSpotAreaPct"])
                # Recommendation
                row_for_rec = {
                    "AcneScore": row["AcneScore"],
                    "Oiliness": row["Oiliness"],
                    "SpecularPct": row["SpecularPct"],
                    "Brightness": row["Brightness"],
                    "DarkSpotAreaPct": row["DarkSpotAreaPct"],
                    "WrinkleScore": row["WrinkleScore"],
                    "PoreDensity": row["PoreDensity"]
                }
                row["Recommendation"] = recommendations(row_for_rec)

                # Logging: optionally log only every N frames to reduce CSV size
                frame_counter += 1
                if frame_counter % LOG_EVERY_N_FRAMES == 0:
                    log_row = {
                        "Date": datetime.now(),
                        "Region": "full_face_aggregated",
                        "Brightness": row["Brightness"],
                        "Oiliness": row["Oiliness"],
                        "Redness": row["Redness"],
                        "Texture": row["Texture"],
                        "SkinType": row["SkinType"],
                        "AcneScore": row["AcneScore"],
                        "DarkSpotAreaPct": row["DarkSpotAreaPct"],
                        "WrinkleScore": row["WrinkleScore"],
                        "PoreDensity": row["PoreDensity"],
                        "ToneUniformity": row["ToneUniformity"],
                        "SpecularPct": row["SpecularPct"],
                        "SunDamageProxy": row["SunDamageProxy"],
                        "Recommendation": row["Recommendation"]
                    }
                    pd.DataFrame([log_row]).to_csv(LOG_CSV, mode='a', header=False, index=False)

                # ---------- Overlays ----------
                # draw bounding box for face on main frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                # overlay acne boxes (red)
                for (bx,by,bw,bh) in overlay_boxes["acne"]:
                    cv2.rectangle(frame, (xmin+bx, ymin+by), (xmin+bx+bw, ymin+by+bh), (0,0,255), 1)
                # overlay dark boxes (yellow)
                for (bx,by,bw,bh) in overlay_boxes["dark"]:
                    cv2.rectangle(frame, (xmin+bx, ymin+by), (xmin+bx+bw, ymin+by+bh), (0,255,255), 1)
                # overlay pore boxes (small blue)
                for (bx,by,bw,bh) in overlay_boxes["pores"]:
                    cv2.rectangle(frame, (xmin+bx, ymin+by), (xmin+bx+bw, ymin+by+bh), (255,0,0), 1)
                # wrinkle edges overlay (green edges)
                if overlay_boxes["wrinkle_edges"] is not None:
                    edges_img = overlay_boxes["wrinkle_edges"]
                    # map edges to color and overlay
                    colored = np.zeros_like(frame[ymin:ymax, xmin:xmax])
                    colored[edges_img>0] = (0,255,0)
                    alpha = 0.5
                    frame[ymin:ymax, xmin:xmax] = cv2.addWeighted(frame[ymin:ymax, xmin:xmax], 1.0, colored, alpha, 0)
                # specular overlay (white highlight)
                if overlay_boxes["specular_mask"] is not None:
                    spec = overlay_boxes["specular_mask"]
                    colored = np.zeros_like(frame[ymin:ymax, xmin:xmax])
                    colored[spec>0] = (255,255,255)
                    alpha = 0.35
                    frame[ymin:ymax, xmin:xmax] = cv2.addWeighted(frame[ymin:ymax, xmin:xmax], 1.0, colored, alpha, 0)

                # display text metrics on top-left
                cx = 10
                cy = 30
                step = 30
                cv2.putText(frame, f"SkinType: {row['SkinType']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); cy += step
                cv2.putText(frame, f"Brightness: {row['Brightness']:.1f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy += step
                cv2.putText(frame, f"Oiliness: {row['Oiliness']:.1f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy += step
                cv2.putText(frame, f"Acne: {row['AcneScore']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1); cy += step
                cv2.putText(frame, f"Dark%: {row['DarkSpotAreaPct']:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1); cy += step
                cv2.putText(frame, f"WrinkleScore: {row['WrinkleScore']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1); cy += step
                cv2.putText(frame, f"PoreDensity: {row['PoreDensity']:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1); cy += step
                cv2.putText(frame, f"Specular%: {row['SpecularPct']:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy += step
                cv2.putText(frame, f"SunProxy: {row['SunDamageProxy']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1); cy += step
                # show first part of recommendation
                rec_text = row["Recommendation"][:120] + ("..." if len(row["Recommendation"])>120 else "")
                cv2.putText(frame, f"Rec: {rec_text}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        else:
            cv2.putText(frame, "Invalid face coordinates", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # show frame
    cv2.imshow("Advanced Skin Analysis", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # save a snapshot of face region and latest CSV row
        cv2.imwrite(f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)

cap.release()
cv2.destroyAllWindows()
