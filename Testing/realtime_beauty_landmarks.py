#!/usr/bin/env python3
"""
realtime_beauty_landmarks.py

Threaded real-time facial landmarks + basic beauty metrics using dlib + OpenCV.
Adjust predictor_path to your shape_predictor_68_face_landmarks.dat location.
"""

import time
import threading
from collections import deque
import cv2
import dlib
import argparse
import os
import numpy as np
import math

class ThreadedCam:
    """Threaded camera reader to reduce frame I/O blocking."""
    def __init__(self, src=0, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stopped = False
        self.lock = threading.Lock()
        self.grabbed, self.frame = self.cap.read()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            if not grabbed:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.stopped = True
        self.cap.release()

def rect_to_bb(rect):
    x, y = rect.left(), rect.top()
    w, h = rect.right() - x, rect.bottom() - y
    return x, y, w, h

def shape_to_np(shape, dtype="int"):
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
    return np.array(coords, dtype=dtype)

def eye_openness(eye_points):
    """Compute eye openness ratio (height/width)."""
    # vertical distances: (p2-p6, p3-p5)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # horizontal distance: (p1-p4)
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ratio = (A + B) / (2.0 * C) if C != 0 else 0
    return ratio

def smile_intensity(mouth_points):
    """Compute smile intensity ratio (width/height)."""
    width = np.linalg.norm(mouth_points[0] - mouth_points[6])
    height = np.linalg.norm(mouth_points[3] - mouth_points[9])
    ratio = width / height if height != 0 else 0
    return ratio

def head_tilt(face_points):
    """Compute simple head tilt (angle between eyes)."""
    left_eye_center = np.mean(face_points[36:42], axis=0)
    right_eye_center = np.mean(face_points[42:48], axis=0)
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def face_alignment(face_points):
    """Approximate face alignment score (distance between eyes and nose base)."""
    left_eye = np.mean(face_points[36:42], axis=0)
    right_eye = np.mean(face_points[42:48], axis=0)
    nose = face_points[33]  # tip of nose
    eye_mid = (left_eye + right_eye) / 2
    dist = np.linalg.norm(eye_mid - nose)
    return dist

def main(predictor_path, src=0, display_scale=1.0, process_scale=0.5, skip_frames=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cam = ThreadedCam(src=src)
    time.sleep(0.2)

    show_face = True
    show_eyes = True
    show_landmarks = True
    skip_mode = False

    fps_deque = deque(maxlen=16)
    last_time = time.time()
    frame_count = 0

    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("Controls: f=toggle face rect, e=toggle eyes, l=toggle landmarks, k=toggle skip mode, s=save snapshot, q/Esc=quit")

    while True:
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        proc = cv2.resize(frame, (0,0), fx=process_scale, fy=process_scale)
        gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        do_process = not (skip_mode and (frame_count % (skip_frames + 1) != 0))

        face_shapes = []
        face_bbs = []

        if do_process:
            rects = detector(gray_proc, 0)
            for rect in rects:
                inv_scale = 1.0 / process_scale
                bb_x, bb_y, bb_w, bb_h = [int(val * inv_scale) for val in rect_to_bb(rect)]
                face_bbs.append((bb_x, bb_y, bb_w, bb_h))

                shape = predictor(gray_proc, rect)
                pts = shape_to_np(shape)
                pts = (pts.astype("float") * inv_scale).astype("int")
                face_shapes.append(pts)

        display = frame.copy()

        for idx, bb in enumerate(face_bbs):
            (x, y, w, h) = bb
            if show_face:
                cv2.rectangle(display, (x,y), (x+w, y+h), (255,0,0), 2)

        for pts in face_shapes:
            if show_landmarks:
                for (px, py) in pts:
                    cv2.circle(display, (px, py), 2, (0,255,0), -1)
            if show_eyes:
                try:
                    left_eye = pts[36:42]; right_eye = pts[42:48]
                    lx, ly = int(left_eye[:,0].mean()), int(left_eye[:,1].mean())
                    rx, ry = int(right_eye[:,0].mean()), int(right_eye[:,1].mean())
                    cv2.circle(display, (lx, ly), 4, (0,165,255), -1)
                    cv2.circle(display, (rx, ry), 4, (0,165,255), -1)
                except Exception:
                    pass

            # Compute beauty metrics
            eye_ratio = (eye_openness(pts[36:42]) + eye_openness(pts[42:48])) / 2
            smile = smile_intensity(pts[48:68])
            tilt = head_tilt(pts)
            alignment = face_alignment(pts)

            cv2.putText(display, f"Eye: {eye_ratio:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, f"Smile: {smile:.2f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, f"Tilt: {tilt:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, f"Align: {alignment:.1f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        fps_deque.append(fps)
        fps_smoothed = sum(fps_deque) / len(fps_deque)

        status = f"FPS:{fps_smoothed:.1f} | face:{'ON' if show_face else 'OFF'} eye:{'ON' if show_eyes else 'OFF'} lm:{'ON' if show_landmarks else 'OFF'} skip:{skip_mode}"
        cv2.putText(display, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Real-time Beauty Landmarks", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('f'):
            show_face = not show_face
        elif key == ord('e'):
            show_eyes = not show_eyes
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('k'):
            skip_mode = not skip_mode
            print("Skip mode:", skip_mode)
        elif key == ord('s'):
            ts = int(time.time())
            fname = os.path.join(snapshot_dir, f"snapshot_{ts}.png")
            cv2.imwrite(fname, display)
            print("Saved", fname)

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--predictor", required=False, default="shape_predictor_68_face_landmarks.dat",
                    help="path to dlib's shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-c", "--camera", type=int, default=0)
    ap.add_argument("--process-scale", type=float, default=0.5)
    ap.add_argument("--display-scale", type=float, default=1.0)
    ap.add_argument("--skip-frames", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(args.predictor):
        print("Predictor not found at:", args.predictor)
        exit(1)
    main(predictor_path=args.predictor, src=args.camera, display_scale=args.display_scale, process_scale=args.process_scale, skip_frames=args.skip_frames)
