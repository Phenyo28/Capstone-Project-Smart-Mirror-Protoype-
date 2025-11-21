#!/usr/bin/env python3
"""
realtime_landmarks_ar.py

Threaded real-time facial landmarks using dlib + OpenCV with AR overlays.
Supports glasses, hat, mustache, and beauty measurements.
"""

import time
import threading
from collections import deque
import cv2
import dlib
import argparse
import os
import numpy as np

# --- Threaded Camera ---
class ThreadedCam:
    def __init__(self, src=0, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stopped = False
        self.lock = threading.Lock()
        self.grabbed, self.frame = self.cap.read()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            if not grabbed: time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- Helper functions ---
def rect_to_bb(rect):
    return (rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top())

def shape_to_np(shape, dtype="int"):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=dtype)

def overlay_image_alpha(img, img_overlay, x, y, overlay_size=None):
    """Overlay img_overlay (with alpha) onto img at (x, y)."""
    if overlay_size: img_overlay = cv2.resize(img_overlay, overlay_size, interpolation=cv2.INTER_AREA)
    b,g,r,a = cv2.split(img_overlay)
    overlay_color = cv2.merge((b,g,r))
    mask = cv2.merge((a,a,a))
    h, w = overlay_color.shape[:2]
    if y+h>img.shape[0] or x+w>img.shape[1] or x<0 or y<0: return  # skip if out of bounds
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (roi*(1 - mask/255) + overlay_color*(mask/255)).astype(np.uint8)

# --- Main function ---
def main(predictor_path, src=0, display_scale=1.0, process_scale=0.5, skip_frames=0):
    # Load dlib models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cam = ThreadedCam(src=src)
    time.sleep(0.2)  # warm-up

    # Load AR images (make sure you have PNGs with transparency)
    glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    hat_img = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
    mustache_img = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)

    # Toggles
    show_face = True
    show_eyes = True
    show_landmarks = True
    skip_mode = False
    ar_glasses = True
    ar_hat = True
    ar_mustache = True

    # FPS smoothing
    fps_deque = deque(maxlen=16)
    last_time = time.time()
    frame_count = 0

    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("Controls: f=face rect, e=eyes, l=landmarks, k=skip, g=glasses, h=hat, m=mustache, s=snapshot, q/Esc=quit")

    while True:
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        proc = cv2.resize(frame, (0,0), fx=process_scale, fy=process_scale, interpolation=cv2.INTER_LINEAR)
        gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        do_process = True
        if skip_mode and (frame_count % (skip_frames+1) != 0): do_process = False

        face_shapes = []
        face_bbs = []

        if do_process:
            rects = detector(gray_proc, 0)
            for rect in rects:
                x, y, w, h = rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()
                inv_scale = 1.0 / process_scale
                bb_x, bb_y, bb_w, bb_h = int(x*inv_scale), int(y*inv_scale), int(w*inv_scale), int(h*inv_scale)
                face_bbs.append((bb_x, bb_y, bb_w, bb_h))
                shape = predictor(gray_proc, rect)
                pts = shape_to_np(shape)
                pts = (pts.astype("float") * inv_scale).astype("int")
                face_shapes.append(pts)

        display = frame.copy()

        # Draw overlays
        for idx, bb in enumerate(face_bbs):
            x,y,w,h = bb
            if show_face: cv2.rectangle(display, (x,y), (x+w,y+h), (255,0,0), 2)

        for pts in face_shapes:
            if show_landmarks:
                for (px, py) in pts: cv2.circle(display, (px, py), 2, (0,255,0), -1)
            if show_eyes:
                try:
                    left_eye, right_eye = pts[36:42], pts[42:48]
                    lx, ly = int(left_eye[:,0].mean()), int(left_eye[:,1].mean())
                    rx, ry = int(right_eye[:,0].mean()), int(right_eye[:,1].mean())
                    cv2.circle(display, (lx,ly), 4, (0,165,255), -1)
                    cv2.circle(display, (rx,ry), 4, (0,165,255), -1)
                except Exception: pass

            # --- AR overlays ---
            try:
                eye_center = ((left_eye[:,0].mean()+right_eye[:,0].mean())/2,
                              (left_eye[:,1].mean()+right_eye[:,1].mean())/2)
                face_width = int(np.linalg.norm(right_eye[3]-left_eye[0]))
                # Glasses
                if ar_glasses: overlay_image_alpha(display, glasses_img, int(eye_center[0]-face_width*0.5),
                                                   int(eye_center[1]-face_width*0.25),
                                                   overlay_size=(face_width, int(face_width*glasses_img.shape[0]/glasses_img.shape[1])))
                # Hat
                if ar_hat:
                    forehead_y = min(left_eye[:,1].min(), right_eye[:,1].min()) - 50
                    overlay_image_alpha(display, hat_img, int(eye_center[0]-face_width),
                                        int(forehead_y - hat_img.shape[0]*0.5),
                                        overlay_size=(face_width*2, int(face_width*2*hat_img.shape[0]/hat_img.shape[1])))
                # Mustache
                if ar_mustache:
                    nose_y = pts[33,1]  # nose tip
                    overlay_image_alpha(display, mustache_img, int(eye_center[0]-face_width*0.4),
                                        int(nose_y),
                                        overlay_size=(int(face_width*0.8), int(face_width*0.3)))
            except Exception: pass

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0/dt if dt>0 else 0
        fps_deque.append(fps)
        fps_smoothed = sum(fps_deque)/len(fps_deque)

        status = f"FPS:{fps_smoothed:.1f} | face:{'ON' if show_face else 'OFF'} eye:{'ON' if show_eyes else 'OFF'} lm:{'ON' if show_landmarks else 'OFF'} skip:{skip_mode}"
        cv2.putText(display, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("AR Facial Landmarks", display)
        key = cv2.waitKey(1) & 0xFF

        # Key controls
        if key==27 or key==ord('q'): break
        elif key==ord('f'): show_face=not show_face
        elif key==ord('e'): show_eyes=not show_eyes
        elif key==ord('l'): show_landmarks=not show_landmarks
        elif key==ord('k'): skip_mode=not skip_mode
        elif key==ord('g'): ar_glasses=not ar_glasses
        elif key==ord('h'): ar_hat=not ar_hat
        elif key==ord('m'): ar_mustache=not ar_mustache
        elif key==ord('s'):
            ts=int(time.time())
            fname=os.path.join(snapshot_dir,f"snapshot_{ts}.png")
            cv2.imwrite(fname, display)
            print("Saved", fname)

    cam.stop()
    cv2.destroyAllWindows()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--predictor",required=False,default="shape_predictor_68_face_landmarks.dat",
                    help="path to dlib's shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-c","--camera",type=int,default=0,help="camera device index")
    ap.add_argument("--process-scale",type=float,default=0.5,help="scale to downsample frame for processing")
    ap.add_argument("--display-scale",type=float,default=1.0,help="scale for display")
    ap.add_argument("--skip-frames",type=int,default=0,help="number of frames to skip")
    args=ap.parse_args()
    if not os.path.exists(args.predictor):
        print("Predictor not found at:", args.predictor)
        exit(1)
    main(predictor_path=args.predictor, src=args.camera, display_scale=args.display_scale,
         process_scale=args.process_scale, skip_frames=args.skip_frames)
