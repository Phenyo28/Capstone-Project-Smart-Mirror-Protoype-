#!/usr/bin/env python3
"""
realtime_makeup_ar.py

Threaded real-time facial landmarks with AR makeup using dlib + OpenCV.
Add shape_predictor_68_face_landmarks.dat and AR images (glasses.png, eyebrow.png, lipstick.png) in the same folder.
"""

import time
import threading
from collections import deque
import cv2
import dlib
import argparse
import os
import numpy as np

class ThreadedCam:
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

def shape_to_np(shape, dtype="int"):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=dtype)

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]
    if overlay.shape[2] < 4:
        return background
    alpha_overlay = overlay[:,:,3] / 255.0
    alpha_background = 1.0 - alpha_overlay
    for c in range(3):
        background[y:y+h, x:x+w, c] = alpha_overlay * overlay[:h,:w,c] + alpha_background * background[y:y+h, x:x+w,c]
    return background

def main(predictor_path, src=0, display_scale=1.0, process_scale=0.5, skip_frames=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Load AR images
    glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    eyebrow_img = cv2.imread("eyebrow.png", cv2.IMREAD_UNCHANGED)
    lipstick_img = cv2.imread("lipstick.png", cv2.IMREAD_UNCHANGED)

    cam = ThreadedCam(src=src)
    time.sleep(0.2)

    show_face = True
    show_eyes = True
    show_landmarks = True
    show_ar = True
    skip_mode = False

    fps_deque = deque(maxlen=16)
    last_time = time.time()
    frame_count = 0

    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("Controls: f=toggle face, e=toggle eyes, l=toggle landmarks, a=toggle AR, k=skip mode, s=snapshot, q/Esc=quit")

    while True:
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        proc = cv2.resize(frame, (0,0), fx=process_scale, fy=process_scale)
        gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        do_process = True
        if skip_mode and (frame_count % (skip_frames + 1) != 0):
            do_process = False

        face_shapes = []
        face_bbs = []

        if do_process:
            rects = detector(gray_proc, 0)
            for rect in rects:
                inv_scale = 1.0 / process_scale
                x, y, w, h = rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()
                bb_x, bb_y, bb_w, bb_h = int(x*inv_scale), int(y*inv_scale), int(w*inv_scale), int(h*inv_scale)
                face_bbs.append((bb_x, bb_y, bb_w, bb_h))

                shape = predictor(gray_proc, rect)
                pts = shape_to_np(shape)
                pts = (pts.astype("float") * inv_scale).astype("int")
                face_shapes.append(pts)

        display = frame.copy()

        # Draw face rectangle
        for idx, bb in enumerate(face_bbs):
            x, y, w, h = bb
            if show_face:
                cv2.rectangle(display, (x,y), (x+w, y+h), (255,0,0), 2)

        for pts in face_shapes:
            if show_landmarks:
                for (px, py) in pts:
                    cv2.circle(display, (px, py), 2, (0,255,0), -1)
            if show_eyes:
                try:
                    left_eye = pts[36:42]
                    right_eye = pts[42:48]
                    lx, ly = int(left_eye[:,0].mean()), int(left_eye[:,1].mean())
                    rx, ry = int(right_eye[:,0].mean()), int(right_eye[:,1].mean())
                    cv2.circle(display, (lx, ly), 4, (0,165,255), -1)
                    cv2.circle(display, (rx, ry), 4, (0,165,255), -1)
                except Exception:
                    pass

            if show_ar:
                # Glasses
                if glasses_img is not None:
                    eye_center_x = int((left_eye[:,0].mean() + right_eye[:,0].mean()) / 2)
                    eye_center_y = int((left_eye[:,1].mean() + right_eye[:,1].mean()) / 2)
                    eye_width = int(np.linalg.norm(left_eye[0] - right_eye[3])) * 2
                    glasses_scaled = cv2.resize(glasses_img, (eye_width, int(glasses_img.shape[0] * (eye_width/glasses_img.shape[1]))))
                    display = overlay_transparent(display, glasses_scaled, eye_center_x - eye_width//2, eye_center_y - glasses_scaled.shape[0]//2)

                # Eyebrows
                if eyebrow_img is not None:
                    for brow in [pts[17:22], pts[22:27]]:
                        x_b, y_b = brow[:,0].min(), brow[:,1].min()
                        w_b, h_b = brow[:,0].max()-x_b, brow[:,1].max()-y_b
                        brow_scaled = cv2.resize(eyebrow_img, (w_b, h_b))
                        display = overlay_transparent(display, brow_scaled, x_b, y_b)

                # Lipstick
                if lipstick_img is not None:
                    lips = pts[48:60]
                    x_l, y_l = lips[:,0].min(), lips[:,1].min()
                    w_l, h_l = lips[:,0].max()-x_l, lips[:,1].max()-y_l
                    lips_scaled = cv2.resize(lipstick_img, (w_l, h_l))
                    display = overlay_transparent(display, lips_scaled, x_l, y_l)

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0/dt if dt>0 else 0.0
        fps_deque.append(fps)
        fps_smoothed = sum(fps_deque)/len(fps_deque)
        status = f"FPS:{fps_smoothed:.1f} | face:{'ON' if show_face else 'OFF'} eye:{'ON' if show_eyes else 'OFF'} lm:{'ON' if show_landmarks else 'OFF'} AR:{'ON' if show_ar else 'OFF'} skip:{skip_mode}"
        cv2.putText(display, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Real-time Makeup AR", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('f'):
            show_face = not show_face
        elif key == ord('e'):
            show_eyes = not show_eyes
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('a'):
            show_ar = not show_ar
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
    ap.add_argument("-c", "--camera", type=int, default=0, help="camera device index")
    ap.add_argument("--process-scale", type=float, default=0.5, help="scale to downsample frame for processing")
    ap.add_argument("--display-scale", type=float, default=1.0, help="scale for display")
    ap.add_argument("--skip-frames", type=int, default=0, help="frames to skip between processing")
    args = ap.parse_args()
    if not os.path.exists(args.predictor):
        print("Predictor not found at:", args.predictor)
        print("Place shape_predictor_68_face_landmarks.dat in the same folder or pass -p /path/to/file")
        exit(1)
    main(predictor_path=args.predictor, src=args.camera, display_scale=args.display_scale,
         process_scale=args.process_scale, skip_frames=args.skip_frames)
