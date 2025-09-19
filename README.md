# The Smart Mirror Prototype

This project is my first smart mirror prototype. It is an interactive beauty and self-care device powered by a Raspberry Pi that transforms the traditional mirror into a smart assistant. It combines augmented reality (AR), computer vision, and AI-driven analysis. Designed to enhance daily routines at home or in salons, the smart mirror goes beyond reflection to provide a personalized, engaging, and practical self-care experience, with the potential to expand into additional features like reminders, scheduling, and smart home integration.

With this smart mirror, users can:
1. Try on different hairstyles virtually using augmented reality overlays.
2. Experiment with makeup looks to preview styles before applying them.
3. Watch video tutorials or listen to music while getting ready.
4. Analyze skin and hair condition with the built-in camera and AI-based detection tools.
---

## Hardware Components

- Raspberry Pi 4 Model B (2GB+)
- Raspberry Pi Camera (OV5647 – adjustable focus)
- 7″ HDMI IPS LCD Display
- Acrylic Two-Way Mirror Sheet
- LED Light Strips (for illumination)
- 32GB Micro SD Card (with Raspberry Pi OS)
- Power Supply (5V/3A for Pi, 12V/2A for display if needed)
- Monitor/Screen frame or DIY 3D housing


---
## Software Stack
1. Raspberry Pi OS as the base.
2. Python as the main language.
3. OpenCV + MediaPipe for the core AR camera magic.
4. TensorFlow Lite for the on-device AI analysis.
5. Kivy or PyGame to build the user interface around the video feed.
6. python-vlc to play tutorial videos in the background.
---

## Software Development Workflow

1. Plan Features → Define core functions (time/date, camera, AR overlays, skin/hair analysis). Install Raspberry Pi OS → Flash OS and configure environment.
2. Install Core Software → MagicMirror², Python, OpenCV, TensorFlow Lite/MediaPipe.
3. Test Software → Run sample scripts for camera, overlays, and detection.
4. Develop Features → Build modules (calendar, weather, tutorials), AR overlays, and analysis scripts.
5. Prepare for Integration → Keep modular code for smooth hardware connection later.

---
##  Action/Performance Flow

1. Start → User stands in front of the mirror.
2. Capture → Raspberry Pi Camera records the user’s face and/or hair.
3. Processing → Computer vision (OpenCV/Mediapipe) detects the face, hair, and skin regions.
4. Select Core Feature → User chooses one of the main modes:
- Hairstyle AR Overlay → Different hairstyles are projected onto the user’s reflection.
- Makeup AR Overlay → Virtual makeup looks (lipstick, eyeshadow, foundation) are applied in real time.
- Skin & Hair Analysis → Camera captures close-up details to assess basic skin/hair condition (e.g., dryness, acne, hair thickness).
- Tutorials/Entertainment → Mirror plays makeup tutorials, hair styling videos, or music.
5. Output → Results are displayed live on the mirror screen, allowing the user to interact with the AR overlays or content.
6. End → User can switch between features or exit.

## Action/ Perfomance Flow Diagram
<img width="1235" height="617" alt="Screenshot 2025-09-10 113918" src="https://github.com/user-attachments/assets/fe5ddf67-7e9f-4938-9b4c-a99cadcc726d" />
---

