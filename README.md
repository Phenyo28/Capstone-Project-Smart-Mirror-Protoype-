# The Smart Mirror Prototype

This project is my first smart mirror prototype. It is an interactive beauty and self-care device powered by a Raspberry Pi that transforms the traditional mirror into a smart assistant. It combines augmented reality (AR), computer vision, and AI-driven analysis. Designed to enhance daily routines at home or in salons, the smart mirror goes beyond reflection to provide a personalized, engaging, and practical self-care experience, with the potential to expand into additional features like reminders, scheduling, and smart home integration.

With this smart mirror, users can:
1. Try on different hairstyles virtually using augmented reality overlays.
2. Experiment with makeup looks to preview styles before applying them.
3. Watch video tutorials or listen to music while getting ready.
4. Analyze skin and hair condition with the built-in camera and AI-based detection tools.
---

## Hardware Components

| Component                                           | Description                              | Purpose                                                      |
| --------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **Raspberry Pi 4 Model B (2GB+)**                   | Quad-core SBC with USB, HDMI, Wi-Fi      | Main processing unit for the smart mirror                    |
| **Raspberry Pi Camera (OV5647 – adjustable focus)** | 5MP camera module with manual focus      | Captures face images for skin analysis & AR overlays         |
| **7″ HDMI IPS LCD Display**                         | 1024×600 IPS mini-monitor                | Displays MagicMirror UI and camera results                   |
| **Acrylic Two-Way Mirror Sheet**                    | Reflective & transparent mirror          | Allows the display to shine through while acting as a mirror |
| **LED Light Strips**                                | White/neutral lighting                   | Provides even illumination for accurate skin analysis        |
| **32GB Micro SD Card (Raspberry Pi OS)**            | Storage + operating system               | Holds code, ML models, MagicMirror, and assets               |
| **Power Supply (5V/3A for Pi; 12V/2A for display)** | Stable power input                       | Powers the Raspberry Pi and LCD display safely               |
| **Monitor/screen frame or DIY 3D housing**          | Wooden, acrylic, or 3D-printed enclosure | Holds the mirror, display, camera, and electronics together  |
| **USB or Wireless Keyboard**                        | Full/mini keyboard                       | Used for system setup, debugging, and configuration          |



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

## Skin Analysis
What it does: 
- Brightness
- Oiliness
- Redness
- Texture
<img width="798" height="364" alt="image" src="https://github.com/user-attachments/assets/bf39b12b-6c25-4961-83fc-81ac984638eb" />

What it does: 
- Brightness
- Oiliness
- Redness
- Texture
- Acne Score
- Dark Spots
- Wrinkle Score
<img width="1009" height="866" alt="image" src="https://github.com/user-attachments/assets/c9414369-5318-4886-bb3b-f8ed36a041e2" />

## Makeup Overlay

### Lipstick overlay
<img width="1028" height="840" alt="image" src="https://github.com/user-attachments/assets/bfa29f32-ac2f-4bd3-944a-2cf0e830e8c1" />

### Foundation Overlay
<img width="749" height="656" alt="image" src="https://github.com/user-attachments/assets/cea8dc55-58e9-4086-b953-e1d676b951b2" />


