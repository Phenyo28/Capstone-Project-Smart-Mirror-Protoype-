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

## Software Stack
1. Raspberry Pi OS as your base.
2. Python as your main language.
3. OpenCV + MediaPipe for the core AR camera magic.
4. TensorFlow Lite for the on-device AI analysis.
5. Kivy or PyGame to build the user interface around the video feed.
6. python-vlc to play tutorial videos in the background.
