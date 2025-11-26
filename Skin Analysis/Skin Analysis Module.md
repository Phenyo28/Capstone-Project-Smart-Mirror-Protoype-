## Skin Analysis Module – Smart Mirror Project
This folder contains the Skin Analysis feature of the Smart Mirror system.
The goal of this module is to analyze skin characteristics in real-time using a Raspberry Pi camera, OpenCV, and TensorFlow Lite.

This version is built without MediaPipe, using custom computer-vision techniques and optional TensorFlow Lite models for future ML upgrades.

### Features Included
1. TensorFlow Lite Version (optional model)
- Supports loading a .tflite model for future classification tasks such as:
- Acne detection
- Hyperpigmentation
- Dryness/Oiliness classification
Model runs efficiently on Raspberry Pi.

File:
`tensorflow_lite_model.tflite` (placeholder)

2. Brightness Detection

This checks whether the face is:
- Underexposed (too dark)
- Overexposed (too bright)
- Well-lit

Uses:
- Mean pixel intensity in the ROI
- Adaptive thresholds
- Purpope: Ensures skin analysis happens in proper lighting conditions.

3. Smoothness Detection
Estimates skin texture using:
Laplacian variance (measures sharpness)
Higher variance → more visible texture
Lower variance → smoother skin appearance
Used to approximate:
Presence of texture
Small bumps
Blurriness (if camera is out of focus)

4. Uneven Skin Tone Detection
Detects unevenness by:
Converting the face region to LAB color space
Analyzing the “L” (lightness) and “A/B” (color) channels
Checking for abrupt changes in color values
Useful for:
Hyperpigmentation
Dark spots
Redness
Discoloration

5. ROI Extraction (Region of Interest)
Uses facial landmarks to extract:
Forehead
Cheeks
Chin
These regions are used for:
Tone comparison
Texture analysis
Lighting adjustments
Landmarks come from:
A lightweight face detection model
Dlib 68 landmarks or your custom solution

6. Camera + Face Landmarks Code
This feature includes:
- Real-time video capture using OpenCV
- Face detection
- Landmark extraction
- Cropping and analyzing facial ROI
- Overlaying results on live video feed

Files:
- `skin_analysis.py`
- `ROI_extraction.py`
- `brightness_check.py`
- `uneven_tone_detection.py`
