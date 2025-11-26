## AR Makeup Module – Smart Mirror Project

This folder contains all Augmented Reality Makeup features for the Smart Mirror system.
These features run in real-time using face landmarks, OpenCV, and custom overlay blending techniques.

### The module includes:
- Lipstick overlays
- Eyebrow overlays
- Foundation blending
- Afro Ponies (hairstyle overlay)
- PNG assets
- Real-time face landmark detection scripts

---

### Features Included
####1. Lip Overlay System
Real-time lipstick application using landmark points (mouth region).
Features:
- Automatically detects upper & lower lips
- Scales and rotates the lip PNG to match the user's lip shape
- Supports multiple lip PNGs:
- `purple_lip.png`
- `red_lip.png`
- `pink_lip.png`
- `brownlinerandgloss.png`
- `lip_image.png`
Includes alpha blending for natural results
Built to switch between multiple lipstick styles

File:
`lip_overlay.py`

---

#### 2. Eyebrow Overlay System
Applies eyebrow PNGs to both brows using 68 landmark detection.
Features:
- Rotates and mirrors PNG for left and right eyebrows
- Matches eyebrow angle and curve
- Smooth blending to look natural
- Supports different eyebrow styles

File:
`eyebrow_overlay.py`

### 3. Foundation Overlay (Skin Tone Makeup)
Simulates natural foundation using:
- Face mask extraction
- Color blending
- Feather smoothing
- Uses the cheek, chin, forehead ROI to apply foundation evenly across the face.

File:
`foundation_overlay.py`

#### 4. Afro Ponies (Hairstyle Overlay)
Adds an Afro ponytail hairstyle on top of the head, aligned to forehead and hairline landmarks.
Features:
- Tracks head rotation
- Scales with face size
- Uses transparent PNGs for realistic overlay

File:
`afro_ponies_overlay.py`
