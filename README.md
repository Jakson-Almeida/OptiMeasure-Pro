# OptiMeasure Pro

OptiMeasure Pro is a research tool for real-time optical fiber thickness measurement from microscope or webcam video. It uses HSV-based segmentation to detect the fiber, fits a bounding polygon, and displays a measurement cursor with thickness in micrometers (µm) and percent variation relative to a user-defined reference—useful for optical fiber sensing and quality control.

## Features

- **Live video** from webcam (or fallback to `video.mp4`) with play/pause
- **HSV segmentation** with tunable sliders (H/S/V min/max) and area filter to isolate the fiber
- **Automatic fiber detection** via contour analysis and 4-sided bounding polygon
- **Measurement cursor** — hover over the fiber to see a cross-fiber line with:
  - **Fiber size** in µm (scaled by configurable reference diameter, default 62.5 µm)
  - **Percent** of a user-set reference (set reference by left-click or Space when paused)
- **Fiber size input** — click the size box to type a known fiber diameter (e.g. 6250 for 62.50 µm)
- **View modes**: normal, mask-only, polygon overlay, contours, rotated frame, “hawk eye” (aligned fiber view)
- **Save frames** to `./data/images/` with timestamped filenames
- **Packaging** via PyInstaller for a standalone Windows executable

## Table of Contents

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Building the Executable](#building-the-executable)

## Requirements

- Python 3.x
- **opencv-python** (cv2)
- **numpy**
- **pygame** (for camera enumeration; main capture uses OpenCV)
- **shapely** (for line–polygon intersection in measurement)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-org>/OptiMeasure-Pro.git
   cd OptiMeasure-Pro
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python numpy pygame shapely
   ```

4. **Create the images output folder** (optional; used when saving frames)
   ```bash
   mkdir data\images
   ```

5. **Run the main application**
   ```bash
   python loop_vision.py
   ```

If no webcam is available, the app falls back to `video.mp4`. You can record a sample with:

```bash
python webcam_recorder.py
```

Press `q` to stop recording; this creates `video.mp4` in the project folder.

## Usage

1. **Start** `loop_vision.py`. The main window **OptiMeasure Pro** and the **HSV Controls** window will open.
2. **Tune HSV** and **Filter** (area threshold) in **HSV Controls** until the fiber is clearly isolated in the mask (use view keys `1` / `2` to switch between normal and mask view).
3. **Position the mouse** over the detected fiber. When the cursor is inside the bounding polygon, the measurement line and readings (fiber size in µm and percent) appear.
4. **Set reference**: Pause the video (click the play/pause control or press **Space**), then **left-click** on the fiber or press **Space** again to set the reference thickness. The percent value is relative to this reference.
5. **Set fiber diameter**: Move the mouse over the fiber size box (bottom-left when measuring), click to focus, type the known diameter (e.g. `6250` for 62.50 µm), press **Enter** to apply.
6. **Save a frame**: Press **S** to save the current view to `./data/images/` with a timestamp.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit |
| `Space` | Play / Pause; when paused, also sets measurement reference |
| `S` | Save current frame to `./data/images/` |
| `P` | Toggle polygon overlay |
| `C` | Toggle contours |
| `1` | Normal view |
| `2` | Mask-only view (and disable polygon) |
| `H` | Toggle “hawk eye” (aligned fiber) window |
| `R` | Toggle frame rotation (angle follows mouse X) |
| `F` | Toggle filtered mask (by area threshold) |
| `T` | Toggle fiber size text input focus |

## Configuration

- **Camera**: The app tries video indices 0–3 with `CAP_DSHOW` (Windows). If none work, it uses `video.mp4`.
- **HSV and Filter**: Adjusted in the **HSV Controls** window; no config file. Persist by noting values or extending the app to save/load them.
- **Default fiber size**: 62.50 µm, editable via the on-screen text input (see [Usage](#usage)).
- **Saved images**: Written to `./data/images/` in the format `YYYY-MM-DD_HH-MM-SS.jpg`. Create the folder if it does not exist.

## Project Structure

| File | Description |
|------|-------------|
| `loop_vision.py` | Main application: capture, HSV segmentation, fiber detection, measurement cursor, UI. |
| `classes_lib.py` | Geometry (bounding polygon, line–polygon intersection, Bresenham line), `Cursor`, `PlayPause`, `TextInput`. |
| `vision.py` | Static image demo: load `./data/img.jpg`, HSV mask, draw contours. |
| `live_vision.py` | Simple live webcam + HSV contour demo. |
| `webcam_recorder.py` | Record webcam to `video.mp4` for use when no camera is available. |
| `loop_vision.spec` | PyInstaller spec to build the standalone executable. |
| `opticalFlow.py`, `plot_arrow.py`, `line_insect_polygon.py`, `pygame_text.py`, `rotated_text.py` | Helper/experimental scripts. |

## Building the Executable

To build a standalone Windows executable with PyInstaller:

```bash
pip install pyinstaller
pyinstaller loop_vision.spec
```

The executable is produced in `dist/` (name from the spec, e.g. `loop_vision.exe`). The spec references `micro.ico` for the icon; ensure it exists or adjust the spec.