# Real-Time Face Detection with OpenCV

A short Python program that finds faces in live webcam video with OpenCV's Haar cascade classifier, draws a labeled box around each face, and shows the face count and frames per second.

Built by Margaret Gonzalez as a practice exercise for the Advanced Artificial Intelligence course in the Computer Science Graduate Program at Full Sail University. The larger follow-up project, which adds object detection and face recognition, is in the [face-object-recognition](https://github.com/DeniseThaBeast88/face-object-recognition) repository.

## How it works

1. Each webcam frame is converted to grayscale, because the Haar cascade works on intensity values.
2. `detectMultiScale` slides the trained cascade over the image at several scales (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(30, 30)`).
3. A green rectangle and the label "Face" are drawn around each detection, with a face count and frames-per-second readout at the top.

## Skills demonstrated

Python, OpenCV video capture and drawing, Haar cascade object detection, defensive error handling (classifier and camera checks, safe stop when frames stop arriving).

## Requirements

- Windows, macOS, or Linux with a webcam
- Python 3.10 or newer
- OpenCV 4 (`opencv-python` below version 5). OpenCV 5 no longer ships the `haarcascade_frontalface_default.xml` file, so the version is pinned.

## Run it

```
pip install -r requirements.txt
python face_detection.py
```

Click the video window, then press `s` to save a screenshot to the `screenshots` folder or `q` to quit.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not load the face classifier" | `pip uninstall -y opencv-python opencv-contrib-python`, then `pip install "opencv-python<5"` |
| "No webcam found" | Close other apps using the camera and allow camera access in the operating system privacy settings |
| Window does not close with q | Click the video window first so it receives the key press |
| `ModuleNotFoundError: cv2` | Run `pip install -r requirements.txt` in the same terminal you run the program from |
