"""
Practice: Real Time Face Detection
Margaret Gonzalez - Full Sail University

Detects faces in live webcam video with OpenCV's Haar cascade classifier
(Viola-Jones method) and draws a green box around each face.

Controls (click the video window first):
    q  - quit
    s  - save a screenshot of the current frame (evidence)
"""
import os
import time

import cv2

# ---------------------------------------------------------------------------
# 1. Initialize the classifier
#    cv2.data.haarcascades is the folder that ships with opencv-python.
#    The course version (os.path.dirname(cv2.__file__) + "/data/...") is kept
#    as a fallback for older OpenCV installs.
# ---------------------------------------------------------------------------
if hasattr(cv2, "data"):
    cascPath = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
else:
    cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
if faceCascade.empty():
    raise SystemExit(f"Could not load the face classifier from: {cascPath}\n"
                     f"You have OpenCV {cv2.__version__}. OpenCV 5 no longer includes the Haar "
                     "cascade files.\nFix: pip uninstall -y opencv-python  then  "
                     'pip install "opencv-python<5"')

# ---------------------------------------------------------------------------
# 2. Open the webcam (camera 0). On Windows, DirectShow (CAP_DSHOW) opens the
#    camera faster; if it fails, fall back to the default backend.
# ---------------------------------------------------------------------------
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
if not video_capture.isOpened():
    video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise SystemExit("No webcam found. Check that the camera is connected and allowed in "
                     "Windows Settings > Privacy & security > Camera.")

print("Face detection running. Click the video window, press 's' to save a screenshot, 'q' to quit.")

os.makedirs("screenshots", exist_ok=True)
prev_time = time.time()
fps = 0.0

# ---------------------------------------------------------------------------
# 3. Apply faceCascade on each webcam frame
# ---------------------------------------------------------------------------
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    if not ret:                       # camera disconnected or busy
        print("Could not read a frame from the webcam.")
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # shrink the image 10 percent per pyramid level
        minNeighbors=5,      # overlapping detections needed to accept a face
        minSize=(30, 30),    # ignore anything smaller than 30 x 30 pixels
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frames, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Frames per second and face count (shows the detection is real time)
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    cv2.putText(frames, f"Faces: {len(faces)}   FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frames, "s = screenshot   q = quit", (10, frames.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        name = os.path.join("screenshots", time.strftime("face_detection_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(name, frames)
        print(f"Saved {name} ({len(faces)} face(s) detected)")

# ---------------------------------------------------------------------------
# 4. Release the capture frames
# ---------------------------------------------------------------------------
video_capture.release()
cv2.destroyAllWindows()
print("Face detection stopped.")
