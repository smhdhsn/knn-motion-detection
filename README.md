# Change Detection

Spotting motion in a live video stream with classical computer vision. The app pulls a YouTube video frame by frame, isolates whatever is moving with background subtraction, and draws a red box around each changed region on top of the video as it plays. There is no machine learning here, just background modelling, morphology, and contours from OpenCV.

## How it works

The source is a YouTube URL. `yt-dlp` resolves it to a direct stream URL, which OpenCV then opens like any ordinary video, and the first frame is inspected once to size a reusable buffer for the loop.

Each frame runs through a short pipeline. A KNN background subtractor (`cv2.createBackgroundSubtractorKNN`) learns the static scene over a rolling history and returns a foreground mask of the pixels that changed. Morphological opening followed by closing cleans the mask, dropping speckle and filling small gaps. `cv2.findContours` groups the surviving pixels into regions, and every region above a minimum area gets a red bounding box drawn back onto the original frame. The app shows the annotated frame beside the foreground mask, so you can see exactly what the detector is reacting to.

## Setup

Built and tested on Python 3.14. The dependencies are NumPy, OpenCV, and yt-dlp, all pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
make run
```

This streams the video set by `YOUTUBE_URL` in `app/cv/main.py` and draws the detected motion on top. Press `q` to quit.

## Repository structure

```bash
.
├── app/
│   └── cv/
│       └── main.py     # entry point: resolves the stream, runs the capture + detection loop
├── core/
│   ├── capture/        # video-capture context manager around cv2.VideoCapture
│   ├── stream/         # resolves a YouTube URL to a direct stream URL with yt-dlp
│   ├── perceive/       # the change-detection pipeline (process)
│   └── draw/           # side-by-side display helper (multi_show)
├── Makefile
└── requirements.txt
```
