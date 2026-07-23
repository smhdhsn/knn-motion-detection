import cv2
import numpy as np

from core.capture import Capture, get_frame_details
from core.draw import multi_show
from core.perceive import process
from core.stream import get_stream_url

YOUTUBE_URL = "https://youtu.be/zu-I5jNHIxI"


def main():
    url = get_stream_url(YOUTUBE_URL, 1080)

    with Capture(url) as cap:
        bg_sub = cv2.createBackgroundSubtractorKNN(
            history=600,
            detectShadows=False,
        )

        shape, dtype = get_frame_details(cap)

        buf = np.empty(shape, dtype)
        while True:
            has_frame, frame = cap.read(buf)
            if not has_frame:
                break

            key = cv2.waitKey(1)
            if key in [ord("Q"), ord("q")]:
                break

            frame, processed_frame = process(frame, bg_sub)

            multi_show("name", frame, processed_frame)


if __name__ == "__main__":
    main()
