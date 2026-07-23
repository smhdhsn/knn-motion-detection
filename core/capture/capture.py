import cv2
import numpy as np


class Capture:
    def __init__(self, url: str) -> None:
        self._url: str = url
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> cv2.VideoCapture:
        self._cap = cv2.VideoCapture(self._url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not begin capturing on {self._url}")

        return self._cap

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()

        cv2.destroyAllWindows()


def get_frame_details(cap: cv2.VideoCapture) -> tuple[tuple[int, int, int], np.dtype]:
    has_frame, frame = cap.read()
    if not has_frame:
        raise RuntimeError("Could not read frame")

    return frame.shape, frame.dtype
