import cv2
import numpy as np


def multi_show(
    name: str,
    img1: cv2.typing.MatLike,
    img2: cv2.typing.MatLike,
    *,
    padding: int = 5,
    sep_color=(0, 255, 255),
):
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h = max(img1.shape[0], img2.shape[0])

    img1 = np.pad(img1, ((0, h - img1.shape[0]), (0, 0), (0, 0)))
    img2 = np.pad(img2, ((0, h - img2.shape[0]), (0, 0), (0, 0)))

    sep = np.full((h, padding, 3), sep_color, np.uint8)
    cv2.imshow(name, cv2.hconcat([img1, sep, img2]))
