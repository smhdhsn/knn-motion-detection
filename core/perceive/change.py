import cv2


def process(
    frame: cv2.typing.MatLike,
    bg_sub: cv2.BackgroundSubtractorKNN,
    *,
    min_area: float = 500,
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    fg_mask = bg_sub.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, fg_mask
