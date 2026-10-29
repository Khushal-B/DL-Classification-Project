import cv2
import numpy as np

from config import COLOR_RANGES, CROP_SIZE


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, (lo, 80, 50), (hi, 255, 255))
    return mask


def segment_image(img_bgr):
    """Cut a scene into one crop per color (so 4 single-object crops)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    crops = []
    for name, ranges in COLOR_RANGES.items():
        mask = color_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            crops.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8))
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = img_bgr[y0:y1 + 1, x0:x1 + 1]
        crops.append(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)))
    return crops
