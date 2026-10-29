"""Segment every train image into 4 crops and save them to disk."""
import os
import sys

import cv2

from config import COLORS
from segment import segment_image

img_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

for fname in sorted(os.listdir(img_dir)):
    img = cv2.imread(os.path.join(img_dir, fname))
    if img is None:
        continue
    stem = os.path.splitext(fname)[0]
    for color, crop in zip(COLORS, segment_image(img)):
        cv2.imwrite(os.path.join(out_dir, f"{stem}_{color}.png"), crop)
