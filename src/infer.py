"""Detect-then-reason inference: segment, predict 4 shapes, apply the rule."""
import os
import sys

import cv2
import torch

from config import SHAPES
from segment import segment_image
from train import build_model


def rule(shapes):
    # Discovered rule: label is 1 when both a sphere and a cube appear.
    return int("sphere" in shapes and "cube" in shapes)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load("shape_model.pth", map_location=device))
    model.eval()

    print("file,label")
    for fname in sorted(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in segment_image(img)]
        x = torch.stack([torch.from_numpy(c).permute(2, 0, 1).float() / 255.0 for c in crops]).to(device)
        with torch.no_grad():
            preds = model(x).argmax(1).tolist()
        shapes = [SHAPES[p] for p in preds]
        print(f"{fname},{rule(shapes)}")
