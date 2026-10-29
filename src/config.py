# Basic settings shared by all scripts.
SHAPES = ["sphere", "cube", "cylinder", "cone"]
SHAPE_TO_ID = {s: i for i, s in enumerate(SHAPES)}

# Each scene has 4 objects, each with its own color (HSV hue ranges, OpenCV scale 0-179).
COLOR_RANGES = {
    "red": [(0, 10), (170, 179)],
    "green": [(40, 80)],
    "blue": [(100, 130)],
    "yellow": [(20, 35)],
}
COLORS = list(COLOR_RANGES.keys())

CROP_SIZE = 64
