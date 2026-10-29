# Hidden-Rule Image Classification - Detect Objects, Then Reason

**Python, PyTorch, OpenCV, ResNet-18, Qwen-VL, scikit-learn, Decision Trees** | March 2026

I cracked a "simple" 0/1 image classification task on 50K train and 10K test Blender scenes by showing it was secretly rule-based. A decision tree on 50 hand-labeled images revealed that the label depends on whether a sphere and a cube both appear, while end-to-end CNNs stalled at ~58% accuracy.

I built an OpenCV pipeline that uses color to cut every scene into 4 single-object crops, auto-labeled each crop's shape with Qwen-VL, and trained a multi-head ResNet-18 classifier reaching 98% accuracy on individual objects.

I then designed a detect-then-reason inference pipeline: segment each test image, predict all 4 shapes, then apply the discovered rule. This reaches 97% test accuracy (+39% over the CNN baseline) with fully explainable predictions.

## Pipeline

1. `src/rule_discovery.py` - fit a decision tree on ~50 hand-labeled images to find the hidden rule.
2. `src/make_crops.py` - use color segmentation (OpenCV) to cut each scene into 4 single-object crops.
3. `src/auto_label.py` - label each crop's shape with Qwen-VL.
4. `src/train.py` - train a multi-head ResNet-18 (shared backbone, one shape head per object slot) on the crops.
5. `src/infer.py` - segment test images, predict the 4 shapes, and apply the rule.

## Usage

```bash
pip install -r requirements.txt
python src/make_crops.py data/train crops
python src/auto_label.py crops
python src/train.py crops
python src/infer.py data/test > predictions.csv
```

This is a basic first version of the project.
