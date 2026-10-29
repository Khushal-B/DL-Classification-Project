"""Fit a small decision tree on ~50 hand-labeled images to find the hidden rule.

Input csv columns: sphere, cube, cylinder, cone (counts per image) and label.
"""
import sys

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv(sys.argv[1])
features = ["sphere", "cube", "cylinder", "cone"]

tree = DecisionTreeClassifier(max_depth=3).fit(df[features], df["label"])
print(export_text(tree, feature_names=features))
