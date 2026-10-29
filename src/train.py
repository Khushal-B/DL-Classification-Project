"""Train a ResNet-18 to predict the shape of a single-object crop."""
import os
import sys

import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from config import SHAPES, SHAPE_TO_ID


class CropDataset(Dataset):
    def __init__(self, crop_dir):
        df = pd.read_csv(os.path.join(crop_dir, "labels.csv"))
        self.crop_dir = crop_dir
        self.files = df["file"].tolist()
        self.labels = [SHAPE_TO_ID[s] for s in df["shape"]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.crop_dir, self.files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x, self.labels[i]


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(SHAPES))
    return model


if __name__ == "__main__":
    crop_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(CropDataset(crop_dir), batch_size=128, shuffle=True)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"epoch {epoch + 1}: acc {correct / total:.3f}")

    torch.save(model.state_dict(), "shape_model.pth")
