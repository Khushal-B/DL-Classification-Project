"""Train a ResNet-18 to predict the shape of a single-object crop."""
import os
import sys

import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from config import COLORS, SHAPES, SHAPE_TO_ID


class CropDataset(Dataset):
    def __init__(self, crop_dir):
        df = pd.read_csv(os.path.join(crop_dir, "labels.csv"))
        self.crop_dir = crop_dir
        self.files = df["file"].tolist()
        self.labels = [SHAPE_TO_ID[s] for s in df["shape"]]
        # slot = which of the 4 color heads this crop belongs to (from file name suffix)
        self.slots = [COLORS.index(f[:-4].split("_")[-1]) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.crop_dir, self.files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x, self.slots[i], self.labels[i]


class MultiHeadResNet(nn.Module):
    """Shared ResNet-18 backbone with one shape head per object slot (color)."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(feat_dim, len(SHAPES)) for _ in COLORS])

    def forward(self, x, slots):
        feats = self.backbone(x)
        # logits from every head: (batch, num_heads, num_shapes)
        all_logits = torch.stack([head(feats) for head in self.heads], dim=1)
        # keep only the head that matches each crop's slot
        return all_logits[torch.arange(len(x)), slots]


def build_model():
    return MultiHeadResNet()


if __name__ == "__main__":
    crop_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(CropDataset(crop_dir), batch_size=128, shuffle=True)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        correct = total = 0
        for x, slot, y in loader:
            x, slot, y = x.to(device), slot.to(device), y.to(device)
            out = model(x, slot)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"epoch {epoch + 1}: acc {correct / total:.3f}")

    torch.save(model.state_dict(), "shape_model.pth")
