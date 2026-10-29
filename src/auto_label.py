"""Use Qwen-VL to label the shape of each crop. Writes crops/labels.csv."""
import os
import sys

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from config import SHAPES

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
crop_dir = sys.argv[1]

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")

prompt = f"What is the shape of the object? Answer with one word from: {', '.join(SHAPES)}."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)

rows = []
for fname in sorted(os.listdir(crop_dir)):
    if not fname.endswith(".png"):
        continue
    img = Image.open(os.path.join(crop_dir, fname)).convert("RGB")
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=5)
    answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    answer = answer.strip().lower()
    shape = next((s for s in SHAPES if s in answer), None)
    if shape:
        rows.append({"file": fname, "shape": shape})

pd.DataFrame(rows).to_csv(os.path.join(crop_dir, "labels.csv"), index=False)
