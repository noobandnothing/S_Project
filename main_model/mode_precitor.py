#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import pickle

# Load XGBoost model from pickle file
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)


# Define your model architecture (must match training)
class TinyConvRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 16),
            nn.Mish(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


def predict_image(image_path, model_path):
    # Load model
    model = TinyConvRegressionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((20, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        pred = model(img_tensor).item()

    pred = xgb_model.predict(np.array([[round(pred, 4)]]))[0]

    print(f"Predicted concentration for '{image_path}': {pred:.6f}")
    return pred




image_path = '/home/noob/koty/new_before_last/work/data/model_data-after-fix/5_exact_shape.png'
model_path = '/home/noob/koty/new_before_last/work/main_model/final_model_full.pth'

predict_image(image_path, model_path)
