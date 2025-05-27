#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: noob
"""
import cv2
import torch
from PIL import Image
from torchvision import transforms
from TinyConvRegressionModel import TinyConvRegressionModel
from BoxCapture import BoxCapture
from torch.serialization import add_safe_globals

class ModelPredictor():
    def __init__(self):
        
        self.model = TinyConvRegressionModel()
        self.model.load_state_dict(torch.load('final_model_full.pth', map_location=torch.device('cpu')))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])


    def predict_image(self, image_path):
        obj = BoxCapture()
        Tube20 = obj.process_image(image_path)
        
        if Tube20 is None:
            print("Failed to crop image. Skipping prediction.")
            return None
    
        Tube20_rgb = cv2.cvtColor(Tube20, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(Tube20_rgb)
        image_tensor = self.transform(image).unsqueeze(0)
    
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_value = output.item()
    
        return predicted_value

# # Example
# obj = ModelPredictor()
# image_path = "/home/noob/koty/new/deploy/13.jpeg"
# predicted_conc = obj.predict_image(image_path)
# print(f"Predicted Concentration: {predicted_conc:.4f}")

# image_path = "/home/noob/koty/new/deploy/15.jpeg"
# predicted_conc = obj.predict_image(image_path)
# print(f"Predicted Concentration: {predicted_conc:.4f}")
