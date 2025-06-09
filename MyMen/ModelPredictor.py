#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from MyMen.TinyConvRegressionModel import TinyConvRegressionModel
import pickle
import cv2

class ModelPredictor:
    def __init__(self,model_path,fixer_path):
        with open(fixer_path, 'rb') as f:
           self.xgb_model = pickle.load(f)
           self.model = TinyConvRegressionModel()
           self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
           self.model.eval()
           self.transform = transforms.Compose([
            transforms.Resize((20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
           
    def image_tensor_to_predict(self,img_tensor):
        with torch.no_grad():
            pred = self.model(img_tensor).item()
        return self.xgb_model.predict(np.array([[round(pred, 4)]]))[0]
        


    def predict_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
    
        return self.image_tensor_to_predict(img_tensor)
    
    def predict_image_from_obj(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
    
        img_tensor = self.transform(image).unsqueeze(0)
    
        return self.image_tensor_to_predict(img_tensor)
    
    

