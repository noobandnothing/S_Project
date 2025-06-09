#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:24:42 2025

@author: noob
"""
import numpy as np
import cv2
from MyMen.TubeSegmentationModel import TubeSegmentationModel
from MyMen.AnomalyFixer import  AnomalyFixer
from MyMen.ModelPredictor import ModelPredictor

class AppManager():
    def __init__(self):
        model_path = 'saved_model/MainModelFiles/final_model_full.pth'
        fixer_path = 'saved_model/MainModelFiles/xgb_model.pkl'

        self.anomalyFixer = AnomalyFixer()

        self.model_predictor = ModelPredictor(model_path,fixer_path)
        self.tubeSegmentationModel = TubeSegmentationModel('saved_model/Segmentation/deployment_package.pth')


    def autoCrop(self,image):
    
        if image is None:
            print("Failed to read image")
            return
        h, w = image.shape[:2]
    
        top = int(h * 0.40)
        bottom = h - int(h * 0.20)
        left = int(w * 0.30)
        right = w - int(w * 0.20)
    
        cropped_image = image[top:bottom, left:right]
        
        return  cropped_image


    def predict_path(self, image_path):
        tube_pil = self.tubeSegmentationModel.predict_and_crop_exact_shape_obj(
            image_path,
            threshold=0.5,
            padding=0,
        )
        
        tube_np_rgb = np.array(tube_pil.convert("RGB")) 
        tube_np_bgr = cv2.cvtColor(tube_np_rgb, cv2.COLOR_RGB2BGR)
        
        cropped_area = self.autoCrop(tube_np_bgr)
        cropped_area_fixed = self.anomalyFixer.process_only_obj(cropped_area)
        
        return self.model_predictor.predict_image_from_obj(cropped_area_fixed)
    
    def predict_obj(self, image):
        tube_pil = self.tubeSegmentationModel.predict_and_crop_exact_shape_obj_pil_dirrect(
            image,
            threshold=0.5,
            padding=0,
        )
        
        tube_np_rgb = np.array(tube_pil.convert("RGB")) 
        tube_np_bgr = cv2.cvtColor(tube_np_rgb, cv2.COLOR_RGB2BGR)
        
        cropped_area = self.autoCrop(tube_np_bgr)
        cropped_area_fixed = self.anomalyFixer.process_only_obj(cropped_area)
        
        return self.model_predictor.predict_image_from_obj(cropped_area_fixed)
    

    
    
    
# obj = AppManager()
# result = obj.predict_path('/home/noob/koty/new_before_last/work/data/images-all/5.jpeg')
# result = obj.predict_path('/home/noob/koty/new_before_last/work/data/images-all/11.png')
# result = obj.predict_path('/home/noob/koty/new_before_last/work/data/images-all/80.jpeg')
