#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:46:18 2025

@author: noob
"""

from deployment_script import TubeSegmentationModel
# Load model
model = TubeSegmentationModel('deployment_package.pth')

# Single image processing
result = model.predict_and_crop(
    '/home/noob/koty/new_before_last/project-ano/images/10.png', 
    'output_folder/', 
    threshold=0.5, 
    padding=0  # Extra pixels around object
)

if result['success']:
    print(f"Cropped image saved: {result['cropped_path']}")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"Cropped size: {result['cropped_size']}")

# Batch processing
results = model.batch_process(
    '/home/noob/koty/new_before_last/work/data/images-all/', 
    '/home/noob/koty/new_before_last/work/data/cropped_outputs/', 
    threshold=0.5, 
    padding=0
)