#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:36:42 2025

@author: noob
"""

import os
import cv2

# Input and output directories
input_dir = "/home/noob/koty/new_before_last/work/data/cropped_outputs/exact_shape_only/"
output_dir = "/home/noob/koty/new_before_last/work/data/model_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files (adjust extensions if needed)
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

for filename in image_files:
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read {filename}. Skipping.")
        continue

    h, w = image.shape[:2]

    # Calculate crop margins
    top = int(h * 0.40)
    bottom = h - int(h * 0.20)
    left = int(w * 0.30)
    right = w - int(w * 0.20)

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Save the cropped image
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cropped_image)

    print(f"Cropped and saved: {output_path}")
