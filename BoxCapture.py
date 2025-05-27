#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single image bright object cropper
@author: noob
"""
import cv2

class BoxCapture:
    def find_biggest_box(self , image, threshold_value=100):
        """Find the biggest bright object box in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        max_area = 0
        best_box = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print(f"Found box: x={x}, y={y}, w={w}, h={h}")
            area = w * h
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)
    
        return best_box
    
    def process_image(self, image_path, shift_down=5):
        """Process a single image and save outputs in the same directory."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return
        
        # First pass
        first_box = self.find_biggest_box(image)
    
        if first_box is not None:
            x1, y1, w1, h1 = first_box
            crop1 = image[y1:y1+h1, x1:x1+w1]
    
            # Second pass
            second_box = self.find_biggest_box(crop1)
    
            if second_box is not None:
                x2, y2, w2, h2 = second_box
    
                final_x = x1 + x2
                final_y = y1 + y2
                final_w = w2
                final_h = h2
    
                final_crop = image[final_y:final_y+final_h, final_x:final_x+final_w]
                    
                # Center crop coordinates
                center_x = final_w // 2
                center_y = final_h // 2 + shift_down
    
                crop_size = 20
                crop_x1 = max(center_x - crop_size, 0)
                crop_y1 = max(center_y - crop_size, 0)
                crop_x2 = min(center_x + crop_size, final_crop.shape[1])
                crop_y2 = min(center_y + crop_size, final_crop.shape[0])
    
                selected_crop = final_crop[crop_y1:crop_y2, crop_x1:crop_x2]
    
                return selected_crop

        print("No object found for cropping.")

# # Example
# image_path = "/home/noob/koty/new/deploy/13.jpeg"
# obj  = BoxCapture()
# Tube20 = obj.process_image(image_path)
# cv2.imwrite("captured.jpeg", obj)