#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:18:16 2025

@author: noob
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class AnomalyFixer():

    def detect_anomaly_boundaries(self ,image, threshold=None):
        # Edge-based anomaly detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        if threshold is None:
            threshold = np.mean(gradient_magnitude) + 2 * np.std(gradient_magnitude)

        mask = gradient_magnitude > threshold

        return mask.astype(np.uint8)

    def refine_mask(self ,mask, min_area=50):
        # Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        refined_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_mask[labels == i] = 1

        # Morphological operations for smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

        return refined_mask

    def process_image_anomalies(self ,image_path=None, image_array=None):
        # Load image
        if image_array is not None:
            image = image_array
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Could not load image")

        print(f"Processing image of shape: {image.shape}")

        # Try different detection methods
        methods = ['gradient']
        results = {}

        for method in methods:
            print(f"Detecting anomalies using {method} method...")
            mask = self.detect_anomaly_boundaries(image)
            refined_mask = self.refine_mask(mask)

            # Try different replacement methods
            replaced_mean = self.replace_with_surroundings(image, refined_mask, method='mean')

            results[method] = {
                'mask': refined_mask,
                'replaced_mean': replaced_mean,
            }

        return image, results

    def visualize_results(self ,original, results):
        """
        Visualize original image, detected anomalies, and corrected results
        """
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods, 3, figsize=(20, 4*n_methods))

        if n_methods == 1:
            axes = axes.reshape(1, -1)

        for i, (method, result) in enumerate(results.items()):
            # Original
            axes[i, 0].imshow(original, cmap='gray')
            axes[i, 0].set_title(f'Original ({method})')
            axes[i, 0].axis('off')

            # Detected mask
            axes[i, 1].imshow(result['mask'], cmap='hot')
            axes[i, 1].set_title('Detected Anomalies')
            axes[i, 1].axis('off')

            # Mean replacement
            axes[i, 3].imshow(result['replaced_mean'], cmap='gray')
            axes[i, 3].set_title('Mean Replacement')
            axes[i, 3].axis('off')


        plt.tight_layout()
        plt.show()

    def process_image_anomalies_rgb(self ,image_path=None, image_array_rgb=None):

        if image_array_rgb is not None:
            image_rgb = image_array_rgb
        else:
            image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image_rgb is None:
            raise ValueError("Could not load image")

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        mask = self.detect_anomaly_boundaries(image_gray)
        refined_mask = self.refine_mask(mask)

        # Replace anomalous regions in RGB image by surrounding RGB mean
        result_rgb = image_rgb.copy()

        # Find contours of anomalies
        contours, _ = cv2.findContours(refined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            expanded_mask = np.zeros_like(refined_mask, dtype=np.uint8)
            cv2.drawContours(expanded_mask, [contour], -1, 1, thickness=-1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            surrounding_mask = cv2.dilate(expanded_mask, kernel) - expanded_mask

            for c in range(3):  # For each color channel
                surrounding_pixels = image_rgb[:, :, c][surrounding_mask == 1]
                if len(surrounding_pixels) > 0:
                    replacement_value = int(np.mean(surrounding_pixels))
                    result_rgb[:, :, c][expanded_mask == 1] = replacement_value

        return image_rgb, refined_mask, result_rgb
    
    def process_image_anomalies_rgb_obj(self ,image_rgb):
        if image_rgb is None:
            raise ValueError("Could not load image")

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        mask = self.detect_anomaly_boundaries(image_gray)
        refined_mask = self.refine_mask(mask)

        # Replace anomalous regions in RGB image by surrounding RGB mean
        result_rgb = image_rgb.copy()

        # Find contours of anomalies
        contours, _ = cv2.findContours(refined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            expanded_mask = np.zeros_like(refined_mask, dtype=np.uint8)
            cv2.drawContours(expanded_mask, [contour], -1, 1, thickness=-1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            surrounding_mask = cv2.dilate(expanded_mask, kernel) - expanded_mask

            for c in range(3):  # For each color channel
                surrounding_pixels = image_rgb[:, :, c][surrounding_mask == 1]
                if len(surrounding_pixels) > 0:
                    replacement_value = int(np.mean(surrounding_pixels))
                    result_rgb[:, :, c][expanded_mask == 1] = replacement_value

        return image_rgb, refined_mask, result_rgb


    def process_and_plot(self, image_path ,output_path):
        original_rgb, mask, replaced_rgb = self.process_image_anomalies_rgb(image_path=image_path)
        os.makedirs('./output', exist_ok=True)
        cv2.imwrite(output_path, replaced_rgb)
        print(f"Saved RGB replacement image to {output_path}")

        plt.subplot(1, 3, 1)
        plt.title('Original RGB')
        plt.imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Anomaly Mask')
        plt.imshow(mask, cmap='hot')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Mean Replacement RGB')
        plt.imshow(cv2.cvtColor(replaced_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def process_and_replace(self, image_path):
        _, _, replaced_rgb = self.process_image_anomalies_rgb(image_path=image_path)
        output_path = image_path
        cv2.imwrite(output_path, replaced_rgb)

    def process_only(self, image_path):
        _, _, replaced_rgb = self.process_image_anomalies_rgb(image_path=image_path)
        return replaced_rgb

    def process_only_obj(self, image):
        _, _, replaced_rgb = self.process_image_anomalies_rgb_obj(image)
        return replaced_rgb

    def process_all_png_in_dir(self, directory):
        for filename in os.listdir(directory):
            if filename.lower().endswith('.png'):
                image_path = os.path.join(directory, filename)
                self.process_and_replace(image_path)

