#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:18:16 2025

@author: noob
"""

import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def detect_anomaly_boundaries(image, method='gradient', threshold=None):
    """
    Detect anomaly boundaries in grayscale image using various methods
    
    Parameters:
    image: numpy array - grayscale image
    method: str - detection method ('gradient', 'statistical', 'clustering')
    threshold: float - threshold for anomaly detection
    
    Returns:
    mask: binary mask of anomaly regions
    """
    
    if method == 'gradient':
        # Edge-based anomaly detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if threshold is None:
            threshold = np.mean(gradient_magnitude) + 2 * np.std(gradient_magnitude)
        
        mask = gradient_magnitude > threshold
        
    elif method == 'statistical':
        # Statistical anomaly detection based on local variance
        kernel = np.ones((5,5))
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel/25)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean)**2, -1, kernel/25)
        
        if threshold is None:
            threshold = np.mean(local_var) + 3 * np.std(local_var)
        
        mask = local_var > threshold
        
    elif method == 'clustering':
        # Clustering-based anomaly detection
        h, w = image.shape
        features = []
        
        # Create feature vector for each pixel (intensity + spatial info)
        for i in range(h):
            for j in range(w):
                # Local patch statistics
                patch = image[max(0,i-2):min(h,i+3), max(0,j-2):min(w,j+3)]
                features.append([
                    image[i,j],  # pixel intensity
                    np.mean(patch),  # local mean
                    np.std(patch),   # local std
                    i/h, j/w  # normalized position
                ])
        
        features = np.array(features)
        
        # K-means clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Find anomalous cluster (smallest cluster)
        unique, counts = np.unique(labels, return_counts=True)
        anomaly_cluster = unique[np.argmin(counts)]
        
        mask = (labels == anomaly_cluster).reshape(h, w)
    
    return mask.astype(np.uint8)

def refine_mask(mask, min_area=50):
    """
    Refine anomaly mask by removing small regions and smoothing boundaries
    """
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

def inpaint_anomalies(image, mask, method='telea'):
    """
    Replace anomalous regions using inpainting
    
    Parameters:
    image: grayscale image
    mask: binary mask of regions to replace
    method: 'telea' or 'ns' (Navier-Stokes)
    """
    if method == 'telea':
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    
    return inpainted

def replace_with_surroundings(image, mask, method='mean'):
    """
    Replace anomalous regions with content from surrounding areas
    """
    result = image.copy()
    
    # Find anomaly regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Create expanded mask for surrounding area
        expanded_mask = np.zeros_like(mask)
        cv2.drawContours(expanded_mask, [contour], -1, 1, thickness=-1)
        
        # Dilate to get surrounding region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        surrounding_mask = cv2.dilate(expanded_mask, kernel) - expanded_mask
        
        if method == 'mean':
            # Replace with mean of surrounding pixels
            surrounding_pixels = image[surrounding_mask == 1]
            if len(surrounding_pixels) > 0:
                replacement_value = np.mean(surrounding_pixels)
                result[expanded_mask == 1] = replacement_value
                
        elif method == 'texture':
            # Replace with texture synthesis from surrounding area
            surrounding_region = image * surrounding_mask
            # Simple texture replacement - copy random patches from surroundings
            anomaly_coords = np.where(expanded_mask == 1)
            surrounding_coords = np.where(surrounding_mask == 1)
            
            if len(surrounding_coords[0]) > 0:
                for i, j in zip(anomaly_coords[0], anomaly_coords[1]):
                    # Find nearest surrounding pixel
                    distances = np.sqrt((surrounding_coords[0] - i)**2 + (surrounding_coords[1] - j)**2)
                    nearest_idx = np.argmin(distances)
                    nearest_i, nearest_j = surrounding_coords[0][nearest_idx], surrounding_coords[1][nearest_idx]
                    result[i, j] = image[nearest_i, nearest_j]
    
    return result

def process_image_anomalies(image_path=None, image_array=None):
    """
    Complete pipeline for anomaly detection and replacement
    """
    # Load image
    if image_array is not None:
        image = image_array
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Could not load image")
    
    print(f"Processing image of shape: {image.shape}")
    
    # Try different detection methods
    methods = ['gradient', 'statistical', 'clustering']
    results = {}
    
    for method in methods:
        print(f"Detecting anomalies using {method} method...")
        mask = detect_anomaly_boundaries(image, method=method)
        refined_mask = refine_mask(mask)
        
        # Try different replacement methods
        inpainted = inpaint_anomalies(image, refined_mask * 255)
        replaced_mean = replace_with_surroundings(image, refined_mask, method='mean')
        replaced_texture = replace_with_surroundings(image, refined_mask, method='texture')
        
        results[method] = {
            'mask': refined_mask,
            'inpainted': inpainted,
            'replaced_mean': replaced_mean,
            'replaced_texture': replaced_texture
        }
    
    return image, results

def visualize_results(original, results):
    """
    Visualize original image, detected anomalies, and corrected results
    """
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 5, figsize=(20, 4*n_methods))
    
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
        
        # Inpainted
        axes[i, 2].imshow(result['inpainted'], cmap='gray')
        axes[i, 2].set_title('Inpainted')
        axes[i, 2].axis('off')
        
        # Mean replacement
        axes[i, 3].imshow(result['replaced_mean'], cmap='gray')
        axes[i, 3].set_title('Mean Replacement')
        axes[i, 3].axis('off')
        
        # Texture replacement
        axes[i, 4].imshow(result['replaced_texture'], cmap='gray')
        axes[i, 4].set_title('Texture Replacement')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # If you have the image as numpy array:
    # image, results = process_image_anomalies(image_array=your_grayscale_array)
    
    # If you have image file:
    image, results = process_image_anomalies('/home/noob/koty/new_before_last/project-ano/model_deployment/cropped_outputs_cropped_10_percent/15_exact_shape.png')
    
    # Visualize results
    visualize_results(image, results)
    
    # To get the best result for your specific case:
    # best_result = results['gradient']['inpainted']  # or choose based on visual inspection
    
    pass