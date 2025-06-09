#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Jun  8 23:57:26 2025
@author: noob
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل الصورة
image = cv2.imread('/home/noob/koty/new_before_last/project-ano/model_deployment/cropped_outputs_cropped_10_percent/33_exact_shape.png')

# التأكد من تحميل الصورة
if image is None:
    raise FileNotFoundError("لم يتم العثور على الصورة. تحقق من المسار.")

# تحويل الصورة إلى رمادي
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تحديد النقاط البيضاء (قيمة سطوع عالية)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# إيجاد المؤشرات للنقاط البيضاء
points = np.column_stack(np.where(thresh == 255))

# حساب اللون المتوسط واستبدال النقاط البيضاء
if points.size > 0:
    mask_poly = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_poly, [points], 255)

    # حساب اللون المتوسط من المناطق المحيطة
    mean_color = cv2.mean(image, mask=mask_poly)[:3]

    # تحضير قناع لـ floodFill (بحجم أكبر بـ2)
    h, w = image.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # تغيير كل نقطة بيضاء إلى اللون المتوسط باستخدام floodFill
    for point in points:
        cv2.floodFill(image, flood_mask, tuple(point[::-1]), mean_color, flags=cv2.FLOODFILL_FIXED_RANGE)


# تحويل الصورة إلى RGB لعرضها بـ matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# عرض الصورة باستخدام matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.title('Modified Image')
plt.axis('off')
plt.show()



plt.figure(figsize=(10, 8))
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')
# =============================================================================
# import cv2
# import matplotlib.pyplot as plt
# 
# # Load the original image
# image = cv2.imread('/home/noob/koty/new_before_last/project-ano/predictions-stage2/enhanced/33_exact_shape_enhanced.png')
# 
# if image is None:
#     raise FileNotFoundError("Image not found. Please check the path.")
# 
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
# # Show the grayscale image
# plt.figure(figsize=(8, 6))
# plt.imshow(gray, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')
# plt.show()
# 
# 
# =============================================================================
