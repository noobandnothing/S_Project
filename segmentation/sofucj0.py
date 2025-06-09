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
image = cv2.imread('/home/noob/koty/new_before_last/project-ano/predictions-stage2/enhanced/15_exact_shape_enhanced.png')

# التأكد من تحميل الصورة
if image is None:
    raise FileNotFoundError("لم يتم العثور على الصورة. تحقق من المسار.")

# نسخة من الأصل للعرض لاحقًا
original_image = image.copy()

# تحويل الصورة إلى رمادي
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تحديد النقاط البيضاء (سطوع عالٍ)
_, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
white_mask = thresh == 255

# إنشاء قناع للمناطق غير البيضاء لحساب اللون المتوسط منها
non_white_mask = (thresh != 255).astype(np.uint8)

# حساب اللون المتوسط من المناطق غير البيضاء
mean_color = cv2.mean(image, mask=non_white_mask)[:3]

# استبدال جميع النقاط البيضاء مباشرة باللون المتوسط
modified_image = image.copy()
modified_image[white_mask] = mean_color

# تحويل الصور إلى RGB للعرض بـ matplotlib
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
modified_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
modified_gray = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)  # New grayscale result

# عرض الصور الأربع: الأصلية، الرمادية، المعدلة، الرمادية المعدلة
plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(modified_rgb)
plt.title('Modified Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(modified_gray, cmap='gray')
plt.title('Modified Grayscale')
plt.axis('off')

plt.tight_layout()
plt.show()
