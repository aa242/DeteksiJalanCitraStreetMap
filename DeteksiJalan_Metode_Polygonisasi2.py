# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:56:51 2021

@author: admin
"""

import cv2
import numpy as np

#%% Helper funcs




#%%

img = cv2.imread('./Sample Images/ODP-XXX-XXX_XXX XXX_XXX_XXX.XX_lon_106.840475211792_lat_-6.15401897757296_h_23_w_23_map_GEarth_deteksiJalan.bmp')

# create a mask image

# Mat mask = orig == cv::Vec3b(255,0,0) #no need? image already masked

# 1. Bordering : melebarkan image mask

borderType = cv2.BORDER_CONSTANT

top = int(25)  # shape[0] = rows
bottom = top
left = int(25)  # shape[1] = cols
right = left


value = [0, 0 ,0]

borderedImg = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)

borderedImgGray = cv2.cvtColor(borderedImg, cv2.COLOR_BGR2GRAY)

cv2.imwrite('bordered_deteksiJalan_polygonisasi2.jpg', borderedImgGray)


# 2. Cleaning Mask: Open and Close to remove noise

kernel = np.ones((20,20),np.uint8)

foreground = cv2.morphologyEx(borderedImgGray, cv2.MORPH_OPEN, kernel)
foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('cleaned_deteksiJalan_polygonisasi2.jpg', foreground)


# 3. Masking Background: Area of non interest

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
background = cv2.dilate(foreground, kernel, iterations=3)
unknown = cv2.subtract(background, foreground)

cv2.imwrite('bgmask_deteksiJalan_polygonisasi2.jpg', background)

# 4. Running WaterShed

markers = cv2.connectedComponents(foreground)[1]
markers += 1  # Add one to all labels so that background is 1, not 0
markers[unknown==255] = 0  # mark the region of unknown with zero
markers = cv2.watershed(borderedImg, markers)

# tambahkan warna pada marker watershed

hue_markers = np.uint8(179*np.float32(markers)/np.max(markers))
blank_channel = 255*np.ones((borderedImg.shape[0], borderedImg.shape[1]), dtype=np.uint8)
marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)

cv2.imwrite('watershed_deteksiJalan_polygonisasi2.jpg', marker_img)

# 5. Get Polygon of watershed results

ret3,thresh3 = cv2.threshold(borderedImgGray,0,255,\
           cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours1, _= cv2.findContours(thresh3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

polys = []
for cont in contours1:
    approx_curve = cv2.approxPolyDP(cont, 3, False)
    polys.append(approx_curve)
cv2.drawContours(borderedImg, polys, -1, (0, 255, 0), thickness=5, lineType=8)


    
cv2.imwrite('Countours_deteksiJalan_polygonisasi2.jpg', borderedImg)
    
    
    
    
    
