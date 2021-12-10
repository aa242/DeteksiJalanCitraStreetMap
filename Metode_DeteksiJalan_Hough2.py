# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:56:51 2021

@author: admin
"""

import cv2
import numpy as np

#%% Helper funcs

def convBool2Int(img):
    indices = img.astype(np.uint8)  #convert to an unsigned byte
    indices*=255
    
    return indices

def getBordered(image, width):
    
    bg = np.zeros(image.shape)
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    
    for contour in contours:
        
        area = cv2.contourArea(contour) 
        if area > biggest:
            
            biggest = area
            bigcontour = contour
            
    return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool) 

def doDilation(img):
    
    kernel = np.ones((5,5),np.uint8)
    
    return cv2.dilate(img.copy(), kernel, iterations=1)

def GaussianSmoothing(img):
    
    return cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

def OpeningImage(img, kernelSize):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    


#%%

img = cv2.imread('./Sample Images/ODP-XXX-XXX_XXX XXX_XXX_XXX.XX_lon_106.840475211792_lat_-6.15401897757296_h_23_w_23_map_GEarth_deteksiJalan.bmp')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# do opening image

opened = OpeningImage(img, (25,25))

edges = cv2.Canny(opened, 50, 150, apertureSize=3)

cv2.imwrite('opened_deteksiJalan_metode2.jpg', opened)

edges = doDilation(edges)

edges = GaussianSmoothing(edges)

#edges = convBool2Int(getBordered(edges, 10))
   
cv2.imwrite('edges_deteksiJalan_metode2.jpg', edges)

minLineLength = 120

maxLineGap = 10

lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, minLineLength, maxLineGap)



for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
cv2.imwrite('houghlines_deteksiJalan_metode2.jpg', img)
    
    
    
    
    
    
"""
if you use HoughLines: you get a lines array represented by rho and theta. Thus, you can group by theta, then you get all lines which have the same angle, afterwards you group those togeter which differ from their rho-Value only by a small amount
if you use HoughLinesP you would have to compute the slope yourself from the line-segment-ending points, then group those with a similar slope together and afterwards check how far the parallel lines are away from each other
"""