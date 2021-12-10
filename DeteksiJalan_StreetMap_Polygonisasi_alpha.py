# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:56:51 2021

@author: admin
"""

import cv2
import numpy as np

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString

#%% Helper funcs

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)

def ConvertToList(points):
    
    listPoints = []
    
    for i in range(points.shape[0]):
        listPoints.append([points[i,0,0], 640 - points[i,0,1]])
        
    return listPoints

def Sort_Tuple(tup, axis): 
  
    return(sorted(tup, key = lambda x: x[axis]))  

def Sort_Tuple_squared(tup): 
  
    return(sorted(tup, key = lambda x: (x[0]**2 + x[1]**2)))  

def remove_closePoints(listnya, thresh, imgsize, axis):
    
    bins = []
    
    #make empty bins
    
    
    for i in range(int(np.floor(imgsize/thresh))):
        
        bin_thresh = []
        
        for elem in listnya:
            
            #print(elem)
            
            if (elem[axis] > i*thresh) and (elem[axis] < (i+1)*thresh):
                
                bin_thresh.append(elem)
                
    
        bins.append(bin_thresh)
        
    #return first elem of each bin
    
    bin_ret = []
    
    for elem in bins:
        
        if len(elem)>0:
        
            bin_ret.append(elem[0])
        
    return bin_ret


def draw_triangle(img_in, pt1, pt2, pt3):

    
    if True :
        
        cv2.line(img_in, pt1, pt2, [0,0,255], 5)
        cv2.line(img_in, pt2, pt3, [0,0,255], 5)
        cv2.line(img_in, pt3, pt1, [0,0,255], 5)

        

def draw_list_triangles(imgin, list_tri):
    
    for i in range(len(list_tri)):
        tri = list_tri[i]
        
        pt1 = tri[0][0]
        
        pt2 = tri[1][0]
        
        pt3 = tri[2][0]
        
        draw_triangle(imgin, pt1, pt2, pt3)
        
    return imgin

def draw_line(img_in, pt1, pt2):

    
    if True :
        
        #print(pt1)
        #print(pt2)
        
        cv2.line(img_in, (int(pt2[0]), 640-int(pt2[1])), (int(pt1[0]), 640-int(pt1[1])), [0,0,255], 5)
        
def draw_list_voronoi(imgin, list_lines):
    
    for line in list_lines:
        
        #print(line)
        
        pt1 = line[0].tolist()
        
        pt2 = line[1].tolist()
        
        if (pt1[0]< 640 and pt1[0]>0) and (pt1[1]< 640 and pt1[1]>0) and (pt2[0]< 640 and pt2[0]>0) and (pt2[1]< 640 and pt2[1]>0):

            draw_line(imgin, pt1, pt2)
        
    return imgin        
    
def draw_list_result(imgin, list_lines):
    
    for i in range(len(list_lines)-1):
        
        #print(line)
        
        pt1 = (list_lines[i].x, list_lines[i].y)
        
        pt2 = (list_lines[i+1].x, list_lines[i+1].y)
        
        if (pt1[0]< 640 and pt1[0]>0) and (pt1[1]< 640 and pt1[1]>0) and (pt2[0]< 640 and pt2[0]>0) and (pt2[1]< 640 and pt2[1]>0):

            draw_line(imgin, pt1, pt2)
        
    return imgin    
    

#%%

img = cv2.imread('./Sample Images/ODP-XXX-XXX_XXX XXX_XXX_XXX.XX_lon_106.840475211792_lat_-6.15401897757296_h_23_w_23_map_GEarth_deteksiJalan.bmp')

# create a mask image

# Mat mask = orig == cv::Vec3b(255,0,0) #no need? image already masked

# edge detection to detect edges

edges = cv2.Canny(img, 5, 5, apertureSize=3)

cv2.imwrite('edges_deteksiJalan_polygonisasi1.jpg', edges)


imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgc1 = img.copy()

imgc2 = img.copy()


# 2.3. find countours dan approx polygon

ret3,thresh3 = cv2.threshold(imgray,0,255,\
           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
cv2.imwrite('grayscale_deteksiJalan_polygonisasi1.jpg', img)

contours1, _= cv2.findContours(thresh3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

polys = []
for cont in contours1:
    approx_curve = cv2.approxPolyDP(cont, 3, False)
    polys.append(approx_curve)
cv2.drawContours(imgc1, polys, -1, (0, 255, 0), thickness=5, lineType=8)

'''
cnt = contours[4]
cv2.drawContours(countourImg, [cnt], 0, (0,255,0), 3)
'''


cv2.imwrite('contours_deteksiJalan_polygonisasi1.jpg', imgc1)


# 4. Filter Counturs



polys_filtered = []
for cont in contours1:
    
    areaContour = cv2.contourArea(cont)
    
    if areaContour > 2500:
    
        approx_curve = cv2.approxPolyDP(cont, 3, False)
        
        polys_filtered.append(approx_curve)
        
cv2.drawContours(imgc2, polys_filtered, -1, (0, 255, 0), thickness=5, lineType=8)

cv2.imwrite('contoursFiltered_deteksiJalan_polygonisasi1.jpg', imgc2)


#%% Sekeletonisasi

for points in polys_filtered:
    
    points_list = np.array(ConvertToList(points))

    tri = Delaunay(points_list)
    
    plt.triplot(points_list[:,0], points_list[:,1], tri.simplices.copy())
    plt.plot(points_list[:,0], points_list[:,1], '.')
    
    plt.savefig("delauney_deteksiJalan_polygonisasi1.jpg")
    
    plt.close()
    plt.clf()
    
    # draw on original image delauney lines
    
    img_delauney = img.copy()
    
    
    img_delauney = draw_list_triangles(img_delauney.copy(), points[tri.simplices].tolist())
    
    cv2.imwrite('delauneyImage_deteksiJalan_polygonisasi1.jpg', img_delauney)
    
    # 2. Mencari Edges Voronoi yang ada didalam polygon
    
    p = tri.points[tri.vertices]

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T
    
    a = A - C
    b = B - C
    
    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C
    
    # Grab the Voronoi edges
    vc = cc[:,tri.neighbors]
    vc[:,tri.neighbors == -1] = np.nan # edges at infinity, plotting those would need more work...
    
    lines_raw = []
    lines_raw.extend(zip(cc.T, vc[:,:,0].T))
    lines_raw.extend(zip(cc.T, vc[:,:,1].T))
    lines_raw.extend(zip(cc.T, vc[:,:,2].T))
    
    lines = LineCollection(lines_raw, edgecolor='k')
    
    plt.hold(1)
    plt.plot(points_list[:,0], points_list[:,1], '.')
    plt.plot(cc[0], cc[1], '*')
    plt.gca().add_collection(lines)
    plt.axis('equal')
    plt.xlim(0, 640)
    plt.ylim(0, 640)

    plt.savefig("voronoi_deteksiJalan_polygonisasi1.jpg")
    
    plt.close()
    plt.clf()
    
    # draw edges voronoi on image
    
    img_voronoi = img.copy()
    
    img_voronoi = draw_list_voronoi(img_voronoi.copy(), lines_raw)
    
    cv2.imwrite('voronoiImage_deteksiJalan_polygonisasi1.jpg', img_voronoi)
    
    
    vc_list = []

    for i in range(cc.shape[1]):
        
        point = (cc[0,i], cc[1,i])
        
        vc_list.append(point)
        
    # sort vc_list y- axis
    
    #vc_list = Sort_Tuple(vc_list, 1)
    
    # remove all closely related points on y - axis
    
    vc_list = remove_closePoints(vc_list, 15, 640, 1)
    
    # sort vc_list x- axis
    
    #vc_list = Sort_Tuple(vc_list, 0)
    
    # remove all closely related points on x - axis
    
    vc_list = remove_closePoints(vc_list, 15, 640, 0)
    
    #convert elem sorted vc_list into Point
    
    vc_list = Sort_Tuple_squared(vc_list)
    
    vc_list = [Point(elem) for elem in vc_list]
        
    
    vc_within = []
        
    polygon = Polygon(ConvertToList(points))
    
    for point in vc_list:
        
        if polygon.contains(point):
            
            vc_within.append(point)
            
            
    
    vc_line = LineString(vc_within)
    
    
    #vc_line_sorted = simplify_LineString(vc_line)
    
    plt.plot(*vc_line.xy)

    plt.savefig("resLine_deteksiJalan_polygonisasi1.jpg")
    
    plt.close()
    plt.clf()
    
    #Draw result on original image
    
    img_result = img.copy()
    
    img_result = draw_list_result(img_result.copy(), vc_within)
    
    cv2.imwrite('resultImage_deteksiJalan_polygonisasi1.jpg', img_result)
    
    
        
        
    
    

    
    


    
    
    
    
    
