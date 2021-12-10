# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:38:52 2021

@author: admin

"""


import numpy as np
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString

from HelperFuncs_Geos import *

'''

File ini berisis fungsi fungsi untuk pengolahan geometry, shape, drawing.
'''
def dotprod(u, v):
    return u[0]*v[0] + u[1]*v[1]

def crossprod(u, v, w):
    """u x (v x w)"""
    return dotprod(u, w)*v - dotprod(u, v)*w

def ncrossprod(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dotprod(u, v)**2

def sq2(u):
    return dotprod(u, u)

def Point2Latlongpoint(point, tile_x, tile_y, tile_zoom):
    
    point_latlong = konversi_Point2LatLong(
    konversi_PixCoord2Point(
        (point.x, point.y), (tile_x, tile_y),tile_zoom
                     ))
    
    return Point(point_latlong)

def Point2LongLatpoint(point, tile_x, tile_y, tile_zoom):
    
    point_latlong = konversi_Point2LatLong(
    konversi_PixCoord2Point(
        (point.x, point.y), (tile_x, tile_y),tile_zoom
                     ))
    
    return Point(point_latlong[1], point_latlong[0])

def ConvertPointsToList(listPoints):
    
    return [(point.x, point.y) for point in listPoints ]

def Convert_listPoints2listLatlong (list_points, tile_x, tile_y, tile_zoom):
    
    list_latlong = []
    
    for point in list_points:
        
        list_latlong.append(Point2Latlongpoint(point,tile_x, tile_y, tile_zoom))
        
    return list_latlong

def Convert_listPoints2listLongLat (list_points, tile_x, tile_y, tile_zoom):
    
    list_latlong = []
    
    for point in list_points:
        
        list_latlong.append(Point2LongLatpoint(point,tile_x, tile_y, tile_zoom))
        
    return list_latlong

def Convert_listEdges2listEdgesLatlong (list_edges, tile_x, tile_y, tile_zoom):
    
    list_edges_latlong = []
    
    for edge in list_edges:
        
        p1 = Point2Latlongpoint(edge[0],tile_x, tile_y, tile_zoom)
        
        p2 = Point2Latlongpoint(edge[1],tile_x, tile_y, tile_zoom)
        
        line = LineString((p1, p2))
        
        list_edges_latlong.append(line)
        
    return list_edges_latlong

def Convert_listEdges2listEdgesLonglat (list_edges, tile_x, tile_y, tile_zoom):
    
    list_edges_latlong = []
    
    for edge in list_edges:
        
        p1 = Point2LongLatpoint(edge[0],tile_x, tile_y, tile_zoom)
        
        p2 = Point2LongLatpoint(edge[1],tile_x, tile_y, tile_zoom)
        
        line = LineString((p1, p2))
        
        list_edges_latlong.append(line)
        
    return list_edges_latlong


def draw_triangle(img_in, pt1, pt2, pt3):

    
    if True :
        
        cv2.line(img_in, (int(pt2[0]), 256-int(pt2[1])), (int(pt1[0]), 256-int(pt1[1])), [0,0,255], 1)
        cv2.line(img_in, (int(pt2[0]), 256-int(pt2[1])), (int(pt3[0]), 256-int(pt3[1])), [0,0,255], 1)
        cv2.line(img_in, (int(pt2[0]), 256-int(pt2[1])), (int(pt3[0]), 256-int(pt3[1])), [0,0,255], 1)

        

def draw_list_triangles(imgin, list_tri):
    
    for i in range(len(list_tri)):
        tri = list_tri[i]
        
        pt1 = tri[0]
        
        pt2 = tri[1]
        
        pt3 = tri[2]
        
        draw_triangle(imgin, pt1, pt2, pt3)
        
    return imgin

def draw_line(img_in, pt1, pt2):

    
    if True :
        
        #print(pt1)
        #print(pt2)
        
        cv2.line(img_in, (int(pt2[0]), 256-int(pt2[1])), (int(pt1[0]), 256-int(pt1[1])), [0,0,255], 1)
        
def draw_list_voronoi(imgin, list_lines):
    
    for line in list_lines:
        
        #print(line)
        
        pt1 = line[0].tolist()
        
        pt2 = line[1].tolist()
        
        if (pt1[0]< 256 and pt1[0]>0) and (pt1[1]< 256 and pt1[1]>0) and (pt2[0]< 256 and pt2[0]>0) and (pt2[1]< 256 and pt2[1]>0):

            draw_line(imgin, pt1, pt2)
        
    return imgin        
    
def draw_list_result(imgin, list_lines):
    
    for i in range(len(list_lines)-1):
        
        #print(line)
        
        pt1 = (list_lines[i].x, list_lines[i].y)
        
        pt2 = (list_lines[i+1].x, list_lines[i+1].y)
        
        if (pt1[0]< 256 and pt1[0]>0) and (pt1[1]< 256 and pt1[1]>0) and (pt2[0]< 256 and pt2[0]>0) and (pt2[1]< 256 and pt2[1]>0):

            draw_line(imgin, pt1, pt2)
        
    return imgin    

def draw_point(imgin, pt):
    
    cv2.circle(imgin, (int(pt[0]), 256-int(pt[1])), 2, [255,0,0], 1)
    
def draw_points_list(imgin, list_pt):
    
    for i in range(len(list_pt)):
        
        pt = [int(list_pt[i].x), int(list_pt[i].y)]
                   
        draw_point(imgin, pt)
        
    return imgin

def linestring_to_points(feature,line):
    return {feature:line.coords}

def getDistPoints(p1, p2):
    
    return np.sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def draw_list_edges(imgin, list_edges):
    
    for edge in list_edges:
        
        pt1 = (edge[0].x, edge[0].y)
        
        pt2 = (edge[1].x, edge[1].y)
        
        draw_line(imgin, pt1, pt2)
        
    return imgin