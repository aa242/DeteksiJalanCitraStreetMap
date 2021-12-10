# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:24:21 2021

@author: admin
"""

import numpy as np

'''

File ini berisis fungsi fungsi untuk pengolahan koordinat tiles image, mercator 
projection, tiles coordinates dan perhitungan longitude dan latitude.
'''

PI = 3.1415926535

def konversi_LatLong2Point(latlong):
    
    sin_y = np.min[np.max[np.sin(latlong[0] * (PI/180)) , -0.9999], 0.9999]
    
    x = 128 + latlong[1] * (256/360)
    
    y = 128 + 0.5 * np.log((1 + sin_y) / (1 - sin_y)) * (-1)*(256/(2*PI))
    
    return (x, y)

def konversi_Point2LatLong(point):
    
    lat = (2*np.arctan(np.exp((point[1] - 128) / (-1*(256/ (2*PI))))) - 
           PI/2)/ (PI/180)
    
    long = (point[0] - 128) / (256/360)
    
    return (lat, long)

def konversi_LatLong2Tile(latlong, zoom):
    
    t = 2**zoom
    s = 256/t
    
    p = konversi_LatLong2Point(latlong)
    
    x = np.floor(p[0]/s)
    
    y = np.floor(p[1]/s)
    
    return (x, y)

def normal_Tile(tile_point, zoom):
    
    t = 2**zoom
    
    x = ((tile_point[0]%t) + t)%t;
    
    y = ((tile_point[1]%t) + t)%t;
    
    return (x,y)
    

def konversi_Tile2Bounds (tile_point, zoom):
    
    tile_point = normal_Tile(tile_point)
    
    t = 2**zoom
    
    s = 256/t
    
    x_west = tile_point[0]*s
    
    x_east = tile_point[0]*s + s
    
    y_south = tile_point[1]*s + s
    
    y_north = tile_point[1]*s
    
    latlong_NE = konversi_Point2LatLong((x_east, y_north))
    
    latlong_SW = konversi_Point2LatLong((x_west, y_south))
    
    return (latlong_NE, latlong_SW)

def tileSize (zoom):
    
    return 256/(2**zoom)

def konversi_PixCoord2Point(coord, tile_point, zoom):
    
    x = coord[0]/256*tileSize(zoom) + tile_point[0]*tileSize(zoom)
    
    y = (256 - coord[1])/256*tileSize(zoom) + tile_point[1]*tileSize(zoom)
    
    return (x, y)