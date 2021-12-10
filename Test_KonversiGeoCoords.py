# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:22:50 2021

@author: admin
"""


import numpy as np

#%% Geos Conversion functions

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
    
    y = coord[1]/256*tileSize(zoom) + tile_point[1]*tileSize(zoom)
    
    return (x, y)
    


#%% Functions for Conversion


#the test point

test_point = (64, 192)


filename = '52247_33932_16.png'

geos_info = filename[:-4].split('_')

tile_x = float(geos_info[0])

tile_y = float(geos_info[1])

tile_zoom = float(geos_info[2])

image_size = 256.0


# convert the test point

test_latlong = konversi_Point2LatLong(
    konversi_PixCoord2Point(
        test_point, (tile_x, tile_y),tile_zoom
                     ))





