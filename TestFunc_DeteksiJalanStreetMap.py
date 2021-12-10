# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:20:57 2021

@author: admin
"""

import glob, os

from MainFunc_DeteksiJalanStreetMap import DeteksiJalanStreetMap
    
    
#%% untuk testing apakah fungsinya sudah ok


list_all_points_area = []

list_all_lines_area = []


#%% loop pada seluruh datacitra cimahi


os.chdir("./data_Cimahi")

count = 1

for filename in glob.glob("*.png"):
    print(filename)
    print(count)




#%% tiap gambar dapatkan list of points

    returned_points, returned_lines = DeteksiJalanStreetMap(filename, outputPointsOnly=False, outputImages=False, outputSHP=False)
    
    list_all_points_area.append(returned_points)
    
    list_all_lines_area.append(returned_lines)
    
    count += 1
    
#%% Outputkan file shp all area

from HelperFuncs_Output import *

flat_list_points = []
for sublist in list_all_points_area:
    for item in sublist:
        flat_list_points.append(item)
            

flat_list_edges = []
for sublist in list_all_lines_area:
    for item in sublist:
        flat_list_edges.append(item)            


fileout = 'Cimahi_Deteksi.png'

Output_SHP_Points(flat_list_points, fileout)
    
Output_SHP_Lines(flat_list_edges, fileout)

# dia ngesave di folder imager area nya
    
    
    
    