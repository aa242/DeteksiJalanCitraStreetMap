# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:47:01 2021

@author: admin
"""

import numpy as np

'''

File ini berisis fungsi fungsi untuk sorting, ubah format, filtering dll.
'''


def remove_closePoints(listnya, thresh, imgsize):
    
    bins = []
    
    #make empty bins
    
    
    for i in range(int(np.floor(imgsize/thresh))):
        
        for j in range(int(np.floor(imgsize/thresh))):
        
            bin_thresh = []
            
            for elem in listnya:
                
                #print(elem)
                
                if ((elem[0] >= i*thresh) and (elem[0] < (i+1)*thresh)) and ((elem[1] >= j*thresh) and (elem[1] < (j+1)*thresh)):
                    
                    bin_thresh.append(elem)
                    
        
            bins.append(bin_thresh)
        
    #return first elem of each bin
    
    bin_ret = []
    
    for elem in bins:
        
        if len(elem)>0:
        
            bin_ret.append(elem[0])
        
    return bin_ret

def Sort_Tuple(tup, axis): 
  
    return(sorted(tup, key = lambda x: x[axis]))  

def Sort_Tuple_squared(tup): 
  
    return(sorted(tup, key = lambda x: (x[0]**2 + x[1]**2)))  

def Sort_Points_squared(list_of_Points): 
  
    return(sorted(list_of_Points, key = lambda x: (x.x**2 + x.y**2))) 

def ConvertToList(points):
    
    listPoints = []
    
    for i in range(points.shape[0]):
        listPoints.append([points[i,0,0], 256 - points[i,0,1]])
        
    return listPoints

def ConvertToTuple(points):
    
    listPoints = []
    
    for i in range(points.shape[0]):
        listPoints.append((points[i,0,0], 256 - points[i,0,1]))
        
    return tuple(listPoints)

