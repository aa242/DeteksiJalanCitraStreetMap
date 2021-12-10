# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:22:44 2021

@author: admin
"""

"""
Berikut adalah fungsi - fungsi untuk melakukan merger lines yang dihasilkan 
oleh fungsi Hough transform
"""

import cv2

def get_lines(lines):
    
    if cv2.__version__ < '3.0':
        return lines[0]
    
    return [l[0] for l in lines]


def process_lines(img_in):
    
    img = mpimg.imread(img_in)
    
    gray = 