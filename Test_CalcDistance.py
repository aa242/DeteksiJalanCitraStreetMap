# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:36:33 2021

@author: admin
"""

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

points = [[1, 1], [2, 2], [4, 4], [3, 5], [2, 8], [4, 10], [3, 7], [2, 9], [4, 7]]

dist = euclidean_distances(points, points)