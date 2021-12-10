# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:20:57 2021

@author: admin
"""

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.geometry import MultiPoint, MultiLineString

from sklearn.metrics.pairwise import euclidean_distances

from HelperFuncs_Geom import *
from HelperFuncs_Geos import *
from HelperFuncs_Misc import *
from HelperFuncs_Output import *

def DeteksiJalanStreetMap(filename, outputPointsOnly=True, outputImages=False, outputSHP=False):
    
    #filename = '52249_33932_16.png'
    
    geos_info = filename[:-4].split('_')
    
    tile_x = float(geos_info[0])
    
    tile_y = float(geos_info[1])
    
    tile_zoom = float(geos_info[2])
    
    img = cv2.imread(filename)
    
    img_mask = img.copy()
    
    # create a mask image
    
    img_mask[np.where((img_mask != [255,255,255]).all(axis = 2))] = [0,0,0]
    
    # save mask image result
    
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    
    if outputImages:
    
        cv2.imwrite('maskimage_streetmap_detJalan.jpg', img_mask)
    
    
    #%% Mendeteksi contours yang ada pada image
    
    ret, thresh1 = cv2.threshold(img_mask, 150, 255, cv2.THRESH_BINARY)
    
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    # susunan format list hierarchy
    
    # [Next, Previous, First_Child, Parent]
    
    if outputImages:
    
        image_countours1 = img.copy()
        
        cv2.drawContours(image_countours1, contours2, -1, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite('countours1_streetmap_detJalan.jpg', image_countours1)
    
    # get all parent countours as list
    
    list_parents = np.unique(hierarchy2[0,:,3].tolist())
    
    # get all parents without child
    
    list_have_child = hierarchy2[0,:,2].tolist()
    
    # get list of all children of parent
    
    list_child_parents = hierarchy2[0,:,3].tolist()
    
    list_childess = []
    
    for idx in range(len(list_have_child)):
        
        if list_have_child[idx] == -1 and list_child_parents[idx] == -1 :
            
            list_childess.append(idx)
    
    # bangun dict parent-child
    
    dict_ParentChild = {}
    
    for idx in list_parents:
        
        if not idx == -1:
            
            dict_ParentChild[idx] = []
            
            
    for idx in range(len(list_child_parents)):
        
        if not list_child_parents[idx] == -1:
        
            dict_ParentChild[list_child_parents[idx]].append(idx)
            
    # draw all parent countours
    
    #parents_countours = [x for x, y in zip(contours2, list_parents) if y]
    
    parents_countours = []
    
    for idx in list_parents:
        
        parents_countours.append(contours2[idx])
        
    if outputImages:
    
        image_countours2 = img.copy()
        
        cv2.drawContours(image_countours2, parents_countours, -1, (255, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite('countours2_streetmap_detJalan.jpg', image_countours2)
    
    if outputImages:
    
    # draw all child contours
    
        for idx_parent in list_parents:
            
            if idx_parent != -1:
            
                parent = [contours2[idx_parent]]
                
                list_children = dict_ParentChild[idx_parent]
                
                children_countours = []
            
                for idx in list_children:
                    
                    children_countours.append(contours2[idx])
                
                #draw parent first
                
                image_countours3 = img.copy()
            
                cv2.drawContours(image_countours3, parent, -1, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.drawContours(image_countours3, children_countours, -1, (0, 127, 127), 1, cv2.LINE_AA)
                
                cv2.imwrite('countours3_' + str(idx_parent) + '_streetmap_detJalan.jpg', image_countours3)
    
    
    
    
    # draw all childess countours
    
    childless_countours = []
    
    for idx in list_childess:
        
        childless_countours.append(contours2[idx])
        
    if outputImages:
        
        image_countours4 = img.copy()
        
        cv2.drawContours(image_countours4, childless_countours, -1, (127, 127, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite('countours4_streetmap_detJalan.jpg', image_countours4)  
    
    
    #3. Delauney Triangulation of all contours
    
    
    polys = []
    
    for cont in contours2:
        approx_curve = cv2.approxPolyDP(cont, 3, False)
        polys.append(approx_curve)
    
    # convert all detected polygos into just the points only
    
    points_list = []
    
    for points in contours2: # kalau mau cepat pakai polys
    
        list2append = ConvertToList(points)
        
        for point in list2append:
            
            points_list.append(point)
    
    
    points_list = np.array(points_list)

    tri = Delaunay(points_list)
    
    if outputImages:
    
        plt.triplot(points_list[:,0], points_list[:,1], tri.simplices.copy())
        plt.plot(points_list[:,0], points_list[:,1], '.')
        
        plt.savefig("delauney_streetmap_detJalan.jpg")
        
        plt.close()
        plt.clf()
    
    if outputImages:
    
        # draw on original image delauney lines
        
        img_delauney = img.copy()
        
        img_delauney = draw_list_triangles(img_delauney.copy(), points_list[tri.simplices].tolist())
        
        cv2.imwrite('delauneyImage_streetmap_detJalan.jpg', img_delauney)
    
    
    # 4. voronoi Diagram
    
    p = tri.points[tri.vertices]

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T
    
    a = A - C
    b = B - C
    
    cc = crossprod(sq2(a) * b - sq2(b) * a, a, b) / (2*ncrossprod(a, b)) + C
    
    # Grab the Voronoi edges
    vc = cc[:,tri.neighbors]
    vc[:,tri.neighbors == -1] = np.nan # edges at infinity, plotting those would need more work...
    
    lines_raw = []
    lines_raw.extend(zip(cc.T, vc[:,:,0].T))
    lines_raw.extend(zip(cc.T, vc[:,:,1].T))
    lines_raw.extend(zip(cc.T, vc[:,:,2].T))
    
    lines = LineCollection(lines_raw, edgecolor='k')
    
    if outputImages:
    
        plt.hold(1)
        plt.plot(points_list[:,0], points_list[:,1], '.')
        plt.plot(cc[0], cc[1], '*')
        plt.gca().add_collection(lines)
        plt.axis('equal')
        plt.xlim(0, 256)
        plt.ylim(0, 256)
    
        plt.savefig("voronoi_streetmap_detJalan.jpg")
        
        plt.close()
        plt.clf()    
        
    # draw edges voronoi on image
    
    if outputImages:
    
        img_voronoi = img.copy()
        
        img_voronoi = draw_list_voronoi(img_voronoi.copy(), lines_raw)
        
        cv2.imwrite('voronoiImage_streetmap_detJalan.jpg', img_voronoi)
    
    # 5. Get all points within polysgons
    
    vc_list = []

    for i in range(cc.shape[1]):
        
        point = (cc[0,i], cc[1,i])
        
        vc_list.append(point)
        
    # create parent polygons with children
    
    list_poly_with_holes = []
    
    for pa_idx in list_parents:
        
        if pa_idx != -1:
            
            parent_points_tuple = ConvertToTuple(contours2[pa_idx])
            
            # get children points
            
            list_children2 = dict_ParentChild[pa_idx]
            
            multi_children_points_tuple = []
            
            for ch_idx in list_children2:
                
                children_points_tuple = ConvertToTuple(contours2[ch_idx])
                
                multi_children_points_tuple.append(children_points_tuple)
                
            
            multi_children_points_tuple = tuple(multi_children_points_tuple)
            
            
            # bangun polygon parent ini dengan holesnya
            
            ParentPolygonAndChildren = Polygon(parent_points_tuple, multi_children_points_tuple)
            
            polygonExterior = ParentPolygonAndChildren.exterior
            polygonInteriors = []
            for i in range(len(ParentPolygonAndChildren.interiors)):
                polygonInteriors.append(ParentPolygonAndChildren.interiors[i])
            
            ParentPolygonWithHoles = Polygon(polygonExterior, [[pt for pt in inner.coords] for inner in polygonInteriors])
            
            list_poly_with_holes.append(ParentPolygonWithHoles)
            
            
    # add also childless parents
    
    for chless_idx in list_childess:
        
        if len(polys[chless_idx]) >2:
        
            childless_poly = Polygon(ConvertToList(contours2[chless_idx]))  
            
            list_poly_with_holes.append(childless_poly)
            
        
    # create the mutipolygon combining all polygons
    
    Multipoly = MultiPolygon(list_poly_with_holes)
    
    vc_list = remove_closePoints(vc_list, 4, 256)

    
    
    vc_list_points = [Point(elem) for elem in vc_list]
    
    vc_within = []
    
    for point in vc_list_points:
        
        # check if within parent
        
        to_return = 0
        
        for pa_idx in list_parents:
            
            if pa_idx != -1:
                
                pa_points = ConvertToList(contours2[pa_idx])
                
                Pa_Poly = Polygon(pa_points)
                
                if Pa_Poly.contains(point):
                    
                    to_return = point
                    
                    # check through all children
                    
                    list_children3 = dict_ParentChild[pa_idx]
                    
                    for ch_idx in list_children3:
                        
                        ch_points = ConvertToList(contours2[ch_idx])
                        
                        Ch_Poly = Polygon(ch_points)             
                        
                        if Ch_Poly.contains(point):
                            
                            to_return = 0
        
        
        # loop through childless

        for chless_idx in list_childess:
            
            if len(polys[chless_idx]) >2:
                
                chless_points = ConvertToList(contours2[chless_idx])
                
                Chless_Poly = Polygon(chless_points)
                
                if Chless_Poly.contains(point):
                    
                    to_return = point 
                    
        if to_return != 0:
            
            vc_within.append(to_return)
            
    if outputImages:
    
        # display all vc_wthin points in image
        
        img_res_points = img.copy()
        
        img_res_points = draw_points_list(img_res_points.copy(), vc_within)
        
        cv2.imwrite('pointsJalan_streetmap_detJalan.jpg', img_res_points)
        
 
    #%%
    
    # convert points to list
    
    vc_within_array = ConvertPointsToList(vc_within)
    
    dist = euclidean_distances(vc_within_array, vc_within_array)
    
    
    # draw edges givent the vertices
    
    list_edges = []
    
    # function to get edges from vertices
    
    for pt1_idx in range(len(vc_within)):
        
        list_curr_edges = [] # maks 3 aja
        
        list_curr_dist = []
        
        curr_maks_dist = 256
        
        for pt2_idx in range(pt1_idx+1, len(vc_within)):
            
            if dist[pt1_idx, pt2_idx] < 24 :
                
                # kalau masih kurang dari 3, maka add aja
                
                if len(list_curr_edges) < 1:
                
                    list_curr_edges.append([vc_within[pt1_idx], vc_within[pt2_idx]]) 
                    list_curr_dist.append(dist[pt1_idx, pt2_idx])
                    
                    curr_maks_dist = max(list_curr_dist)
                
                elif  dist[pt1_idx, pt2_idx] < curr_maks_dist: # lebih dari 3, maka ganti yang tertinggi dengan data baru
                
                    # get idx of current max distance
                    
                    idx_max = list_curr_dist.index(curr_maks_dist)
                    
                    list_curr_edges[idx_max] = [vc_within[pt1_idx], vc_within[pt2_idx]]
                    list_curr_dist[idx_max] = dist[pt1_idx, pt2_idx]
                    
                    curr_maks_dist = max(list_curr_dist)
                    
        list_edges.append(list_curr_edges)
        
    
    
    # unwrap list of edges
    
    flat_list_edges = []
    for sublist in list_edges:
        for item in sublist:
            flat_list_edges.append(item)
                    
                    

    if outputImages:
    
        # display all lines as connected points in image
        
        img_lines = img.copy()
        
        img_lines = draw_list_edges(img_lines.copy(), flat_list_edges)
        
        cv2.imwrite('linesJalan_streetmap_detJalan.jpg', img_lines)     

#%%
    # 6. Convert detected points to longlat
    
    # konversi all points vc_within to latlong
    
    vc_within_longlat = Convert_listPoints2listLongLat(vc_within, tile_x, tile_y, tile_zoom)
    
    list_edges_longlat = Convert_listEdges2listEdgesLonglat(flat_list_edges, tile_x, tile_y, tile_zoom)
    
    if outputSHP:
    
        Output_SHP_Points(vc_within_longlat, filename)
        
        Output_SHP_Lines(list_edges_longlat, filename)
    
    if outputPointsOnly == True:
        
        return vc_within_longlat
    
    else:
        
        return vc_within_longlat, list_edges_longlat
    
    
#%% untuk testing apakah fungsinya sudah ok


    
    