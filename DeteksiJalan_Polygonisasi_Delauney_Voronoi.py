# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:15:41 2021

@author: admin
"""

import cv2
import numpy as np

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.geometry import MultiPoint, MultiLineString

from sklearn.metrics.pairwise import euclidean_distances

from osgeo import ogr, osr

#%% Helper functions

PI = 3.1415926535

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

def Sort_Tuple(tup, axis): 
  
    return(sorted(tup, key = lambda x: x[axis]))  

def Sort_Tuple_squared(tup): 
  
    return(sorted(tup, key = lambda x: (x[0]**2 + x[1]**2)))  

def Sort_Points_squared(list_of_Points): 
  
    return(sorted(list_of_Points, key = lambda x: (x.x**2 + x.y**2))) 

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

def Output_SHP_Points(listPoints, filename):
    
    multipoint_vc_within = MultiPoint(listPoints)
    
    
    # output to a shp file
    
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(filename[:-4] + '_PointsResult.shp')
    layer = ds.CreateLayer('', sr, ogr.wkbMultiPoint)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    ## If there are multiple geometries, put the "for" loop here
    

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
    
    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(multipoint_vc_within.wkb)
    feat.SetGeometry(geom)
    
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    
    # Save and close everything
    ds = layer = feat = geom = None
    
def Output_SHP_Lines(listLines, filename):
    
    multiline = MultiLineString(listLines)
    
    
    # output to a shp file
    
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(filename[:-4] + '_LinesResult.shp')
    layer = ds.CreateLayer('', sr, ogr.wkbMultiLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    ## If there are multiple geometries, put the "for" loop here
    

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
    
    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(multiline.wkb)
    feat.SetGeometry(geom)
    
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    
    # Save and close everything
    ds = layer = feat = geom = None
    

#%% main part

def main():
    print("Program Deteksi jalan street map imagery!")

if __name__ == "__main__":
    main()
    
    filename = '52249_33932_16.png'
    
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
    
    cv2.imwrite('maskimage_streetmap_detJalan.jpg', img_mask)
    
    
    #%% Mendeteksi contours yang ada pada image
    
    ret, thresh1 = cv2.threshold(img_mask, 150, 255, cv2.THRESH_BINARY)
    
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    # susunan format list hierarchy
    
    # [Next, Previous, First_Child, Parent]
    
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
    
    image_countours2 = img.copy()
    
    cv2.drawContours(image_countours2, parents_countours, -1, (255, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imwrite('countours2_streetmap_detJalan.jpg', image_countours2)
    
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
    
    plt.triplot(points_list[:,0], points_list[:,1], tri.simplices.copy())
    plt.plot(points_list[:,0], points_list[:,1], '.')
    
    plt.savefig("delauney_streetmap_detJalan.jpg")
    
    plt.close()
    plt.clf()
    
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
                    
                    


    # display all lines as connected points in image
    
    img_lines = img.copy()
    
    img_lines = draw_list_edges(img_lines.copy(), flat_list_edges)
    
    cv2.imwrite('linesJalan_streetmap_detJalan.jpg', img_lines)     

#%%
    # 6. Convert detected points to longlat
    
    # konversi all points vc_within to latlong
    
    vc_within_longlat = Convert_listPoints2listLongLat(vc_within, tile_x, tile_y, tile_zoom)
    
    list_edges_longlat = Convert_listEdges2listEdgesLonglat(flat_list_edges, tile_x, tile_y, tile_zoom)
    
    Output_SHP_Points(vc_within_longlat, filename)
    
    Output_SHP_Lines(list_edges_longlat, filename)
    

    

    
                                  
 
