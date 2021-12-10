# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:00:08 2021

@author: admin
"""

from shapely.geometry import MultiPoint, MultiLineString

from osgeo import ogr, osr

'''

File ini berisis fungsi fungsi untuk output hasil dlaam bentuk shp, kml, dll.
'''


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