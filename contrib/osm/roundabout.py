import copy
import math
import pyproj
import numpy as np

from shapely.geometry import Point

from ..geo.utils import *


def project(data, point):
    zoom = 2**data.zoom_level
    
    px = point[0] / data.extent
    py = 1 - point[1] / data.extent
    
    x0 = data.col + px
    y0 = zoom - data.row - 1 + py
    
    lon = 360 * (x0 / zoom) - 180
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y0 / zoom)))) * 180 / math.pi
    return lat, lon
        
class Roundabout(object):
    def __init__(self, zoom_level, data, row, col, extent):
        self.zoom_level = zoom_level
        self.data = data
        self.row = row
        self.col = col
        self.extent = extent
        
    def __str__(self):
        return "({}, {})x{}: {}".format(self.row, self.col, self.extent, self.data)

    def coordinates(self):
        return np.asarray(flatten_geointerface_points(self.data[0]['geometry']['coordinates']))
    
    def geointerface(self, projection=None):
        geometry = copy.deepcopy(self.data[0]['geometry'])
        
        if projection is not None:
            proj = pyproj.Proj(projection)            
            geometry['coordinates'] = map_geointerface_points(lambda c: reproject(self, c, proj), geometry['coordinates'])
            
        return geometry
    
    def enclosingCircle(self, projection=None, scale_radius=False):
        bbox = greatest((*x, *x) for x in self.coordinates())
        cx, cy = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
        r = max(bbox[2] - cx, bbox[3] - cy)
        
        if projection is not None:
            proj = pyproj.Proj(projection)
            cx, cy = reproject(self, (cx, cy), proj)
            
        resolution = 1
        if scale_radius:
            resolution = 40075.016686 * 1000 / self.extent * abs(math.cos(cx)) / (2**self.zoom_level)
            
        return (cx, cy), r * resolution

    def enclosingGeointerface(self, projection=None):
        center, r = self.enclosingCircle(projection=None)
        coordinates = list(Point(center).buffer(r).exterior.coords)
        
        if projection is not None:
            proj = pyproj.Proj(projection)            
            coordinates = map_geointerface_points(
                lambda c: reproject(self, c, proj), 
                coordinates
            )
            
        return {
            'type': 'LineString',
            'coordinates': coordinates
        }
