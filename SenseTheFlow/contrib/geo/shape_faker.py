from .utils import *

class ShapeFaker(object):
    class FakedShape(object):
        def __init__(self, interface):
            self.bbox = greatest((*x, *x) for x in flatten_geointerface_points(interface['coordinates']))
            self.__geo_interface__ = interface
            
    class FakedPoly(object):
        def __init__(self, points):
            self.shape = ShapeFaker.FakedShape(points)
            self.record = 0
            
    def __init__(self, geointerfaces):
        self.iterable = [ShapeFaker.FakedPoly(interface) for interface in geointerfaces]
        self.numRecords = len(self.iterable)
        self.bbox = greatest(poly.shape.bbox for poly in self.iterable)

    def iterShapeRecords(self):
        return self.iterable
