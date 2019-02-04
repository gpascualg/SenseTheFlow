import random
import cv2
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, MultiLineString

from SenseTheFlow.config import bar

from .utils import *


class GeoPatch(object):
    def __init__(self, bbox, center, width, height, **data):
        self._bbox = bbox
        self._center = center
        self._width = width
        self._height = height
        self._position = self._bbox[:2]
        self.data = data

    def bbox(self):
        return self._bbox

    def crop(self, width, height, x0=None, y0=None):
        tw = self._width - width
        th = self._height - height
        assert tw > 0 and th > 0
        
        x0 = x0 or int(random.random() * tw)
        y0 = y0 or int(random.random() * th)
        assert x0 < tw and y0 < th

        ratio = (
            (self._bbox[2] - self._bbox[0]) / self._width,
            (self._bbox[3] - self._bbox[1]) / self._height
        )

        self._width = width
        self._height = height
        self._position = [
            self._position[0] + x0 * ratio[0],
            self._position[1] + y0 * ratio[1]
        ]
        self._bbox = [
            *self._position,
            self._position[0] + width * ratio[0],
            self._position[1] + height * ratio[1]
        ]
        self._center = (
            self._bbox[0] + (self._bbox[2] - self._bbox[0]) / 2,
            self._bbox[1] + (self._bbox[3] - self._bbox[1]) / 2,
        )
        
    def ratio(self):
        return (
            (self._bbox[2] - self._bbox[0]) / self._width,
            (self._bbox[3] - self._bbox[1]) / self._height
        )
        
    def real_coordinate(self, p):
         return (
             self._bbox[2] - (1 - p[0] / self._width) * (self._bbox[2] - self._bbox[0]),
             self._bbox[3] - ((p[1] - self._height) / self._height + 1) * (self._bbox[3] - self._bbox[1])
         )
        
    def map_coordinate(self, p):
        return (
            int(self._width * (1 - (self._bbox[2] - p[0]) / (self._bbox[2] - self._bbox[0]))),
            self._height - int(self._height * (1 - (self._bbox[3] - p[1]) / (self._bbox[3] - self._bbox[1])))
        )

    def _extract_coords(self, p):
        coordinates = list(map(self.map_coordinate, p.coords))
        return np.asarray([coordinates])

    def encode(self, polys, fail_silently=False):
        try:
            arr = np.zeros((self._height, self._width), dtype=np.uint8)
            for poly, class_mappings in polys:
                s = shape(poly.shape.__geo_interface__)
                try:
                    Polygon(s)
                    coordinates = self._extract_coords(Polygon(poly.shape.__geo_interface__['coordinates']))
                except:
                    try:
                        MultiLineString(s)
                        p = MultiLineString(poly.shape.__geo_interface__['coordinates'])
                        coordinates = [self._extract_coords(m) for m in p]
                    except:
                        try:
                            MultiPolygon(s)
                            p = MultiPolygon(poly.shape.__geo_interface__['coordinates'])
                            coordinates = [self._extract_coords(Polygon(m).exterior) for m in p]
                        except:
                            try:
                                coordinates = self._extract_coords(s)
                            except:
                                p = MultiLineString(poly.shape.__geo_interface__['coordinates'])
                                coordinates = [self._extract_coords(m) for m in p]
                        
                cv2.fillPoly(
                    arr, 
                    coordinates, 
                    class_mappings(poly.record)
                )

            return arr
        except Exception as e:
            if not fail_silently:
                raise e

            return np.zeros((self._height, self._width), dtype=np.int8)
