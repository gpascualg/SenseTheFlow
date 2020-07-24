import shapefile as sf
from pyqtree import Index

from ...config import bar


class GeoReader(object):
    def __init__(self, patches_iterator, shapefiles, class_mappings, bbox=None):
        assert (shapefiles is None and class_mappings is None) or len(shapefiles) == len(class_mappings)

        self.patches_iterator = patches_iterator
    
        if shapefiles is None:
            self.readers = []
        else:
            if isinstance(shapefiles[0], str):
                self.readers = [sf.Reader(instance) for instance in shapefiles]
            else:
                self.readers = [ShapeFaker(instance) for instance in shapefiles]
    
        self.class_mappings = class_mappings
        self.q = None
        self.q_at = None
        self.__bbox = bbox
        self.location_cache = {}
        self.legend_handles = None
                
    def bbox(self):
        if self.__bbox is None:
            self.__bbox = greatest(reader.bbox for reader in self.readers)
        return self.__bbox
        
    def build(self):
        assert self.q is None
        self.q = Index(self.bbox(), maxitems=100, maxdepth=1000)
        
        for k, reader in enumerate(bar(self.readers)):
            shapeRecords = reader.iterShapeRecords()
            num = reader.numRecords
            
            for shape in bar(shapeRecords, total=num):
                try:
                    self.q.insert((shape, k), shape.shape.bbox)
                except:
                    print('Shape without bbox')

    def build_at(self):
        assert self.q_at is None        
        self.q_at = Index(self.bbox(), maxitems=100, maxdepth=1000)
        
        for patch in bar(self.patches_iterator()):
            self.q_at.insert(patch, patch.bbox())
       
    def __getitem__(self, coords):
        assert self.q_at is not None     
        return self.q_at.intersect((*coords, *coords))
        
    def _load_polys(self, patch):
        if self.q is None:
            return set()
        
        return self.q.intersect(patch.bbox)
    
    def _load_patch(self, patch):
        patch.polys = self._load_polys(patch)
        patch.img = imread(GeoData.IMG_PATH.format(self.year, patch.name))
        
        if patch.crop:
            x,y,w,h = patch.crop
            rh,_,_ = patch.img.shape
            patch.img = patch.img[rh-(y+h):rh-y,x:x+w]
        
        return patch

    def __iter__(self):
        for patch in self.patches_iterator():
            yield patch, patch.encode((poly, self.class_mappings[cls_mapper_idx]) \
                for poly, cls_mapper_idx in self.q.intersect(patch.bbox()))

    def polys_at(self, bbox):
        for poly, cls_mapper_idx in self.q.intersect(bbox):
            yield poly, self.class_mappings[cls_mapper_idx]
