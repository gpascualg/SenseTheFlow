def greatest(bboxes):
    bboxes = iter(bboxes)
    xmin, ymin, xmax, ymax = next(bboxes)

    # Iter remaining
    for bbox in bboxes:
        xmin = min(xmin, bbox[0])
        ymin = min(ymin, bbox[1])
        xmax = max(xmax, bbox[2])
        ymax = max(ymax, bbox[3])

    return [xmin, ymin, xmax, ymax]

def map_geointerface_points(function, points):
    if isinstance(points, (list, tuple)):
        if isinstance(points[0], (list, tuple)):
            return [map_geointerface_points(function, p) for p in points]
        
        return function(points)

def flatten_geointerface_points(points):
    if isinstance(points, (list, tuple)):
        # It is a point already
        if not isinstance(points[0], (list, tuple)):
            return [points]
        else:
            return sum((flatten_geointerface_points(sublist) for sublist in points), [])

def reproject(patch, coords, proj):
    lat, lon = project(patch, coords)
    return proj(lon, lat)
    