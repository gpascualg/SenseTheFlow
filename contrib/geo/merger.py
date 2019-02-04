import numpy as np

from ...config import bar


def merge_into_bbox(top_left, tile_size, ratio, reader, load_patch_image):
    buffer = np.empty((*tile_size, 3))
    notdone = np.ones(tile_size, dtype=bool)

    limits = top_left
    rw = 0
    rh = 0

    pixels_total = np.prod(tile_size)
    done_so_far = 0
    updater = bar(total=pixels_total)
    updater.set_description('({}, {})'.format(0, 0))

    while True:
        argmax = notdone.argmax()
        xi, yi = argmax // tile_size[0], argmax % tile_size[1]
        updater.set_description('({}, {})'.format(xi, yi))

        if not notdone[xi, yi]:
            break

        coords = (
            top_left[0] + ratio[0] * yi,
            top_left[1] - ratio[1] * xi
        )

        has_found_valid_patch = False
        for patch in reader[coords]:
            pixel_point = patch.map_coordinate(coords)

            im = load_patch_image(patch).astype(float) / 255
            rw = min(tile_size[1] - yi, patch._width - pixel_point[0])
            rh = min(tile_size[0] - xi, patch._height - pixel_point[1])

            if rw == 0 or rh == 0:
                continue

            has_found_valid_patch = True
            buffer[xi:xi+rh, yi:yi+rw] = im[pixel_point[1]:pixel_point[1]+rh, pixel_point[0]:pixel_point[0]+rw]
            notdone[xi:xi+rh, yi:yi+rw] = False

        if not has_found_valid_patch:
            break

        done = pixels_total - notdone.sum()
        updater.update(done - done_so_far)
        done_so_far = done

    is_incomplete = notdone.sum() > 0
    if is_incomplete:
        buffer[np.where(notdone)] = 0

    limits = (
        limits[0],
        limits[1] - tile_size[1] * ratio[1],
        limits[0] + tile_size[0] * ratio[0],
        limits[1]
    )

    patch = GeoPatch(limits, ((limits[0] + limits[2]) / 2, (limits[1] + limits[3]) / 2), *tile_size)
    try:
        polys = [(poly, reader.class_mappings[cls_mapper_idx]) for poly, cls_mapper_idx in reader.q.intersect(limits)]
        polys = patch.encode(polys)
    except:
        polys = np.zeros(buffer.shape[:2])

    return patch, buffer, polys, not is_incomplete
