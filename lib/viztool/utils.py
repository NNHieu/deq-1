import torch
import os
import h5py
from .landscape import Dir2D, Surface

def name_surface_file(rect, res, dir_file):
    # use args.dir_file as the perfix
    surf_file = dir_file
    xmin, ymin, xmax, ymax = rect
    xnum, ynum = res
    # resolution
    surf_file += '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum))
    surf_file += 'x[%s,%s,%d]' % (str(ymin), str(ymax), int(ynum))

    return surf_file + ".h5"

def create_surfile(model, layers, dir_file, surf_file, rect, resolution, logger):
    if not os.path.exists(dir_file):
        logger.info('Create dir file at {}'.format(dir_file))
        dir2d = Dir2D(model=model)
        try:
            with h5py.File(dir_file, 'w') as f:
                dir2d.save(f)
        except Exception as e:
            os.remove(dir_file)
            raise e
    
    if not os.path.exists(surf_file):
        logger.info('Create surface file at {}'.format(surf_file))
        surface = Surface(dir_file, rect, resolution, surf_file, {})
        surface.add_layer(*layers)
        surface.save()
    
    return surf_file