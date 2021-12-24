import torch

def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key

        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """ Read group with name as the key from the hdf5 file and return a list numpy vectors. """
    grp = f[name]
    return [grp[str(i)][:] for i in range(len(grp))]

def name_surface_file(rect, res, dir_file):
    # use args.dir_file as the perfix
    surf_file = dir_file
    xmin, ymin, xmax, ymax = rect
    xnum, ynum = res
    # resolution
    surf_file += '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum))
    surf_file += 'x[%s,%s,%d]' % (str(ymin), str(ymax), int(ynum))

    return surf_file + ".h5"