import sys
import numpy as np

sys.path.append('../')
from lib.viztool.landscape import Surface, Dir2D, Sampler
from lib.viztool import projection as proj, scheduler

sur1 = Surface.load('LM-TFMdeq_rank0_nproc2/wt103/default/surf_[-1.0,1.0,31]x[-1.0,1.0,31].h5')
sur2 = Surface.load('LM-TFMdeq_rank1_nproc2/wt103/default/surf_[-1.0,1.0,31]x[-1.0,1.0,31].h5')

layers = sur1.layers.keys()

for layer in layers:
    mer = np.amax(np.stack((sur1.layers[layer],sur2.layers[layer]), axis=-1), axis=-1)
    assert mer.shape == sur1.layers[layer].shape
    sur1.layers[layer][:] = mer[:]
    sur2.layers[layer][:] = mer[:]
sur1.save('w')
sur2.save('w')
