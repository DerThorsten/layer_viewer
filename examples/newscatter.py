
from layer_viewer.widgets.scatter.vispy_scatter import *

from sklearn import cluster, datasets


import numpy
import pyqtgraph as pg
app = pg.mkQApp()



import logging
logging.basicConfig(level=logging.DEBUG)

n_samples = 1000000




pos, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
values = numpy.random.normal(size=pos.shape[0]) + pos[:,0] + pos[:,1]

n_samples = n_samples 
pos2, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
pos2 += 2.0
values2 = numpy.random.normal(size=pos2.shape[0]) + pos2[:,0] + pos2[:,1]


data_list = [
   {
        "pos" : pos,
        "values" : values,
        "name": "main0",
        "color": (1,0,0)
    },
    # {
    #     "pos" : pos2,
    #     "values" : values2,
    #     "name": "background",
    #     "color": (0,1,0)
    # }
]

widget = VisPyScatter(n_bins_histxy=100,sigma_histxy=1.0, show_edges=False)
widget.set_data(data_list=data_list)
widget.show()

