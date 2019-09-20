
from layer_viewer.widgets.scatter.vispy_scatter import *

from sklearn import cluster, datasets


import numpy
import pyqtgraph as pg
app = pg.mkQApp()

#w = QMainWindow()
def gen_toy_data(n = 100000):

    data = np.empty((n, 2))
    lasti = 0
    for i in range(1, 20):
        nexti = lasti + (n - lasti) // 2
        scale = np.abs(np.random.randn(2)) + 0.1
        scale[1] = scale.mean()
        data[lasti:nexti] = np.random.normal(size=(nexti-lasti, 2),
                                             loc=np.random.randn(2),
                                             scale=scale / i)
        lasti = nexti
    data = data[:lasti]
    return data





n_samples = 100000
pos, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
values = numpy.random.normal(size=pos.shape[0]) + pos[:,0] + pos[:,1]

n_samples = 1000000
pos2, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
pos2 += 2.0
values2 = numpy.random.normal(size=pos2.shape[0]) + pos2[:,0] + pos2[:,1]


# data_list = [
#     "main" : {
#         "pos" : pos,
#         "values" : values,
#         "name": "main",
#     },
#     "background" : {
#         "pos" : pos2,
#         "values" : values2,
#         "name": "background",
#     }
# ]


# widget = VisPyScatter()
# widget.set_data(data_list=data_list)#, data2, values2)
# widget.show()


if True:



    widget = VisPyScatter(with_background_scatter=True)
    widget.set_levels(values.min(), values.max())
    widget.set_data(pos, values, pos2, values2)
    widget.show()
