from sklearn import cluster, datasets


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph.console

from layer_viewer.layer_scatter import *
import numpy
import skimage.data

app = pg.mkQApp()



n_samples = 1000
pos, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
values = numpy.random.normal(size=pos.shape[0]) + pos[:,0] + pos[:,1]

n_samples = n_samples 
pos2, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
pos2 += 7.0
values2 = numpy.random.normal(size=pos2.shape[0]) + pos2[:,0] + pos2[:,1]






viewer = LayerScatter()
    
lut = numpy.random.rand(n_samples, 3)
layer = LUTScatterLayer("lut")
layer.set_lut(lut)
layer.set_data(pos=pos, values=numpy.arange(pos.shape[0]))
viewer.add_layer(layer)    

layer = FloatScatterLayer("float", pos=None, values=None)
layer.set_data(pos=pos, values=values)
layer.set_data(pos=pos2, values=values2)
viewer.add_layer(layer)    
viewer.set_range()


# # take the colormap of an other layer
# layer = LinkedCmFloatScatterLayer("lfloat", pos=pos2, values=values2, linked_layer=layer)
# viewer.add_layer(layer)    
# viewer.set_range()


viewer.setWindowTitle('LayerScatter')
viewer.show()


   



if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
