"""
Hello World
=========================
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph.console

from layer_viewer import dcolors
from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *
import numpy
import skimage.data


app = pg.mkQApp()

image = skimage.data.astronaut().swapaxes(0,1)




viewer = LayerViewerWidget()
viewer.setWindowTitle('LayerViewer')
viewer.show()

layer = MultiChannelImageLayer(name='img', data=image[...])
viewer.addLayer(layer=layer)


labels = numpy.zeros(image.shape[0:2], dtype='uint8')
label_layer = LabelLayer(name='labels', data=None)
viewer.addLayer(layer=label_layer)
viewer.setData('labels',image=labels)


# connect stuff
def foo(layer):
	print(labels)
label_layer.labelsChangedSignal.connect(foo)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()
