
from layer_viewer.widgets.scatter.vispy_scatter import *




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



from vispy.color import Colormap

#  per image scatter
cm = Colormap(['r', 'b'])
data =gen_toy_data(1000000)
values = numpy.random.normal(size=data.shape[0]) + data[:,0]*data[:,1] - data[:,0]




widget = VisPyScatter(with_background_scatter=False)
widget.set_levels(values.min(), values.max())
widget.set_data(data, values)#, data2, values2)
widget.show()



# global scatter
data2 = gen_toy_data(n=100000)
values2 = numpy.random.normal(size=data2.shape[0]) + data2[:,0]


widget = VisPyScatter(with_background_scatter=True)
widget.set_levels(values.min(), values.max())
widget.set_data(data, values, data2, values2)
widget.show()
