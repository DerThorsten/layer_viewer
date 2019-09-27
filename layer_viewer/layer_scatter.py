from PyQt5.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from abc import ABCMeta,abstractmethod
import vispy.plot
import vispy.app
import vispy
import sys
import numpy
from vispy import scene, app
from vispy.color import ColorArray
from vispy.visuals.filters import Clipper, Alpha, ColorFilter

from .layers.layer_controller import LayerItemWidget
from .layer_ctrl_widget import LayerCtrlWidget



from contextlib import contextmanager


def clip_norm(data, minv, maxv):
    c  = numpy.clip(data, minv, maxv)
    c -= minv
    c  /=(maxv -  minv)
    return c



def set_padding(self, minp=10, maxp=10):
    padding_left =  self.grid._grid_widgets[0][-1]
    padding_right =  self.grid._grid_widgets[1][-1]
    padding_bottom =  self.grid._grid_widgets[2][-1]

    # padding left
    padding_left.width_min = minp
    padding_left.width_max = maxp
    padding_right.width_min = minp
    padding_right.width_max = maxp
    padding_bottom.height_min = minp
    padding_bottom.height_max = maxp


def myconv2ds(self, fg_color=None):
    if self._configured:
        return

    if fg_color is None:
        fg = self._fg
    else:
        fg = fg_color

    #     c0        c1      c2      c3      c4       v.freeze()
    #
    #  r0 +---------+-------+-------+-------+-------+---------+---------+
    #     |         |                       | title |         |         |
    #  r1 |         +-----------------------+-------+---------+         |
    #     |         |                       | cbar  |         |         |
    #  r2 |         +-------+-------+-------+-------+---------+         |
    #     |         | cbar  | ylabel| yaxis |  view | cbar    | padding |
    #  r3 | padding +-------+-------+-------+-------+---------+         |
    #     |         |                       | xaxis |         |         |
    #  r4 |         +-----------------------+-------+---------+         |
    #     |         |                       | xlabel|         |         |
    #  r5 |         +-----------------------+-------+---------+         |
    #     |         |                       | cbar  |         |         |
    #  r6 |---------+-----------------------+-------+---------+---------|
    #     |                           padding                           |
    #     +---------+-----------------------+-------+---------+---------+

    # padding left
    padding_left = self.grid.add_widget(None, row=0, row_span=5, col=0)
    padding_left.width_min = 5
    padding_left.width_max = 10

    # padding right
    padding_right = self.grid.add_widget(None, row=0, row_span=5, col=6)
    padding_right.width_min = 5
    padding_right.width_max = 10

    # padding right
    padding_bottom = self.grid.add_widget(None, row=6, col=0, col_span=6)
    padding_bottom.height_min = 2
    padding_bottom.height_max = 4

    # row 0
    # title - column 4 to 5
    self.title_widget = self.grid.add_widget(self.title, row=0, col=4)
    self.title_widget.height_min = self.title_widget.height_max = 40

    # row 1
    # colorbar - column 4 to 5
    self.cbar_top = self.grid.add_widget(None, row=1, col=4)
    self.cbar_top.height_max = 1

    # row 2
    # colorbar_left - column 1
    # ylabel - column 2
    # yaxis - column 3
    # view - column 4
    # colorbar_right - column 5
    self.cbar_left = self.grid.add_widget(None, row=2, col=1)
    self.cbar_left.width_max = 1

    self.ylabel = scene.Label("", rotation=-90)
    ylabel_widget = self.grid.add_widget(self.ylabel, row=2, col=2)
    ylabel_widget.width_max = 1

    self.yaxis = scene.AxisWidget(orientation='left',
                                  text_color=fg,
                                  axis_color=fg, tick_color=fg)

    yaxis_widget = self.grid.add_widget(self.yaxis, row=2, col=3)
    yaxis_widget.width_max = 40

    self.view = self.grid.add_view(row=2, col=4,
                                   border_color='grey', bgcolor="#efefef")
    self.view.camera = 'panzoom'
    self.camera = self.view.camera

    self.cbar_right = self.grid.add_widget(None, row=2, col=5)
    self.cbar_right.width_max = 1

    # row 3
    # xaxis - column 4
    self.xaxis = scene.AxisWidget(orientation='bottom', text_color=fg,
                                  axis_color=fg, tick_color=fg)
    xaxis_widget = self.grid.add_widget(self.xaxis, row=3, col=4)
    xaxis_widget.height_max = 40

    # row 4
    # xlabel - column 4
    self.xlabel = scene.Label("")
    xlabel_widget = self.grid.add_widget(self.xlabel, row=4, col=4)
    xlabel_widget.height_max = 40

    # row 5
    self.cbar_bottom = self.grid.add_widget(None, row=5, col=4)
    self.cbar_bottom.height_max = 1

    self._configured = True
    self.xaxis.link_view(self.view)
    self.yaxis.link_view(self.view)



@contextmanager
def unfreeze(*args, **kwargs):

    for arg in args:
        arg.unfreeze()
    for k,v in kwargs.items():
        v.unfreeze()
    yield 
    for arg in args:
        arg.freeze()
    for k,v in kwargs.items():
        v.freeze()







class VispyFig(vispy.plot.Fig):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)


        def conf_plot(plt):
            plt._configure_2d()
            set_padding(plt)
            plt.set_gl_state(depth_test=False)

        with unfreeze(self):
            k = 4
            self.plot_scatter = self[0:k, 0:k]
            conf_plot(self.plot_scatter)
            self.plot_hist_x = self[k, 0:k]
            conf_plot(self.plot_hist_x)
            self.plot_hist_y = self[0:k, k]
            conf_plot(self.plot_hist_y)
            self.plots = [self.plot_scatter, 
                          self.plot_hist_x, 
                          self.plot_hist_y]

    def set_range(self):
        for p in self.plots:
            p.view.camera.set_range()


class ScatterLayerBase(QtCore.QObject):
    def __init__(self):
        super().__init__()

    def setZValue(self, z):
        raise NotImplementedError()

    def on_visible_changed(self, v):
        for item in self.items:
            item.visible = v


class LUTScatterLayer(ScatterLayerBase):

    def __init__(self, name, pos=None, values=None, lut=None):
        super().__init__()
        self._name = name
        self._pos = None
        self._values = None
        self._colors = None
        self._lut = lut
        self._w = LayerItemWidget(name=name)
        self._w.layer = self

        self.item = vispy.plot.Markers()
        self.set_data(pos=pos, values=values)

    def name(self):
        return self._name

    def set_lut(self, lut):
        self._lut = lut
        self.set_data(pos=self._pos, values=self._values)

    def set_data(self, pos, values):
        if pos is not None and values is not None:
            assert self._lut is not None,"call set_lut"
            self._pos = pos
            self._values = values
            self._colors = numpy.take(self._lut, self._values, axis=0)
            print(self._colors.shape)
            self.item.set_data(pos=self._pos, face_color=self._colors) 

    def scatter_items(self):
        return [self.item]

    def ctrl_widget(self):
        return self._w


    def setZValue(self, z):
        pass



class FloatScatterLayer(ScatterLayerBase):
    sigGradientChanged = QtCore.Signal(object)
    def __init__(self, name, pos, values, cmap='thermal', levels=None):
        super().__init__()

        self._name = name
        self._pos = None
        self._values = None
        self._values01 = None
        self._colors = None

        self._w = LayerItemWidget(name=name, add_gradient_widget=True)
        self._w.gradientWidget.loadPreset(cmap)
        self._w.layer = self
        self.levels = levels
        if levels is None and values is not None:
            self.levels = values.min(),values.max()

        self.item = vispy.plot.Markers()
        self.item.antialias = False

        self.set_data(pos=pos, values=values)

  
        
        #self.item = vispy.plot.Markers(pos=self._pos, face_color=self._colors) 
        self.items = [self.item]
        self._connect_ui()

    def set_data(self, pos, values):
        if pos is not None and values is not  None:
            if self.levels is None:
                self.levels = values.min(),values.max()
            self._pos = pos
            self._values = values
            self._values01 = clip_norm(self._values, self.levels[0], self.levels[1])
            
            cm = self._w.gradientWidget.colorMap()
            self._colors = cm.mapToFloat(self._values01)    
            self.item.set_data(pos=self._pos, face_color=self._colors, edge_color=self._colors)#, edge_width=0) 

    def _connect_ui(self):
        self._w.toggleEye.stateChanged.connect(self.on_visible_changed)
        self._w.gradientWidget.sigGradientChanged.connect(self.on_cmap_changed)

    def on_cmap_changed(self, gradientWidget):
        cm = gradientWidget.colorMap()
        self._colors = cm.mapToFloat(self._values01)    
        self.item.set_data(pos=self._pos, face_color=self._colors, edge_color=self._colors) 



    def name(self):
        return self._name

    def scatter_items(self):
        return [self.item]


    def ctrl_widget(self):
        return self._w


    def setZValue(self, z):
        self.item.order = z

    @property
    def gradientWidget(self):
        return self._w.gradientWidget
    


class LinkedCmFloatScatterLayer(ScatterLayerBase):
    def __init__(self, name, pos, values, linked_layer):
        super().__init__()

        self._name = name
        self._pos = pos
        self._values = values
        self._w = LayerItemWidget(name=name)
        self._w.layer = self
        self.linked_layer = linked_layer
        self.levels = linked_layer.levels

        self._values01 = clip_norm(self._values, self.levels[0], self.levels[1])
        cm = self.linked_layer.gradientWidget.colorMap()
        self._colors = cm.mapToFloat(self._values01)    


        self.item = vispy.plot.Markers(pos=self._pos, face_color=self._colors, edge_color=self._colors) 
        self.items = [self.item]
        self._connect_ui()

    def _connect_ui(self):
        self._w.toggleEye.stateChanged.connect(self.on_visible_changed)
        self.linked_layer.gradientWidget.sigGradientChanged.connect(self.on_cmap_changed)

    def on_cmap_changed(self, gradientWidget):
        cm = gradientWidget.colorMap()
        self._colors = cm.mapToFloat(self._values01)    
        self.item.set_data(pos=self._pos, face_color=self._colors, edge_color=self._colors) 


    def name(self):
        return self._name

    def scatter_items(self):
        return [self.item]


    def ctrl_widget(self):
        return self._w


    def setZValue(self, z):
        self.item.order = z








class LayerScatter(QtGui.QWidget):
    def __init__(self):
        super().__init__()

        self.vispy_fig = VispyFig(show=False)
        self.layer_stack  = LayerCtrlWidget()
        self.layers = dict()
        self._init_ui()

    def _init_ui(self):
        self.setLayout(QtGui.QGridLayout())
        l =  self.layout()

        # the vispy fig
        l.addWidget(self.vispy_fig.native, 0, 0)


        # make sure the vispy fig takes most space
        self.vispy_fig.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        #self.histlut.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.layer_stack.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # the layer stack
        l.addWidget(self.layer_stack, 0, 1)



    def add_layer(self, layer):
        layer.root = self
        assert layer.name() not in self.layers
        self.layers[layer.name()] = layer
        for item in layer.scatter_items():
            if item is not None:
                item.parent = self.vispy_fig.plot_scatter.view.scene

        self.layer_stack.add_layer(layer)


    def set_range(self):
        self.vispy_fig.set_range()
 