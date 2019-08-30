import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *

from .layer_view_widget import LayerViewWidget
from .layer_ctrl_widget import LayerCtrlWidget
from .settings_widget import SettingsWidget




class LayerViewerObject(object):
    def __init__(self, parent=None):
        #sQtCore.QObject.__init__(self, parent)

        self.m_settings_widget = SettingsWidget()
        self.m_layer_view_widget = LayerViewWidget(settings_widget=self.settings_widget)
        self.m_layer_ctrl_widget = LayerCtrlWidget()


        self.m_layers = dict()
        self.m_exlusive_layer = None

    @property
    def layer_view_widget(self):
        return self.m_layer_view_widget

    @property
    def layer_ctrl_widget(self):
        return self.m_layer_ctrl_widget

    @property
    def settings_widget(self):
        return self.m_settings_widget


    @property
    def view_box(self):
        return self.m_layer_view_widget.view_box

    def addLayer(self, layer, opacity=1.0, visible=True):

        image_item = layer.get_image_item()
        self.m_layer_ctrl_widget.addLayer(layer)
        self.view_box.addItem(image_item)
        self.m_layers[layer.name] = layer
        self.setLayerVisibility(layer.name, bool(visible))
        self.setLayerOpacity(layer.name, float(opacity))
        layer.viewer = self

    def removeAllLayers(self):
        to_rm = list(self.m_layers.keys())
        for l in to_rm:
            self.removeLayer(l)

    def removeLayer(self, layer_name):
        layer = self.m_layers[layer_name]
        self.view_box.removeItem(layer.get_image_item())
        self.m_layer_ctrl_widget.removeLayer(layer)
        del self.m_layers[layer_name]

    def hasLayer(self, layer_name):
        return layer_name in self.m_layers

    def layerVisibility(self, layer_name):
        layer = self.m_layers[layer_name]
        image_item = layer.get_image_item()
        return image_item.isVisible()

    def layerOpacity(self, layer_name):
        layer = self.m_layers[layer_name]
        image_item = layer.get_image_item()
        return image_item.opacity()

    def setLayerVisibility(self, layer_name, visible):
        self.m_layers[layer_name].setVisible(visible)

    def setLayerOpacity(self, layer_name, opacity):
        self.m_layers[layer_name].setOpacity(opacity)

    def updateData(self, layer_name, **kwargs):
        self.m_layers[layer_name].updateData(**kwargs)

    def setData(self, layer_name, **kwargs):
        self.m_layers[layer_name].setData(**kwargs)

    def showAndHideOthers(self, layer_name):

        for ln in self.m_layers.keys():
            layer = self.m_layers[ln]
            if ln == layer_name:
                layer.setVisible(2)
                self.m_exlusive_layer = layer
            else:
                layer.setVisible(False)


class LayerViewerWidget(QtGui.QWidget, LayerViewerObject):
    def __init__(self, gui_style="splitter", parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.m_hbox = QtGui.QHBoxLayout()
        self.setLayout(self.m_hbox)
        self.gui_style = gui_style

        self.layer_viewer_object = LayerViewerObject()


        if gui_style == "dock":

            self.area = DockArea()
            self.m_hbox.addWidget(self.area)
            d_view = Dock("Viewer", size=(500, 500))
            d_ctrl = Dock("Ctrl", size=(200, 500))
            d_view.addWidget(self.m_layer_view_widget)
            d_ctrl.addWidget(self.m_layer_ctrl_widget)
            self.area.addDock(d_view)
            self.area.addDock(d_ctrl, "right", d_view)

        elif gui_style == "splitter":
            self.splitter = QtGui.QSplitter()
            self.m_hbox.addWidget(self.splitter)
            self.splitter.addWidget(self.m_layer_view_widget)
            self.splitter.addWidget(self.m_layer_ctrl_widget)


